"""SM90 single-pass Hy4-Flash iHC post + pre + RMSNorm against an fp32 reference."""

import sys

import pytest
import torch

from sglang.kernels.ops.layernorm import hy4_ihc_single_pass as sp
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="the single-pass kernels are SM90-only",
)

HIDDEN = sp.HIDDEN_SIZE
HC = sp.HC_MULT
MAGNITUDE, NORM_EPS, HC_EPS, RMS_EPS, OUT_EPS = 2.0, 1e-5, 1e-6, 1e-5, 1e-5
BF16_HALF_ULP = 2.0**-8


def _inputs(T: int, has_out: bool, out_norm: bool = True, seed: int = 0) -> dict:
    """Ranges follow the tencent/Hy4-flash-FP8 checkpoint (hc params, norm weights)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, generator=g, device="cuda")

    def rand(*shape):
        return torch.rand(shape, generator=g, device="cuda")

    stream_scale = (0.75 + 1.25 * rand(HC))[None, :, None]
    shared, indep = randn(T, 1, HIDDEN), randn(T, HC, HIDDEN)
    x = dict(
        output=None,
        residual=(stream_scale * (0.6 * shared + 0.8 * indep)).bfloat16(),
        post=None,
        hc_fn=randn(2 * HC, HC * HIDDEN) * 0.02,
        hc_scale=torch.cat([0.2 + 0.4 * rand(1), 0.05 + 0.25 * rand(1)]),
        hc_base=torch.cat([-1.6 + 1.3 * rand(HC), -2.7 + 2.9 * rand(HC)]),
        magnitude=MAGNITUDE,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        rms_weight=(0.05 + 0.25 * rand(HIDDEN)).bfloat16(),
        rms_eps=RMS_EPS,
        out_norm_weight=None,
        out_norm_eps=0.0,
    )
    if has_out:
        x["output"] = ((0.25 + 1.75 * rand(T, 1)) * randn(T, HIDDEN)).bfloat16()
        x["post"] = MAGNITUDE * torch.sigmoid(-1.0 + 0.75 * randn(T, HC)) + HC_EPS
        if out_norm:
            x["out_norm_weight"] = (1.0 + 0.1 * randn(HIDDEN)).bfloat16()
            x["out_norm_eps"] = OUT_EPS
    return x


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _reference(x: dict):
    """fp32 math, bf16 roundings where the unfused model path stores bf16."""
    residual = x["residual"]
    if x["output"] is None:
        new_res = residual
    else:
        o = x["output"].float()
        if x["out_norm_weight"] is not None:
            o = (o * _rms(o, OUT_EPS) * x["out_norm_weight"].float()).bfloat16().float()
        new_res = (x["post"][:, :, None] * o[:, None, :] + residual.float()).bfloat16()
    flat = new_res.flatten(1).float()
    mix = (flat.double() @ x["hc_fn"].double().t()).float() * _rms(flat, NORM_EPS)
    scale, base = x["hc_scale"], x["hc_base"]
    pre = torch.sigmoid(mix[:, :HC] * scale[0] + base[:HC]) + HC_EPS
    next_post = MAGNITUDE * torch.sigmoid(mix[:, HC:] * scale[1] + base[HC:]) + HC_EPS
    r = (pre[:, :, None] * new_res.float()).sum(1).bfloat16().float()
    y = (r * _rms(r, RMS_EPS) * x["rms_weight"].float()).bfloat16()
    return new_res, y, next_post


def _differs(actual: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """True where the bf16 values differ by any ulp."""
    a, r = actual.float(), ref.float()
    return (a - r).abs() > BF16_HALF_ULP * r.abs()


def _assert_close(actual, ref, has_out: bool):
    new_res, y, next_post = actual
    ref_new_res, ref_y, ref_post = ref
    if has_out:
        # The out-norm reduction order may flip single bf16 roundings of o.
        assert _differs(new_res, ref_new_res).float().mean() <= 1e-3
        err = (new_res.float() - ref_new_res.float()).abs()
        assert (err <= 2.0**-5 + 2.0**-7 * ref_new_res.float().abs()).all()
    else:
        assert new_res is ref_new_res
    assert _differs(y, ref_y).float().mean() <= 5e-3
    y_err = (y.float() - ref_y.float()).abs()
    y_tol = 1.6e-2 * ref_y.float().abs() + 2.0**-7 * ref_y.float().abs().max()
    assert (y_err <= y_tol).all()
    post_err = (next_post - ref_post).abs()
    assert (post_err <= 1e-3).all()
    assert (post_err > 1e-4).sum() <= max(1, int(1e-3 * post_err.numel()))


def _call(x: dict):
    return sp.hy4_ihc_single_pass(**x)


@pytest.fixture(scope="module", autouse=True)
def _prepared():
    # As at model init, which runs under the model's default dtype.
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        sp.prepare_hy4_ihc_single_pass(
            device=torch.device("cuda"), out_norm_variants=(True, False)
        )
    finally:
        torch.set_default_dtype(default_dtype)


# Split-K block sizes 128 (T <= 8) and 256, the split-K / tiled crossovers (32
# without output, 64 with), every tiled config, and T that are not tile multiples.
# Decode buckets 40/48 without output and DFly verify 61..78, 90, 114 are served
# shapes.
_T_OUT = [1, 2, 8, 9, 33, 48, 64, 65, 78, 90, 114, 129, 257, 385, 769, 2049, 2437]
_T_NOOUT = [1, 8, 9, 32, 33, 40, 48, 61, 78, 129, 257, 769, 2437]


@pytest.mark.parametrize("T", _T_OUT)
@pytest.mark.parametrize("out_norm", [True, False])
def test_with_output_matches_reference(T, out_norm):
    x = _inputs(T, has_out=True, out_norm=out_norm, seed=T)
    _assert_close(_call(x), _reference(x), has_out=True)


@pytest.mark.parametrize("T", _T_NOOUT)
def test_without_output_matches_reference(T):
    x = _inputs(T, has_out=False, seed=T)
    _assert_close(_call(x), _reference(x), has_out=False)


@pytest.mark.parametrize(
    "T,has_out", [(1, True), (64, True), (32, False), (48, False), (78, True)]
)
def test_graph_replays_match_eager(T, has_out):
    """The split-K tickets reset themselves: every replay of a captured call, and
    every eager call after it, finishes each token exactly once."""
    static = _inputs(T, has_out=has_out, seed=1)
    _call(static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _call(static)
    for seed in (2, 3):
        fresh = _inputs(T, has_out=has_out, seed=seed)
        for k, v in fresh.items():
            if isinstance(v, torch.Tensor):
                static[k].copy_(v)
        graph.replay()
        expected = _call(fresh)
        for got, want in zip(out, expected):
            assert torch.equal(got, want)


def _compiled_variants() -> int:
    kernels = (
        sp._ihc_splitk_kernel,
        sp._out_inv_rms_kernel,
        sp._ihc_dot_partials_kernel,
        sp._ihc_stage2_kernel,
    )
    return sum(len(k.device_caches[torch.cuda.current_device()][0]) for k in kernels)


def test_no_compile_after_prepare():
    """prepare_hy4_ihc_single_pass compiles every variant the dispatcher picks;
    serving-time token counts must not add a specialization."""
    before = _compiled_variants()
    for T in (1, 3, 16, 17, 31, 32, 47, 63, 64, 96, 127, 200, 300, 600, 1500, 3000):
        for has_out, out_norm in ((False, False), (True, True), (True, False)):
            _call(_inputs(T, has_out=has_out, out_norm=out_norm, seed=T))
    assert _compiled_variants() == before


def test_ticket_buffer_not_allocated_in_capture():
    """A zeroing during capture is only recorded into the graph, not executed, so
    a ticket buffer first allocated there would be uninitialized for eager calls
    and other graphs."""
    sp._tickets_by_device.clear()
    x = _inputs(1, has_out=True)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(AssertionError, match="outside CUDA-graph capture"):
        with torch.cuda.graph(graph):
            _call(x)
    # Allocated eagerly through the index-less device, as model init does.
    sp._tickets(torch.device("cuda"))
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        _call(x)


def test_supported_shapes_only():
    ok = dict(hc_mult=HC, hidden_size=HIDDEN, dtype=torch.bfloat16)
    assert sp.hy4_ihc_single_pass_supported(**ok)
    for change in (dict(hidden_size=4096), dict(hc_mult=3), dict(dtype=torch.float16)):
        assert not sp.hy4_ihc_single_pass_supported(**{**ok, **change})


def test_out_norm_without_output_is_rejected():
    x = _inputs(1, has_out=False)
    x["out_norm_weight"] = x["rms_weight"]
    with pytest.raises(ValueError, match="output RMSNorm"):
        _call(x)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
