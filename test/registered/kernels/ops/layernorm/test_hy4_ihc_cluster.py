"""Thread-block-cluster Hy4 iHC post + pre + RMSNorm against an fp32 reference,
the single-pass Triton kernels and the fp8 group quant op."""

import functools
import sys

import pytest
import torch

from sglang.kernels.jit.utils.compile.parallel import parallel_compile
from sglang.kernels.ops.layernorm import hy4_ihc_cluster as cl
from sglang.kernels.ops.layernorm import hy4_ihc_single_pass as sp
from sglang.kernels.ops.layernorm.hy4_ihc import hy4_ihc_post_pre_norm
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability() not in cl._CAPABILITIES,
    reason="thread-block clusters with st.async: SM90 / SM100 / SM103",
)

MAGNITUDE, NORM_EPS, HC_EPS, RMS_EPS, OUT_EPS = 2.0, 1e-5, 1e-6, 1e-5, 1e-5
BF16_HALF_ULP = 2.0**-8
# Hy4-Flash (hc_mult 4, hidden 3072) and other shapes the kernel claims.
SHAPES = [(4, 3072), (2, 2048), (8, 4096), (1, 1024), (4, 5120)]
VARIANTS = [  # (has_output, out_norm, quant)
    (True, True, False),
    (True, True, True),
    (True, False, False),
    (False, False, False),
    (False, False, True),
]


def _inputs(T, hc, hidden, has_out, out_norm=True, seed=0):
    """Ranges follow the tencent/Hy4-flash-FP8 checkpoint (hc params, norm weights)."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, generator=g, device="cuda")

    def rand(*shape):
        return torch.rand(shape, generator=g, device="cuda")

    stream_scale = (0.75 + 1.25 * rand(hc))[None, :, None]
    shared, indep = randn(T, 1, hidden), randn(T, hc, hidden)
    x = dict(
        output=None,
        residual=(stream_scale * (0.6 * shared + 0.8 * indep)).bfloat16(),
        post=None,
        hc_fn=randn(2 * hc, hc * hidden) * 0.02,
        hc_scale=torch.cat([0.2 + 0.4 * rand(1), 0.05 + 0.25 * rand(1)]),
        hc_base=torch.cat([-1.6 + 1.3 * rand(hc), -2.7 + 2.9 * rand(hc)]),
        magnitude=MAGNITUDE,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
        rms_weight=(0.05 + 0.25 * rand(hidden)).bfloat16(),
        rms_eps=RMS_EPS,
        out_norm_weight=None,
        out_norm_eps=0.0,
    )
    if has_out:
        x["output"] = ((0.25 + 1.75 * rand(T, 1)) * randn(T, hidden)).bfloat16()
        x["post"] = MAGNITUDE * torch.sigmoid(-1.0 + 0.75 * randn(T, hc)) + HC_EPS
        if out_norm:
            x["out_norm_weight"] = (1.0 + 0.1 * randn(hidden)).bfloat16()
            x["out_norm_eps"] = OUT_EPS
    return x


def _rms(x, eps):
    return torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _reference(x):
    """fp32 math, bf16 roundings where the unfused model path stores bf16."""
    residual = x["residual"]
    hc = residual.shape[1]
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
    pre = torch.sigmoid(mix[:, :hc] * scale[0] + base[:hc]) + HC_EPS
    next_post = MAGNITUDE * torch.sigmoid(mix[:, hc:] * scale[1] + base[hc:]) + HC_EPS
    r = (pre[:, :, None] * new_res.float()).sum(1).bfloat16().float()
    y = (r * _rms(r, RMS_EPS) * x["rms_weight"].float()).bfloat16()
    return new_res, y, next_post


def _differs(actual, ref):
    a, r = actual.float(), ref.float()
    return (a - r).abs() > BF16_HALF_ULP * r.abs()


def _assert_close(actual, ref, has_out):
    """The tolerances of test_hy4_ihc_single_pass.py."""
    new_res, y, next_post = actual[:3]
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


def _assert_fp8_matches_quant_op(y, y_fp8):
    """Bitwise the pair linear_gate's DeepGEMM path would have quantized itself."""
    q_ref, s_ref = sglang_per_token_group_quant_fp8(
        y, 128, column_major_scales=True, scale_tma_aligned=True
    )
    q, s = y_fp8
    assert s.stride() == s_ref.stride() and s.shape == s_ref.shape
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s.view(torch.int32), s_ref.view(torch.int32))


def _call(x, quant=False, config=None):
    return cl.hy4_ihc_cluster(**x, quant=quant, config=config)


def _variant(hc, hidden, has_out, out_norm, quant):
    return cl._Variant(hc, hidden, has_out, out_norm, quant)


def _max_tokens(hc, hidden, has_out, out_norm, quant):
    return len(cl._plan(_variant(hc, hidden, has_out, out_norm, quant), 0)) - 1


@pytest.fixture(scope="module", autouse=True)
def _compiled():
    jobs = [
        functools.partial(cl._module, _variant(hc, hidden, *v), c)
        for hc, hidden in SHAPES
        for v in VARIANTS
        for c in cl._candidate_configs(hc, hidden)
    ]
    parallel_compile(jobs)


def _ts(max_t):
    """Decode sizes, DFly verify sizes (6 x C), tails of every tokens-per-cluster
    and the largest count one wave holds."""
    ts = {1, 2, 3, 5, 6, 8, 9, 16, 33, 48, 61, 64, 78, 90, 114, max_t}
    return sorted(t for t in ts if t <= max_t)


@pytest.mark.parametrize("hc,hidden", SHAPES)
@pytest.mark.parametrize("has_out,out_norm,quant", VARIANTS)
def test_matches_reference(hc, hidden, has_out, out_norm, quant):
    for T in _ts(_max_tokens(hc, hidden, has_out, out_norm, quant)):
        x = _inputs(T, hc, hidden, has_out, out_norm, seed=T)
        got = _call(x, quant=quant)
        _assert_close(got, _reference(x), has_out)
        if quant:
            _assert_fp8_matches_quant_op(got[1], got[3])
        else:
            assert got[3] is None


@pytest.mark.parametrize("hc,hidden", [(4, 3072), (8, 4096)])
def test_rows_independent_of_config_and_batch(hc, hidden):
    """A group's partial does not depend on the cluster size or the tokens per
    cluster, so every config, and every batch a row sits in, gives the same bits."""
    T = 61
    x = _inputs(T, hc, hidden, has_out=True, seed=7)
    # Group magnitudes 2^-6..2^6 make both row sums depend on their summation
    # order, so a config-dependent order flips bf16 roundings of y.
    g = torch.Generator(device="cuda").manual_seed(8)
    groups = torch.randint(-6, 7, (hidden // 128,), generator=g, device="cuda")
    col_scale = (2.0 ** groups.float()).repeat_interleave(128)
    x["residual"] = (x["residual"].float() * col_scale).bfloat16()
    x["output"] = (x["output"].float() * col_scale).bfloat16()
    expected = _call(x, quant=True, config=cl._candidate_configs(hc, hidden)[0])
    for config in cl._candidate_configs(hc, hidden)[1:]:
        got = _call(x, quant=True, config=config)
        for a, b in zip(got[:3] + got[3], expected[:3] + expected[3]):
            assert torch.equal(a, b), config
    row = {
        k: (
            v[5:6]
            if isinstance(v, torch.Tensor) and k in ("output", "residual", "post")
            else v
        )
        for k, v in x.items()
    }
    single = _call(row, quant=True)
    for a, b in zip(single[:3], expected[:3]):
        assert torch.equal(a, b[5:6])


def test_repeat_runs_bitwise_equal():
    x = _inputs(48, 4, 3072, has_out=True, seed=3)
    first = _call(x, quant=True)
    for _ in range(20):
        again = _call(x, quant=True)
        for a, b in zip(first[:3] + first[3], again[:3] + again[3]):
            assert torch.equal(a, b)


@pytest.mark.parametrize("T", [1, 6, 47, 64])
def test_matches_single_pass_triton(T):
    """Against the kernels it replaces at these token counts (same rounding points,
    different summation order): mostly the same bf16 values."""
    x = _inputs(T, 4, 3072, has_out=True, seed=11)
    sp.prepare_hy4_ihc_single_pass(
        device=torch.device("cuda"), out_norm_variants=(True,)
    )
    got = _call(x)
    want = sp.hy4_ihc_single_pass(**x)
    assert _differs(got[0], want[0]).float().mean() <= 1e-3
    assert _differs(got[1], want[1]).float().mean() <= 5e-3
    assert ((got[2] - want[2]).abs() <= 1e-4).all()


@pytest.mark.parametrize("T,has_out", [(1, True), (78, True), (48, False)])
def test_graph_replays_match_eager(T, has_out):
    static = _inputs(T, 4, 3072, has_out=has_out, seed=1)
    _call(static, quant=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _call(static, quant=True)
    for seed in (2, 3):
        fresh = _inputs(T, 4, 3072, has_out=has_out, seed=seed)
        for k, v in fresh.items():
            if isinstance(v, torch.Tensor):
                static[k].copy_(v)
        graph.replay()
        expected = _call(fresh, quant=True)
        for got, want in zip(out[:3] + out[3], expected[:3] + expected[3]):
            assert torch.equal(got, want)


def test_dispatcher_hands_larger_batches_to_triton():
    """hy4_ihc_post_pre_norm takes the cluster kernel up to its one-wave limit and
    the single-pass Triton kernels above it, where the fp8 pair is None."""
    cl.prepare_hy4_ihc_cluster(
        hc_mult=4,
        hidden_size=3072,
        device=torch.device("cuda"),
        variants=((True, True, True),),
    )
    max_t = _max_tokens(4, 3072, True, True, True)
    assert max_t >= 64
    for T, fused in ((max_t, True), (max_t + 1, False)):
        x = _inputs(T, 4, 3072, has_out=True, seed=T)
        args = [x[k] for k in ("output", "residual", "post", "hc_fn", "hc_scale")]
        args += [x[k] for k in ("hc_base", "magnitude", "norm_eps", "hc_eps")]
        args += [
            x[k] for k in ("rms_weight", "rms_eps", "out_norm_weight", "out_norm_eps")
        ]
        got = hy4_ihc_post_pre_norm(*args, single_pass=True, cluster=True, quant=True)
        assert (got[3] is not None) == fused
        _assert_close(got, _reference(x), has_out=True)
        assert len(hy4_ihc_post_pre_norm(*args, single_pass=True, cluster=True)) == 3


def test_no_compile_after_prepare():
    """prepare_hy4_ihc_cluster builds every config the dispatch table can pick;
    serving-time token counts must not load another module."""
    variants = ((True, True, True), (True, True, False), (False, False, True))
    cl.prepare_hy4_ihc_cluster(
        hc_mult=4, hidden_size=3072, device=torch.device("cuda"), variants=variants
    )
    real = cl.load_jit

    def no_compile(*args, **kwargs):
        raise AssertionError(f"compiled after prepare: {args}")

    cl.load_jit = no_compile
    try:
        for has_out, out_norm, quant in variants:
            for T in range(1, _max_tokens(4, 3072, has_out, out_norm, quant) + 1):
                _call(_inputs(T, 4, 3072, has_out, out_norm, seed=T), quant=quant)
    finally:
        cl.load_jit = real


def test_supported_shapes_only():
    ok = dict(hc_mult=4, hidden_size=3072, dtype=torch.bfloat16)
    assert cl.hy4_ihc_cluster_supported(**ok)
    for change in (
        dict(hidden_size=3000),  # not a multiple of 128
        dict(hidden_size=128 * 13),  # 13 groups: no cluster size divides them
        dict(hc_mult=3),
        dict(dtype=torch.float16),
    ):
        assert not cl.hy4_ihc_cluster_supported(**{**ok, **change})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
