"""CPU unit tests for the Cake ``moe_fp8_grouped`` route of the DeepGEMM MoE runner.

The FlashInfer adapter, DeepGEMM and the SwiGLU/quant kernels are replaced by
fakes; no GPU is needed.  Covered: route switch off -> DeepGEMM path (Cake API
never touched); route on + admission -> prepared Cake runners launched once per
shape and reused; first sight of a shape inside CUDA-graph capture -> DeepGEMM
fallback; the UE8M0 scale unpack and the ``-1`` padding fill the route relies on.
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("triton")
import torch

import sglang.kernels.ops.moe.dsv4 as dsv4
import sglang.kernels.ops.moe.ep_moe_kernels as ep_moe_kernels
import sglang.kernels.ops.quantization.fp8_kernel as fp8_kernel
from sglang.kernels.cake_kernels import _routes
from sglang.srt.layers.moe.moe_runner import deep_gemm as dg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

M, K, N, G = 256, 512, 256, 4
H = N // 2
FP8 = torch.float8_e4m3fn
CAKE_FILL = 1.0  # fake Cake runners write 1.0
DEEPGEMM_FILL = 2.0  # fake DeepGEMM writes 2.0


def _fill(tensor, value):
    if tensor.dtype == FP8:
        tensor.view(torch.uint8).fill_(int(value))
    else:
        tensor.fill_(value)


class _FakeRunner:
    def __init__(self, outputs, log, name):
        self.outputs = outputs
        self.log = log
        self.name = name
        self.launches = 0

    def launch(self):
        self.launches += 1
        self.log.append(self.name)
        for out in self.outputs:
            _fill(out, CAKE_FILL)
        return self.outputs[0] if len(self.outputs) == 1 else tuple(self.outputs)


class _FakeApi:
    """Stands in for the Cake adapter ``supports_*`` + ops/gemm/cake.py wrappers."""

    def __init__(self, *, plain=True, fused=True):
        self.plain, self.fused = plain, fused
        self.log = []
        self.prepared = []

    def supports_plain(self, a, b, a_scale, b_scale, m_indices, out=None):
        assert a.dtype == FP8 and a_scale.dtype == torch.float32
        assert b_scale.dtype == torch.float32
        assert tuple(b_scale.shape) == (
            b.shape[0],
            b.shape[1] // 128,
            b.shape[2] // 128,
        )
        assert tuple(out.shape) == (a.shape[0], b.shape[1])
        return self.plain

    def supports_fused(self, a, b, a_scale, b_scale, m_indices, out_q=None, out_s=None):
        assert tuple(out_q.shape) == (a.shape[0], b.shape[1] // 2)
        assert tuple(out_s.shape) == (a.shape[0], b.shape[1] // 256)
        return self.fused

    def prepare_plain(self, a, b, a_scale, b_scale, m_indices, out=None, **_):
        runner = _FakeRunner([out], self.log, f"plain:{tuple(b.shape)}")
        self.prepared.append(runner)
        return runner

    def prepare_fused(
        self, a, b, a_scale, b_scale, m_indices, out_q=None, out_s=None, **_
    ):
        runner = _FakeRunner([out_q, out_s], self.log, f"fused:{tuple(b.shape)}")
        self.prepared.append(runner)
        return runner


def _pack_ue8m0(exponents: torch.Tensor) -> torch.Tensor:
    """(rows, kg) uint8 exponents -> DeepGEMM-style (rows, ceil(kg/4)) int32, mn-major strides."""
    rows, kg = exponents.shape
    padded = torch.zeros((rows, 4 * ((kg + 3) // 4)), dtype=torch.uint8)
    padded[:, :kg] = exponents
    packed = padded.view(torch.int32)  # (rows, ceil(kg/4))
    return packed.t().contiguous().t()


def _padded_m_indices():
    # Two 128-row expert blocks: 100 valid + 28 padding, 120 valid + 8 padding.
    m_indices = torch.full((M,), -1, dtype=torch.int32)
    m_indices[:100] = 0
    m_indices[128:248] = 2
    return m_indices


def _weights():
    kg = K // 128
    w13_scale = _pack_ue8m0(
        torch.randint(120, 130, (G * N, kg), dtype=torch.uint8)
    ).view(G, N, -1)
    return SimpleNamespace(
        w13_weight=torch.zeros((G, N, K), dtype=torch.bfloat16).to(FP8),
        w13_scale=w13_scale,
        w2_weight=torch.zeros((G, K, H), dtype=torch.bfloat16).to(FP8),
        w2_scale=torch.rand((G, K // 128, H // 128), dtype=torch.float32),
    )


def _activations():
    kg = K // 128
    return SimpleNamespace(
        hidden_states=torch.zeros((M, K), dtype=torch.bfloat16).to(FP8),
        hidden_states_scale=_pack_ue8m0(
            torch.randint(120, 130, (M, kg), dtype=torch.uint8)
        ),
        m_indices=_padded_m_indices(),
    )


def _request(weights=None, *, layout_alignment=128, **overrides):
    weights = weights or _weights()
    acts = _activations()
    fields = dict(
        hidden_states=acts.hidden_states,
        hidden_states_scale=acts.hidden_states_scale,
        m_indices=acts.m_indices,
        w13_weight=weights.w13_weight,
        w13_scale=weights.w13_scale,
        w2_weight=weights.w2_weight,
        w2_scale=weights.w2_scale,
        activation="silu",
        swiglu_limit=None,
        silu_mul_keep_fp32=False,
        use_swizzle=False,
        use_mxfp8=False,
        is_fp4_experts=False,
        activation_scale_block_size=128,
        layout_alignment=layout_alignment,
    )
    fields.update(overrides)
    return dg._CakeContigRequest(**fields)


def _allocate_output():
    return torch.empty((M, K), dtype=torch.bfloat16)


@pytest.fixture
def api(monkeypatch):
    fake = _FakeApi()
    monkeypatch.setattr(dg, "_cake_grouped_fp8_api", lambda: fake)
    monkeypatch.setattr(dg, "_cake_stream_capturing", lambda: False)
    dg._CAKE_CONTIG_FP8.reset_for_tests()
    yield fake
    dg._CAKE_CONTIG_FP8.reset_for_tests()


@pytest.fixture
def fake_silu_quant(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        _fill(kwargs["output"], CAKE_FILL)
        kwargs["output_scale"].fill_(CAKE_FILL)

    monkeypatch.setattr(dsv4, "silu_and_mul_contig_post_quant", fake)
    return calls


@pytest.fixture
def fake_deepgemm(monkeypatch):
    """Fakes for everything the DeepGEMM contiguous path launches on the GPU."""
    calls = []

    def grouped_gemm(lhs, rhs, out, m_indices, recipe_a=None, recipe_b=None):
        calls.append(tuple(rhs[0].shape))
        out.fill_(DEEPGEMM_FILL)

    def legacy_silu(x, out):
        out.fill_(DEEPGEMM_FILL)

    def quant(x, group_size, **kwargs):
        return x.to(FP8), torch.ones((x.shape[0], x.shape[1] // group_size))

    monkeypatch.setattr(
        dg.deep_gemm_wrapper, "grouped_gemm_nt_f8f8bf16_contig", grouped_gemm
    )
    monkeypatch.setattr(dg, "_legacy_silu_and_mul", legacy_silu)
    monkeypatch.setattr(dg, "dispose_tensor", lambda x: None)
    monkeypatch.setattr(fp8_kernel, "sglang_per_token_group_quant_fp8", quant)
    monkeypatch.setattr(ep_moe_kernels, "tma_align_input_scale", lambda s: s)
    return calls


@pytest.fixture(autouse=True)
def _routes_reset():
    _routes.reset_cache_for_tests()
    yield
    _routes.reset_cache_for_tests()


def _run_contiguous(weights, acts):
    """Call the real ``_run_contiguous_gemm`` with a minimal stand-in for ``self``."""
    runner_self = SimpleNamespace(
        config=SimpleNamespace(
            activation="silu",
            silu_mul_keep_fp32=False,
            gemm1_alpha=None,
            gemm1_clamp_limit=None,
        ),
        swiglu_limit=None,
        use_swizzle=False,
        _allocate_down_output=lambda m, k, device: torch.empty(
            (m, k), dtype=torch.bfloat16, device=device
        ),
    )
    runner_input = dg.DeepGemmRunnerInput(
        hidden_states=acts.hidden_states,
        hidden_states_scale=acts.hidden_states_scale,
        use_masked_gemm=False,
        m_indices=acts.m_indices,
        hidden_states_scale_tma_aligned=True,
        activation_scale_block_size=128,
    )
    quant_info = dg.DeepGemmMoeQuantInfo(
        w13_weight=weights.w13_weight,
        w2_weight=weights.w2_weight,
        use_fp8=True,
        w13_scale=weights.w13_scale,
        w2_scale=weights.w2_scale,
        block_shape=[128, 128],
    )
    running_state = {
        "all_tokens": M,
        "hidden_states_device": acts.hidden_states.device,
        "hidden_states_dtype": torch.bfloat16,
        "hidden_states_shape": (M, K),
        "contiguous_layout_alignment": 128,
    }
    return dg.DeepGemmRunnerCore._run_contiguous_gemm(
        runner_self, runner_input, quant_info, running_state
    )


def test_route_off_keeps_deepgemm_path(api, fake_deepgemm):
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(_routes.ENV_VAR, None)
        out = _run_contiguous(_weights(), _activations())
    assert torch.all(out == DEEPGEMM_FILL)
    assert fake_deepgemm == [(G, N, K), (G, K, H)]
    assert api.prepared == [] and api.log == []


def test_route_on_runs_prepared_cake_runners(api, fake_deepgemm):
    weights = _weights()
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: dg._CAKE_ROUTE}):
        out = _run_contiguous(weights, _activations())
        out2 = _run_contiguous(weights, _activations())
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (M, K)
    assert torch.all(out == CAKE_FILL) and torch.all(out2 == CAKE_FILL)
    assert fake_deepgemm == []
    # Fused gate_up+SwiGLU+quant then the down GEMM, each prepared once and
    # launched per call; the result is a fresh copy of the static buffer.
    assert api.log == [f"fused:{(G, N, K)}", f"plain:{(G, K, H)}"] * 2
    assert len(api.prepared) == 2 and all(r.launches == 2 for r in api.prepared)
    assert out2 is not out
    plan = next(iter(dg._CAKE_CONTIG_FP8._plans.values()))
    assert plan.fused
    # Padding rows (-1) were mapped onto the preceding expert, never negative.
    filled = plan.buffers["m_indices"]
    assert torch.equal(filled[:128], torch.zeros(128, dtype=torch.int32))
    assert torch.equal(filled[128:], torch.full((128,), 2, dtype=torch.int32))
    # Packed UE8M0 activation scales were unpacked to exact powers of two.
    a_scale = plan.buffers["a_scale"]
    assert torch.all(torch.log2(a_scale) == torch.log2(a_scale).round())


def test_fused_not_admitted_uses_plain_gemm_plus_swiglu(api, fake_silu_quant):
    api.fused = False
    out = dg._CAKE_CONTIG_FP8.try_run(_request(), _allocate_output)
    assert out is not None and torch.all(out == CAKE_FILL)
    assert api.log == [f"plain:{(G, N, K)}", f"plain:{(G, K, H)}"]
    assert len(fake_silu_quant) == 1
    call = fake_silu_quant[0]
    assert call["scale_ue8m0"] is False and call["transposed"] is False
    assert tuple(call["output_scale"].shape) == (M, H // 128)
    assert tuple(call["input"].shape) == (M, N)


def test_unaligned_expert_boundaries_disable_fused(api, fake_silu_quant):
    out = dg._CAKE_CONTIG_FP8.try_run(_request(layout_alignment=64), _allocate_output)
    assert out is not None
    assert api.log[0] == f"plain:{(G, N, K)}"
    assert not next(iter(dg._CAKE_CONTIG_FP8._plans.values())).fused


def test_capturing_stream_with_unprepared_shape_falls_back(api, monkeypatch):
    weights = _weights()
    monkeypatch.setattr(dg, "_cake_stream_capturing", lambda: True)
    assert dg._CAKE_CONTIG_FP8.try_run(_request(weights), _allocate_output) is None
    assert api.prepared == [] and api.log == []
    # Warm the shape up eagerly, then the same shape is served inside capture.
    monkeypatch.setattr(dg, "_cake_stream_capturing", lambda: False)
    assert dg._CAKE_CONTIG_FP8.try_run(_request(weights), _allocate_output) is not None
    monkeypatch.setattr(dg, "_cake_stream_capturing", lambda: True)
    assert dg._CAKE_CONTIG_FP8.try_run(_request(weights), _allocate_output) is not None
    assert len(api.prepared) == 2


def test_capture_fallback_runs_deepgemm(api, fake_deepgemm, monkeypatch):
    monkeypatch.setattr(dg, "_cake_stream_capturing", lambda: True)
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: dg._CAKE_ROUTE}):
        out = _run_contiguous(_weights(), _activations())
    assert torch.all(out == DEEPGEMM_FILL)
    assert fake_deepgemm == [(G, N, K), (G, K, H)]
    assert api.prepared == []


def test_down_gemm_not_admitted_is_all_or_nothing(api):
    api.plain = False
    weights = _weights()
    assert dg._CAKE_CONTIG_FP8.try_run(_request(weights), _allocate_output) is None
    assert api.prepared == []
    # Negative admission is cached per shape/weights: no re-probe on the next call.
    api.plain = True
    assert dg._CAKE_CONTIG_FP8.try_run(_request(weights), _allocate_output) is None


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"activation": "situ"}, "activation"),
        ({"use_mxfp8": True}, "mxfp8"),
        ({"w13_weight": torch.zeros((G, 384, K)).to(FP8)}, "inconsistent"),
        ({"activation_scale_block_size": 32}, "scale block"),
    ],
)
def test_static_rejections_never_touch_the_api(api, caplog, overrides, reason):
    with caplog.at_level("INFO", logger=dg.logger.name):
        assert (
            dg._CAKE_CONTIG_FP8.try_run(_request(**overrides), _allocate_output) is None
        )
    assert api.prepared == []
    assert any(reason in rec.getMessage() for rec in caplog.records)


def test_ue8m0_unpack_matches_powers_of_two():
    exps = torch.randint(100, 150, (3, 5), dtype=torch.uint8)
    dst = torch.empty((3, 5), dtype=torch.float32)
    dg._cake_unpack_ue8m0_into(dst, _pack_ue8m0(exps))
    torch.testing.assert_close(dst, torch.exp2(exps.float() - 127.0))


def test_weight_scale_fp32_from_row_repeated_packed_layout():
    kg = K // 128
    block = torch.randint(100, 150, (G, N // 128, kg), dtype=torch.uint8)
    rows = block.repeat_interleave(128, dim=1).reshape(G * N, kg)
    packed = _pack_ue8m0(rows).view(G, N, -1)
    try:
        got = dg._cake_weight_scale_fp32(packed, G, N, K)
        assert got is not None and tuple(got.shape) == (G, N // 128, kg)
        torch.testing.assert_close(got, torch.exp2(block.float() - 127.0))
        assert dg._cake_weight_scale_fp32(packed, G, N, K) is got  # cached
        plain = torch.rand((G, N // 128, kg))
        assert dg._cake_weight_scale_fp32(plain, G, N, K) is plain
        assert dg._cake_weight_scale_fp32(torch.rand((G, N, kg)), G, N, K) is None
    finally:
        dg._cake_weight_scale_cache.clear()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
