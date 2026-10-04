"""CPU unit tests for the Cake ``moe_fp8_grouped`` route of the DeepGEMM MoE runner.

The FlashInfer adapter, DeepGEMM and the SwiGLU/quant kernel are replaced by
fakes; no GPU is needed.  Covered: route switch off -> DeepGEMM path (Cake API
never touched); route on -> one prepared runner pair per (per-token geometry,
weights, alignment), rebound to the caller's tensors on every call with the
packed UE8M0 int32 scales and the ``-1`` padding rows passed through untouched
and the output written in place; the down GEMM input quantized with the
DeepGEMM path's UE8M0 MN-major scale layout; the FP32 scale family
(``fill_padding``); mixed scale dtypes, non-32-multiple alignments and static
shape mismatches -> DeepGEMM; first sight of a shape inside CUDA-graph capture
-> DeepGEMM fallback; the compact-layout alignment policy; the adapter's
block-scaled contract probe and scale admission.
"""

import os
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("triton")
import torch

import sglang.kernels.ops.moe.dsv4 as dsv4
import sglang.kernels.ops.moe.ep_moe_kernels as ep_moe_kernels
import sglang.kernels.ops.quantization.fp8_kernel as fp8_kernel
from sglang.kernels.cake_kernels import _routes
from sglang.kernels.cake_kernels import gemm_grouped_fp8 as adapter
from sglang.srt.layers.moe.moe_runner import deep_gemm as dg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a-test-cpu")

M, K, N, G = 256, 512, 256, 4
H = N // 2
FP8 = torch.float8_e4m3fn
CAKE_FILL = 1.0  # fake Cake runners write 1.0 into valid rows
DEEPGEMM_FILL = 2.0  # fake DeepGEMM writes 2.0


def _fill(tensor, value):
    if tensor.dtype == FP8:
        tensor.view(torch.uint8).fill_(int(value))
    else:
        tensor.fill_(value)


def _scale_cols(k):
    return -(-(k // 128) // 4)


class _FakeRunner:
    """Prepared-runner stand-in: records the operands bound at prepare and at
    every ``launch(a=, a_scale=, m_indices=, out=)``, enforcing the contract
    that rebinding keeps shape / dtype / strides."""

    def __init__(self, name, log, bound, *, fill_padding, alignment):
        self.name, self.log, self.bound = name, log, bound
        self.fill_padding, self.alignment = fill_padding, alignment
        self.launches = []
        self.released = False

    def release_prepared_operands(self):
        self.released = True

    def launch(self, a=None, a_scale=None, m_indices=None, out=None):
        assert self.released, "plan runners must release their prepared operands"
        rebound = dict(a=a, a_scale=a_scale, m_indices=m_indices, out=out)
        for key, tensor in rebound.items():
            prepared = self.bound[key]
            assert tensor.shape == prepared.shape and tensor.dtype == prepared.dtype
            assert tensor.stride() == prepared.stride()
        self.launches.append(rebound)
        self.log.append(self.name)
        out[m_indices >= 0] = CAKE_FILL  # ``-1`` rows are skipped natively
        return out


class _FakeApi:
    """Stands in for the Cake adapter ``supports_*`` / device probe and the
    ops/gemm/cake.py prepare wrapper."""

    def __init__(self, *, plain=True, available=True):
        self.plain, self.available = plain, available
        self.log = []
        self.prepared = []
        self.supports_calls = []

    def device_available(self, device_index):
        return self.available

    def supports_plain(
        self, a, b, a_scale, b_scale, m_indices, out=None, *, fill_padding, alignment
    ):
        self.supports_calls.append(dict(fill_padding=fill_padding, alignment=alignment))
        assert a.dtype == FP8 and m_indices.dtype == torch.int32
        assert a_scale.dtype == b_scale.dtype
        m, k = a.shape
        groups, n, _ = b.shape
        if a_scale.dtype == torch.int32:
            cols = _scale_cols(k)
            assert tuple(a_scale.shape) == (m, cols)
            assert b_scale.shape[0] == groups and b_scale.shape[2] == cols
            assert b_scale.shape[1] in (n, n // 128)
            assert not fill_padding
        else:
            assert tuple(a_scale.shape) == (m, k // 128)
            assert tuple(b_scale.shape) == (groups, n // 128, k // 128)
            assert fill_padding
        assert out is not None and tuple(out.shape) == (m, n)
        return self.plain

    def prepare_plain(
        self,
        a,
        b,
        a_scale,
        b_scale,
        m_indices,
        out=None,
        *,
        validate_indices=False,
        fill_padding=False,
        alignment=128,
    ):
        runner = _FakeRunner(
            f"plain:{tuple(b.shape)}",
            self.log,
            dict(a=a, a_scale=a_scale, m_indices=m_indices, out=out),
            fill_padding=fill_padding,
            alignment=alignment,
        )
        self.prepared.append(runner)
        return runner


def _packed_activation_scale(rows, k):
    """DeepGEMM dispatcher layout: int32 ``(rows, ceil(K/512))`` with strides ``(1, rows)``."""
    return torch.randint(0, 2**31 - 1, (_scale_cols(k), rows), dtype=torch.int32).t()


def _packed_weight_scale(groups, n, k):
    """``transform_scale_ue8m0`` layout: int32 ``(G, N, ceil(K/512))`` strides ``(N*cols, 1, N)``."""
    return torch.randint(
        0, 2**31 - 1, (groups, _scale_cols(k), n), dtype=torch.int32
    ).permute(0, 2, 1)


def _padded_m_indices(rows=M):
    # Two 128-row expert blocks: 100 valid + 28 padding, 120 valid + 8 padding.
    m_indices = torch.full((rows,), -1, dtype=torch.int32)
    half = rows // 2
    m_indices[: half - 28] = 0
    m_indices[half : rows - 8] = 2
    return m_indices


def _weights(*, fp32_scales=False):
    if fp32_scales:
        w13_scale = torch.rand((G, N // 128, K // 128), dtype=torch.float32)
        w2_scale = torch.rand((G, K // 128, H // 128), dtype=torch.float32)
    else:
        w13_scale = _packed_weight_scale(G, N, K)
        w2_scale = _packed_weight_scale(G, K, H)
    return SimpleNamespace(
        w13_weight=torch.zeros((G, N, K), dtype=torch.bfloat16).to(FP8),
        w13_scale=w13_scale,
        w2_weight=torch.zeros((G, K, H), dtype=torch.bfloat16).to(FP8),
        w2_scale=w2_scale,
    )


def _activations(rows=M, *, fp32_scales=False):
    if fp32_scales:
        scale = torch.rand((rows, K // 128), dtype=torch.float32)
    else:
        scale = _packed_activation_scale(rows, K)
    return SimpleNamespace(
        hidden_states=torch.zeros((rows, K), dtype=torch.bfloat16).to(FP8),
        hidden_states_scale=scale,
        m_indices=_padded_m_indices(rows),
    )


def _request(weights=None, acts=None, *, layout_alignment=128, **overrides):
    weights = weights or _weights()
    acts = acts or _activations()
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


def _allocate_output(rows=M):
    return torch.empty((rows, K), dtype=torch.bfloat16)


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
        kwargs["output_scale"].fill_(1)

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


def _quant_info(weights):
    return dg.DeepGemmMoeQuantInfo(
        w13_weight=weights.w13_weight,
        w2_weight=weights.w2_weight,
        use_fp8=True,
        w13_scale=weights.w13_scale,
        w2_scale=weights.w2_scale,
        block_shape=[128, 128],
    )


def _run_contiguous(weights, acts, *, alignment=128):
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
    running_state = {
        "all_tokens": int(acts.hidden_states.shape[0]),
        "hidden_states_device": acts.hidden_states.device,
        "hidden_states_dtype": torch.bfloat16,
        "hidden_states_shape": tuple(acts.hidden_states.shape),
        "contiguous_layout_alignment": alignment,
    }
    return dg.DeepGemmRunnerCore._run_contiguous_gemm(
        runner_self, runner_input, _quant_info(weights), running_state
    )


def _plan():
    (plan,) = dg._CAKE_CONTIG_FP8._plans.values()
    return plan


# -- route -------------------------------------------------------------------


def test_route_off_keeps_deepgemm_path(api, fake_deepgemm):
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(_routes.ENV_VAR, None)
        out = _run_contiguous(_weights(), _activations())
    assert torch.all(out == DEEPGEMM_FILL)
    assert fake_deepgemm == [(G, N, K), (G, K, H)]
    assert api.prepared == [] and api.log == []


def test_route_on_rebinds_callers_tensors_per_call(api, fake_deepgemm, fake_silu_quant):
    weights = _weights()
    acts1, acts2 = _activations(), _activations()
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: dg._CAKE_ROUTE}):
        out1 = _run_contiguous(weights, acts1)
        out2 = _run_contiguous(weights, acts2)
    assert fake_deepgemm == []
    for out in (out1, out2):
        assert out.dtype == torch.bfloat16 and tuple(out.shape) == (M, K)
        assert torch.all(out[acts1.m_indices >= 0] == CAKE_FILL)
    # gate_up then down, prepared once each and launched per call.
    assert api.log == [f"plain:{(G, N, K)}", f"plain:{(G, K, H)}"] * 2
    gateup, down = api.prepared
    assert len(api.prepared) == 2 and len(gateup.launches) == len(down.launches) == 2
    for runner in api.prepared:
        assert runner.alignment == 128 and runner.fill_padding is False
    # Per-token operands are the dispatcher's own tensors: no staging copies,
    # no scale unpacking, ``-1`` padding passed through untouched.
    for launch, acts in zip(gateup.launches, (acts1, acts2)):
        assert launch["a"] is acts.hidden_states
        assert launch["a_scale"] is acts.hidden_states_scale
        assert launch["a_scale"].dtype == torch.int32
        assert launch["a_scale"].stride() == (1, M)
        assert launch["m_indices"] is acts.m_indices
    assert torch.equal(acts1.m_indices, _padded_m_indices())
    assert int((acts1.m_indices < 0).sum()) == 36
    for launch, acts in zip(down.launches, (acts1, acts2)):
        assert launch["m_indices"] is acts.m_indices
    # The down GEMM writes the caller-owned output directly.
    assert down.launches[0]["out"] is out1 and down.launches[1]["out"] is out2
    assert out2 is not out1
    plan = _plan()
    assert plan.scale_ue8m0 and plan.alignment == 128
    assert all(c == dict(fill_padding=False, alignment=128) for c in api.supports_calls)


def test_down_input_quantized_with_deepgemm_ue8m0_layout(api, fake_silu_quant):
    out = dg._CAKE_CONTIG_FP8.try_run(_request(), _allocate_output)
    assert out is not None
    (call,) = fake_silu_quant
    assert call["scale_ue8m0"] is True and call["transposed"] is True
    assert call["swizzle"] is False and call["quant_group_size"] == 128
    gateup, down = api.prepared
    # SwiGLU reads the gate_up output and writes the down GEMM's operands.
    assert call["input"] is gateup.launches[0]["out"]
    assert (
        tuple(call["input"].shape) == (M, N) and call["input"].dtype == torch.bfloat16
    )
    assert call["output"] is down.launches[0]["a"]
    assert tuple(call["output"].shape) == (M, H) and call["output"].dtype == FP8
    scale = call["output_scale"]
    assert scale is down.launches[0]["a_scale"]
    # Packed UE8M0 int32, MN-major: the DeepGEMM path's own down-input layout.
    assert scale.dtype == torch.int32
    assert tuple(scale.shape) == (M, _scale_cols(H)) and scale.stride() == (1, M)


def test_fp32_scale_family_uses_fill_padding(api, fake_silu_quant):
    weights, acts = _weights(fp32_scales=True), _activations(fp32_scales=True)
    out = dg._CAKE_CONTIG_FP8.try_run(_request(weights, acts), _allocate_output)
    assert out is not None and torch.all(out[acts.m_indices >= 0] == CAKE_FILL)
    assert all(r.fill_padding is True and r.alignment == 128 for r in api.prepared)
    assert not _plan().scale_ue8m0
    (call,) = fake_silu_quant
    assert call["scale_ue8m0"] is False and call["transposed"] is False
    scale = call["output_scale"]
    assert scale.dtype == torch.float32 and tuple(scale.shape) == (M, H // 128)
    assert scale.is_contiguous()


def test_mixed_scale_dtypes_fall_back(api, caplog):
    weights = _weights(fp32_scales=True)
    with caplog.at_level("INFO", logger=dg.logger.name):
        out = dg._CAKE_CONTIG_FP8.try_run(_request(weights), _allocate_output)
    assert out is None and api.prepared == [] and api.supports_calls == []
    assert any("one scale family" in rec.getMessage() for rec in caplog.records)


def test_layout_alignment_is_passed_through(api, fake_silu_quant, caplog):
    weights = _weights()
    with caplog.at_level("INFO", logger=dg.logger.name):
        out = dg._CAKE_CONTIG_FP8.try_run(
            _request(weights, layout_alignment=64), _allocate_output
        )
    assert out is not None
    assert all(r.alignment == 64 for r in api.prepared)
    assert any("multi-run" in rec.getMessage() for rec in caplog.records)
    # ``None`` means the 128-row ep_scatter default (DeepEP contiguous layouts).
    assert (
        dg._CAKE_CONTIG_FP8.try_run(
            _request(weights, layout_alignment=None), _allocate_output
        )
        is not None
    )
    assert api.prepared[-1].alignment == 128
    assert len(dg._CAKE_CONTIG_FP8._plans) == 2


def test_alignment_not_multiple_of_32_falls_back(api, caplog):
    with caplog.at_level("INFO", logger=dg.logger.name):
        out = dg._CAKE_CONTIG_FP8.try_run(
            _request(layout_alignment=48), _allocate_output
        )
    assert out is None and api.prepared == []
    assert any("multiple of 32" in rec.getMessage() for rec in caplog.records)


def test_plans_are_cached_per_geometry_and_weights(api, fake_silu_quant):
    route = dg._CAKE_CONTIG_FP8
    weights = _weights()

    def run(rows):
        return route.try_run(
            _request(weights, _activations(rows)), lambda: _allocate_output(rows)
        )

    assert run(256) is not None and len(api.prepared) == 2
    assert run(128) is not None and len(api.prepared) == 4
    assert run(256) is not None and run(128) is not None
    assert len(api.prepared) == 4 and len(route._plans) == 2
    # Other expert weights get their own runners.
    assert route.try_run(_request(_weights()), _allocate_output) is not None
    assert len(api.prepared) == 6


def test_capturing_stream_with_unprepared_shape_falls_back(
    api, fake_silu_quant, monkeypatch
):
    weights = _weights()
    monkeypatch.setattr(dg, "_cake_stream_capturing", lambda: True)
    assert dg._CAKE_CONTIG_FP8.try_run(_request(weights), _allocate_output) is None
    assert api.prepared == [] and api.log == []
    # Warm the shape up eagerly, then the same shape is served inside capture.
    monkeypatch.setattr(dg, "_cake_stream_capturing", lambda: False)
    assert dg._CAKE_CONTIG_FP8.try_run(_request(weights), _allocate_output) is not None
    monkeypatch.setattr(dg, "_cake_stream_capturing", lambda: True)
    assert dg._CAKE_CONTIG_FP8.try_run(_request(weights), _allocate_output) is not None
    assert len(api.prepared) == 2 and len(api.log) == 4


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
    assert api.prepared == []


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"activation": "situ"}, "activation"),
        ({"use_mxfp8": True}, "mxfp8"),
        ({"w13_weight": torch.zeros((G, 384, K)).to(FP8)}, "inconsistent"),
        ({"activation_scale_block_size": 32}, "scale block"),
        ({"hidden_states_scale": _packed_activation_scale(M, 2 * K)}, "scale shape"),
        (
            {"hidden_states_scale": torch.ones((M, K // 128), dtype=torch.bfloat16)},
            "UE8M0 or FP32",
        ),
    ],
)
def test_static_rejections_never_touch_the_api(api, caplog, overrides, reason):
    with caplog.at_level("INFO", logger=dg.logger.name):
        assert (
            dg._CAKE_CONTIG_FP8.try_run(_request(**overrides), _allocate_output) is None
        )
    assert api.prepared == [] and api.supports_calls == []
    assert any(reason in rec.getMessage() for rec in caplog.records)


# -- compact-layout alignment policy -------------------------------------------


@pytest.fixture
def policy_env(api, monkeypatch):
    monkeypatch.setattr(dg, "_cake_runner_use_swizzle", lambda: False)
    monkeypatch.setattr(dg.deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", True)
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: dg._CAKE_ROUTE}):
        yield api


def _policy(default, weights=None, *, activation="silu"):
    return dg._cake_contiguous_layout_alignment(
        default,
        quant_info=_quant_info(weights or _weights()),
        runner_config=SimpleNamespace(activation=activation),
        hidden_size=K,
        device=torch.device("cuda", 0),
    )


def test_alignment_policy_picks_128_for_admitted_layers(policy_env, caplog):
    with caplog.at_level("INFO", logger=dg.logger.name):
        assert _policy(32) == 128
        assert _policy(224) == 128
    assert any("alignment 128" in rec.getMessage() for rec in caplog.records)
    assert _policy(128) == 128


def test_alignment_policy_keeps_deepgemm_choice_when_route_cannot_run(policy_env):
    assert _policy(32, activation="situ") == 32
    assert _policy(64, _weights(fp32_scales=True)) == 64  # UE8M0 dispatch, FP32 weights
    policy_env.available = False
    assert _policy(32) == 32
    policy_env.available = True
    with mock.patch.dict(os.environ, {_routes.ENV_VAR: "gdn_decode"}):
        _routes.reset_cache_for_tests()
        assert _policy(32) == 32


def test_alignment_policy_ignores_non_cuda_devices(policy_env):
    assert (
        dg._cake_contiguous_layout_alignment(
            32,
            quant_info=_quant_info(_weights()),
            runner_config=SimpleNamespace(activation="silu"),
            hidden_size=K,
            device=torch.device("cpu"),
        )
        == 32
    )


# -- adapter: block-scaled contract probe + scale admission --------------------


def _fake_flashinfer(monkeypatch, *, new_contract):
    def prepare_new(
        a,
        b,
        a_scale,
        b_scale,
        m_indices,
        out=None,
        *,
        validate_indices=False,
        fill_padding=False,
        alignment=128,
    ):
        return SimpleNamespace(
            kwargs=dict(
                validate_indices=validate_indices,
                fill_padding=fill_padding,
                alignment=alignment,
            )
        )

    def prepare_old(
        a, b, a_scale, b_scale, m_indices, out=None, *, validate_indices=False
    ):
        return SimpleNamespace(kwargs=dict(validate_indices=validate_indices))

    module = types.ModuleType("flashinfer.gemm.cake_grouped_fp8_gemm")
    module.prepare_group_gemm_fp8_nt_groupwise_contiguous = (
        prepare_new if new_contract else prepare_old
    )

    class _PreparedOld:
        pass

    class _PreparedNew:
        def release_prepared_operands(self):
            pass

    module.PreparedGroupGemmFp8NtGroupwiseContiguous = (
        _PreparedNew if new_contract else _PreparedOld
    )
    pkg_gemm = types.ModuleType("flashinfer.gemm")
    pkg = types.ModuleType("flashinfer")
    pkg.gemm = pkg_gemm
    pkg_gemm.cake_grouped_fp8_gemm = module
    monkeypatch.setitem(sys.modules, "flashinfer", pkg)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", pkg_gemm)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm.cake_grouped_fp8_gemm", module)
    adapter.block_scaled_contract_available.cache_clear()
    return module


@pytest.fixture
def _clear_contract_cache():
    adapter.block_scaled_contract_available.cache_clear()
    yield
    adapter.block_scaled_contract_available.cache_clear()


def test_contract_probe_inspects_alignment_keyword(monkeypatch, _clear_contract_cache):
    _fake_flashinfer(monkeypatch, new_contract=True)
    assert adapter.block_scaled_contract_available() is True
    _fake_flashinfer(monkeypatch, new_contract=False)
    assert adapter.block_scaled_contract_available() is False
    monkeypatch.delitem(sys.modules, "flashinfer.gemm.cake_grouped_fp8_gemm")
    monkeypatch.delitem(sys.modules, "flashinfer.gemm")
    monkeypatch.delitem(sys.modules, "flashinfer")
    monkeypatch.setitem(sys.modules, "flashinfer", None)
    adapter.block_scaled_contract_available.cache_clear()
    assert adapter.block_scaled_contract_available() is False


def test_prepare_forwards_alignment_only_on_the_new_contract(
    monkeypatch, _clear_contract_cache
):
    args = (None,) * 5
    _fake_flashinfer(monkeypatch, new_contract=True)
    runner = adapter.prepare_group_gemm_fp8_nt_groupwise_contiguous(
        *args, fill_padding=True, alignment=64
    )
    assert runner.kwargs == dict(
        validate_indices=False, fill_padding=True, alignment=64
    )
    _fake_flashinfer(monkeypatch, new_contract=False)
    runner = adapter.prepare_group_gemm_fp8_nt_groupwise_contiguous(*args)
    assert runner.kwargs == dict(validate_indices=False)
    with pytest.raises(TypeError, match="pyproject"):
        adapter.prepare_group_gemm_fp8_nt_groupwise_contiguous(*args, alignment=32)
    with pytest.raises(TypeError, match="pyproject"):
        adapter.prepare_group_gemm_fp8_nt_groupwise_contiguous(*args, fill_padding=True)


def test_alignment_ok():
    assert adapter.alignment_ok(None) and adapter.alignment_ok(32)
    assert adapter.alignment_ok(128) and adapter.alignment_ok(224)
    assert not adapter.alignment_ok(0) and not adapter.alignment_ok(48)
    assert not adapter.alignment_ok(-32)


@pytest.mark.parametrize("contract", [True, False])
def test_scales_ok_per_family(monkeypatch, contract):
    monkeypatch.setattr(adapter, "block_scaled_contract_available", lambda: contract)
    device = torch.device("cpu")
    common = dict(m=M, n=N, k=K, groups=G, device=device, allow_block_scaled=True)
    fp32_a = torch.ones((M, K // 128))
    fp32_b = torch.ones((G, N // 128, K // 128))
    assert adapter._scales_ok(fp32_a, fp32_b, **common)
    i32_a = _packed_activation_scale(M, K)  # MN-major strides accepted
    i32_b_rows = _packed_weight_scale(G, N, K)  # (G, N, cols), row-repeated
    i32_b_blocks = torch.ones((G, N // 128, _scale_cols(K)), dtype=torch.int32)
    assert adapter._scales_ok(i32_a, i32_b_rows, **common) is contract
    assert adapter._scales_ok(i32_a, i32_b_blocks, **common) is contract
    assert not adapter._scales_ok(i32_a, fp32_b, **common)  # mixed families
    assert not adapter._scales_ok(fp32_a, i32_b_rows, **common)
    assert not adapter._scales_ok(
        i32_a, i32_b_rows, **dict(common, allow_block_scaled=False)
    )
    assert not adapter._scales_ok(
        torch.ones((M, _scale_cols(K) + 1), dtype=torch.int32), i32_b_rows, **common
    )
    assert not adapter._scales_ok(fp32_a.t().contiguous().t(), fp32_b, **common)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
