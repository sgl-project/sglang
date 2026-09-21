import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.layers import hc_mix_triton, hyperconnection
from sglang.srt.layers.hc_mix_triton import (
    _FUSED_MIX_MAX_ROWS,
    fused_hc_mix,
    fused_hc_mix_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320


def _reference_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc: int,
    hs: int,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mirrors GatedResidual._mix_compute in hyperconnection.py."""
    x = hyper_input_normed.to(compute_dtype)
    t = F.silu(F.linear(x, w_down.to(compute_dtype)) / hc)
    u = torch.sigmoid(F.linear(t, w_up.to(compute_dtype)))
    return (u.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def _make_inputs(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda")
    w_down = (
        torch.randn(LOWRANK, HC_COUNT * HIDDEN_SIZE, dtype=dtype, device="cuda") * 0.02
    )
    w_up = (
        torch.randn(HC_COUNT * HIDDEN_SIZE, LOWRANK, dtype=dtype, device="cuda") * 0.02
    )
    return x, w_down, w_up


_TOLERANCES = {
    torch.bfloat16: dict(rtol=1e-2, atol=5e-3),
    torch.float16: dict(rtol=2e-3, atol=1e-3),
}


@pytest.mark.parametrize("npu_platform", [False, True])
def test_hyperconnection_fallback_compile_policy(monkeypatch, npu_platform):
    calls = []

    def record_compile(fn, **kwargs):
        calls.append((fn.__name__, kwargs))
        return fn

    monkeypatch.setattr(hyperconnection, "_is_npu", npu_platform)
    monkeypatch.setattr(torch, "compile", record_compile)
    # The fallback functions are created regardless of weight allocation.
    hyperconnection.GatedResidual(
        hyperconnection.HyperConnectionConfig(), use_mix=False, use_combine=False
    )
    assert calls == [
        ("_mix_compute", {"disable": npu_platform}),
        ("_combine_compute", {"disable": npu_platform}),
    ]


@pytest.mark.parametrize("npu_platform", [False, True])
@pytest.mark.parametrize("tensor_is_cuda", [False, True])
@pytest.mark.parametrize("ple_norm", [False, True])
def test_grouped_norm_cuda_jit_dispatch(
    monkeypatch, npu_platform, tensor_is_cuda, ple_norm
):
    from sglang.kernels.ops.layernorm import grouped_gemma_rmsnorm as norm_kernel
    from sglang.srt.models import qwen4_exp

    norm_cls = (
        qwen4_exp.Qwen4ExpPLEGroupedNorm
        if ple_norm
        else hyperconnection.GroupedGemmaRMSNorm
    )
    norm = norm_cls(1024, group_size=512).to(dtype=torch.bfloat16)
    x = torch.ones(2, 1024, dtype=torch.bfloat16)
    sentinel = torch.full_like(x, 7)
    calls = []

    def fake_cuda_kernel(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(hyperconnection, "_is_npu", npu_platform)
    monkeypatch.setattr(qwen4_exp, "_is_npu", npu_platform)
    # Exercise the actual forwards without needing a CUDA device or compiler.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: tensor_is_cuda))
    monkeypatch.setattr(norm_kernel, "grouped_gemma_rmsnorm", fake_cuda_kernel)
    npu_calls = []
    if npu_platform and not ple_norm:
        def fake_npu_kernel(*args):
            npu_calls.append(args)
            return sentinel

        # Routing-only test: do not import the actual NPU package on CUDA CI.
        monkeypatch.setitem(
            sys.modules, "sgl_kernel_npu.qwen3_8_flash_next",
            SimpleNamespace(hc=SimpleNamespace(grouped_norm=fake_npu_kernel)),
        )
    actual = norm(x)
    if npu_platform and not ple_norm:
        assert actual is sentinel
        assert len(npu_calls) == 1 and not calls
    elif not npu_platform and tensor_is_cuda:
        assert actual is sentinel
        assert len(calls) == 1
    else:
        assert not calls
        torch.testing.assert_close(actual, x)


def test_npu_cuda_compat_shim_does_not_enable_cuda_kernels(monkeypatch):
    # No device/dtype/shape attributes: the platform gate must short-circuit.
    tensor = SimpleNamespace(is_cuda=True)
    monkeypatch.setattr(hc_mix_triton, "_is_npu", True)
    monkeypatch.setattr(hc_mix_triton, "_deterministic_inference", lambda: False)
    assert not fused_hc_mix_supported(tensor, tensor, tensor)


@pytest.mark.parametrize("ple_norm", [False, True])
def test_npu_grouped_norm_matches_reference(monkeypatch, ple_norm):
    from sglang.srt.models import qwen4_exp

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an NPU")
    monkeypatch.setattr(hyperconnection, "_is_npu", True)
    monkeypatch.setattr(qwen4_exp, "_is_npu", True)
    norm_cls = (
        qwen4_exp.Qwen4ExpPLEGroupedNorm
        if ple_norm
        else hyperconnection.GroupedGemmaRMSNorm
    )
    width, group_size = (1024, 512) if ple_norm else (10240, 2560)
    norm = norm_cls(width, group_size=group_size).to(device="npu", dtype=torch.bfloat16)
    torch.manual_seed(7)
    x = torch.randn(3, width, dtype=torch.bfloat16, device="npu")
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight) * 0.1)
    assert norm._jit_group_size == group_size
    actual = norm(x)
    grouped = x.cpu().float().reshape(3, width // group_size, group_size)
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
    expected = (
        normalized.reshape(3, width) * (1 + norm.weight.detach().cpu().float())
    ).to(x.dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 4, 7, _FUSED_MIX_MAX_ROWS])
def test_fused_hc_mix_matches_reference(dtype, num_tokens):
    x, w_down, w_up = _make_inputs(num_tokens, dtype)
    assert fused_hc_mix_supported(x, w_down, w_up)
    out = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    torch.testing.assert_close(out.to(torch.float64), ref, **_TOLERANCES[dtype])


def test_fused_hc_mix_no_less_accurate_than_eager():
    """The fused kernel (fp32 accumulation throughout) must not be farther
    from the fp64 reference than the eager bf16 chain it replaces."""
    x, w_down, w_up = _make_inputs(8, torch.bfloat16)
    ref = _reference_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    fused = fused_hc_mix(x, w_down, w_up, HC_COUNT, HIDDEN_SIZE)
    eager = _reference_mix(
        x, w_down, w_up, HC_COUNT, HIDDEN_SIZE, compute_dtype=torch.bfloat16
    )
    fused_err = (fused.to(torch.float64) - ref).abs().max()
    eager_err = (eager.to(torch.float64) - ref).abs().max()
    assert fused_err <= eager_err * 1.5 + 1e-6


def test_fused_hc_mix_gate_rejects_prefill_rows():
    x, w_down, w_up = _make_inputs(_FUSED_MIX_MAX_ROWS + 1, torch.bfloat16)
    assert not fused_hc_mix_supported(x, w_down, w_up)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
