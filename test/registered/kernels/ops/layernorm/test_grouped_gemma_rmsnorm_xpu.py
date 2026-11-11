"""Grouped Gemma RMSNorm on XPU against an fp64 reference."""

from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm
from sglang.srt.models.qwen4_exp import Qwen4ExpPLEGroupedNorm
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(), reason="XPU required"
)

_TOLERANCES = {
    torch.bfloat16: dict(rtol=5e-3, atol=5e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
}


def _reference(x, weight, group_size, eps):
    # fp64 is not available on every XPU, so the reference runs on the CPU.
    x = x.cpu().to(torch.float64)
    weight = weight.cpu()
    hidden = x.shape[-1]
    grouped = x.reshape(*x.shape[:-1], hidden // group_size, group_size)
    variance = grouped.pow(2).mean(dim=-1, keepdim=True)
    normed = (grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return normed * (1.0 + weight.to(torch.float64))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_tokens,hidden_size,group_size",
    [(1, 10240, 2560), (7, 10240, 2560), (33, 1024, 512), (5, 512, 512)],
)
def test_matches_reference(dtype, num_tokens, hidden_size, group_size):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="xpu")
    weight = torch.randn(hidden_size, dtype=dtype, device="xpu") * 0.2
    out = grouped_gemma_rmsnorm(x, weight, group_size, 1e-6)
    expected = _reference(x, weight, group_size, 1e-6).to(dtype).to("xpu")
    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_non_contiguous_input(dtype):
    torch.manual_seed(0)
    base = torch.randn(10240, 6, dtype=dtype, device="xpu")
    x = base.t()[:5]
    assert not x.is_contiguous()
    weight = torch.randn(10240, dtype=dtype, device="xpu") * 0.2
    out = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6)
    expected = _reference(x, weight, 2560, 1e-6).to(dtype).to("xpu")
    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


@pytest.mark.parametrize("shape", [(64, 10240), (2, 8, 10240)])
def test_out_buffer_is_written_in_place(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="xpu")
    weight = torch.randn(10240, dtype=torch.bfloat16, device="xpu") * 0.2
    out = torch.full_like(x, float("nan"))
    result = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6, out=out)
    assert result.shape == x.shape
    assert result.data_ptr() == out.data_ptr()
    expected = _reference(x, weight, 2560, 1e-6).to(x.dtype).to("xpu")
    torch.testing.assert_close(out, expected, **_TOLERANCES[x.dtype])


@pytest.mark.parametrize("norm_type", [GroupedGemmaRMSNorm, Qwen4ExpPLEGroupedNorm])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_module_mixed_dtype_dispatch(norm_type, dtype, weight_dtype):
    norm = norm_type(1024, group_size=512).to(device="xpu", dtype=weight_dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.2, 0.2, 1024, device="xpu"))
    x = torch.randn(3, 1024, device="xpu", dtype=dtype)
    with patch(
        "sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm.grouped_gemma_rmsnorm",
        wraps=grouped_gemma_rmsnorm,
    ) as fast:
        result = norm(x)
    assert fast.call_count == (weight_dtype == dtype)
    expected = _reference(x, norm.weight, 512, 1e-6).to(dtype).to("xpu")
    torch.testing.assert_close(result, expected, **_TOLERANCES[dtype])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
