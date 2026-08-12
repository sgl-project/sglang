"""Smoke coverage for the reduced Windows ARM64 CUDA wheel."""

import platform
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or platform.machine().lower() != "arm64",
    reason="Windows ARM64 kernel smoke tests",
)


def test_minimax_core_cuda_ops():
    import sgl_kernel

    assert torch.cuda.is_available()

    torch.manual_seed(0)
    activation_input = torch.randn(4, 256, device="cuda", dtype=torch.float16)
    activation_output = sgl_kernel.silu_and_mul(activation_input)
    activation_reference = (
        torch.nn.functional.silu(activation_input[:, :128]) * activation_input[:, 128:]
    )
    torch.testing.assert_close(
        activation_output, activation_reference, rtol=1e-3, atol=1e-3
    )

    norm_input = torch.randn(4, 128, device="cuda", dtype=torch.float16)
    norm_weight = torch.randn(128, device="cuda", dtype=torch.float16)
    norm_output = sgl_kernel.rmsnorm(
        norm_input, norm_weight, eps=1e-6, enable_pdl=False
    )
    norm_reference = (
        norm_input
        * torch.rsqrt(norm_input.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(
            norm_input.dtype
        )
        * norm_weight
    )
    torch.testing.assert_close(norm_output, norm_reference, rtol=2e-3, atol=2e-3)

    gating_output = torch.randn(8, 32, device="cuda", dtype=torch.float32)
    topk_weights = torch.empty(8, 4, device="cuda", dtype=torch.float32)
    topk_indices = torch.empty(8, 4, device="cuda", dtype=torch.int32)
    sgl_kernel.topk_sigmoid(topk_weights, topk_indices, gating_output)
    weights_reference, indices_reference = torch.topk(
        torch.sigmoid(gating_output), 4, dim=-1
    )
    torch.testing.assert_close(topk_weights, weights_reference, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(topk_indices, indices_reference.int(), rtol=0, atol=0)

    quant_input = torch.randn(32, 512, device="cuda", dtype=torch.float16)
    quant_output = torch.empty_like(quant_input, dtype=torch.float8_e4m3fn)
    quant_scales = torch.empty(32, device="cuda", dtype=torch.float32)
    sgl_kernel.sgl_per_token_quant_fp8(quant_input, quant_output, quant_scales)
    assert torch.isfinite(quant_scales).all()
    assert (quant_scales > 0).all()
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    quant_reference = (
        quant_input.float() * quant_scales.reciprocal().unsqueeze(1)
    ).clamp(min=fp8_info.min, max=fp8_info.max)
    torch.testing.assert_close(
        quant_output.float(),
        quant_reference.to(torch.float8_e4m3fn).float(),
        rtol=1e-3,
        atol=1e-3,
    )

    moe_input = torch.randn(8, 4, 128, device="cuda", dtype=torch.bfloat16)
    moe_output = torch.empty(8, 128, device="cuda", dtype=torch.bfloat16)
    sgl_kernel.moe_sum_reduce(moe_input, moe_output, 1.0)
    torch.testing.assert_close(moe_output, moe_input.sum(dim=1), rtol=2e-2, atol=2e-2)
