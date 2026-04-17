import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")


def ref_silu_mul_quant(input, group_size=128, eps=1e-10):
    """Reference implementation: separate silu_and_mul + fp8 quant."""
    M, K2 = input.shape
    N = K2 // 2
    gate = input[:, :N].float()
    up = input[:, N:].float()
    activated = (gate * torch.sigmoid(gate)) * up  # silu(gate) * up

    # Per-group quantization
    fp8_max = 448.0
    activated_flat = activated.reshape(-1, group_size)
    absmax = activated_flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scales = (absmax / fp8_max).squeeze(-1)
    quantized = (activated_flat / absmax * fp8_max).clamp(-fp8_max, fp8_max)
    quantized = quantized.reshape(M, N).to(torch.float8_e4m3fn)
    scales = scales.reshape(M, N // group_size)
    return quantized, scales


@pytest.mark.parametrize("M", [1, 4, 16, 64])
@pytest.mark.parametrize("N", [128, 256, 384, 768])
@pytest.mark.parametrize("group_size", [128])
def test_fused_silu_mul_quant_correctness(M, N, group_size):
    from sglang.jit_kernel.fused_silu_mul_quant import fused_silu_mul_quant

    input = torch.randn(M, 2 * N, dtype=torch.bfloat16, device="cuda")

    out_q, out_s = fused_silu_mul_quant(input, group_size=group_size)
    ref_q, ref_s = ref_silu_mul_quant(input, group_size=group_size)

    assert out_q.shape == (M, N)
    assert out_q.dtype == torch.float8_e4m3fn
    assert out_s.shape == (M, N // group_size)

    # Compare scales (should be very close)
    torch.testing.assert_close(out_s, ref_s, rtol=1e-3, atol=1e-5)

    # Compare quantized values (FP8 has limited precision)
    out_float = out_q.float()
    ref_float = ref_q.float()
    torch.testing.assert_close(out_float, ref_float, rtol=0.1, atol=1.0)


@pytest.mark.parametrize("M", [1, 16])
def test_fused_silu_mul_quant_column_major(M):
    from sglang.jit_kernel.fused_silu_mul_quant import fused_silu_mul_quant

    N = 256
    group_size = 128
    input = torch.randn(M, 2 * N, dtype=torch.bfloat16, device="cuda")

    out_q, out_s = fused_silu_mul_quant(
        input, group_size=group_size, column_major_scales=True
    )

    assert out_q.shape == (M, N)
    assert out_s.shape == (N // group_size, M)  # column-major


def test_empty_input():
    from sglang.jit_kernel.fused_silu_mul_quant import fused_silu_mul_quant

    input = torch.empty(0, 256, dtype=torch.bfloat16, device="cuda")
    out_q, out_s = fused_silu_mul_quant(input, group_size=128)
    assert out_q.shape == (0, 128)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
