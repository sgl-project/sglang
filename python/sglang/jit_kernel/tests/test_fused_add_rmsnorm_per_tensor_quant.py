import itertools

import pytest
import torch

from sglang.jit_kernel.utils import get_ci_test_range


def reference_fused_add_rmsnorm_per_tensor_quant(
    input, residual, weight, scale, eps=1e-6
):
    updated_residual = residual + input
    x = updated_residual.float()
    mean = x.pow(2).mean(dim=-1, keepdim=True)
    norm = (mean + eps).rsqrt()
    normed = x * norm * weight.float()
    inv_scale = 1.0 / scale
    quantized = (normed * inv_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized, updated_residual


HIDDEN_SIZE_LIST = get_ci_test_range(
    [512, 1024, 1536, 2048, 3072, 4096, 5120, 8192],
    [512, 2048, 8192],
)
BS_LIST = get_ci_test_range(
    [1, 4, 16, 64, 256],
    [1, 16, 256],
)
DTYPE_LIST = [torch.bfloat16, torch.float16]
SCALE_LIST = get_ci_test_range(
    [0.1, 1.0, 4.0, 10.0],
    [0.1, 1.0],
)
DEVICE = "cuda"


@pytest.mark.parametrize(
    "batch_size,hidden_size,dtype,scale_val",
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST, DTYPE_LIST, SCALE_LIST)),
)
def test_fused_add_rmsnorm_per_tensor_quant_correctness(
    batch_size: int, hidden_size: int, dtype: torch.dtype, scale_val: float
) -> None:
    from sglang.jit_kernel.norm import fused_add_rmsnorm_per_tensor_quant

    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    residual = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    scale = torch.tensor([scale_val], device=DEVICE, dtype=torch.float32)

    original_residual = residual.clone()

    output = torch.empty(input.shape, dtype=torch.float8_e4m3fn, device=input.device)
    fused_add_rmsnorm_per_tensor_quant(output, input, residual, weight, scale)
    expected_output, expected_residual = reference_fused_add_rmsnorm_per_tensor_quant(
        input, original_residual, weight, scale
    )

    assert output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        output.float(), expected_output.float(), atol=1.5e-1, rtol=1.5e-1
    )
    torch.testing.assert_close(
        residual, input + original_residual, atol=1e-4, rtol=1e-4
    )


@pytest.mark.parametrize(
    "batch_size,hidden_size",
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST)),
)
def test_fused_add_rmsnorm_per_tensor_quant_matches_separate_ops(
    batch_size: int, hidden_size: int
) -> None:
    from sglang.jit_kernel.norm import (
        fused_add_rmsnorm,
        fused_add_rmsnorm_per_tensor_quant,
    )
    from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8

    dtype = torch.bfloat16
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    residual = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    scale = torch.tensor([1.0], device=DEVICE, dtype=torch.float32)

    # Fused path
    residual_fused = residual.clone()
    fused_output = torch.empty(
        input.shape, dtype=torch.float8_e4m3fn, device=input.device
    )
    fused_add_rmsnorm_per_tensor_quant(
        fused_output, input, residual_fused, weight, scale
    )

    # Separate path: fused_add_rmsnorm then per-tensor fp8 quantize
    input_sep = input.clone()
    residual_sep = residual.clone()
    fused_add_rmsnorm(input_sep, residual_sep, weight)
    separate_output = torch.empty_like(input_sep, dtype=torch.float8_e4m3fn)
    per_tensor_quant_fp8(input_sep, separate_output, scale, is_static=True)

    assert fused_output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(residual_fused, residual_sep, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        fused_output.float(), separate_output.float(), atol=1.5e-1, rtol=1.5e-1
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
