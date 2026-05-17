import itertools

import pytest
import torch

from sglang.jit_kernel.utils import get_ci_test_range


def reference_rmsnorm_per_tensor_quant(input, weight, scale, eps=1e-6):
    x = input.float()
    mean = x.pow(2).mean(dim=-1, keepdim=True)
    norm = (mean + eps).rsqrt()
    normed = x * norm * weight.float()
    inv_scale = 1.0 / scale
    quantized = (normed * inv_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized


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
def test_rmsnorm_per_tensor_quant_correctness(
    batch_size: int, hidden_size: int, dtype: torch.dtype, scale_val: float
) -> None:
    from sglang.jit_kernel.norm import (
        rmsnorm_per_tensor_quant as jit_rmsnorm_per_tensor_quant,
    )

    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    scale = torch.tensor([scale_val], device=DEVICE, dtype=torch.float32)

    output = torch.empty(input.shape, dtype=torch.float8_e4m3fn, device=input.device)
    jit_rmsnorm_per_tensor_quant(output, input, weight, scale)
    expected = reference_rmsnorm_per_tensor_quant(input, weight, scale)

    assert output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        output.float(), expected.float(), atol=1.5e-1, rtol=1.5e-1
    )


@pytest.mark.parametrize(
    "batch_size,hidden_size",
    list(itertools.product(BS_LIST, HIDDEN_SIZE_LIST)),
)
def test_rmsnorm_per_tensor_quant_matches_separate_ops(
    batch_size: int, hidden_size: int
) -> None:
    from sglang.jit_kernel.norm import (
        rmsnorm,
    )
    from sglang.jit_kernel.norm import (
        rmsnorm_per_tensor_quant as jit_rmsnorm_per_tensor_quant,
    )
    from sglang.jit_kernel.per_tensor_quant_fp8 import per_tensor_quant_fp8

    dtype = torch.bfloat16
    input = torch.randn(batch_size, hidden_size, device=DEVICE, dtype=dtype)
    weight = torch.randn(hidden_size, device=DEVICE, dtype=dtype)
    scale = torch.tensor([1.0], device=DEVICE, dtype=torch.float32)

    # Fused path
    fused_output = torch.empty(
        input.shape, dtype=torch.float8_e4m3fn, device=input.device
    )
    jit_rmsnorm_per_tensor_quant(fused_output, input, weight, scale)

    # Separate path: rmsnorm then quantize
    normed = input.clone()
    rmsnorm(normed, weight, out=normed)
    separate_output = torch.empty_like(normed, dtype=torch.float8_e4m3fn)
    per_tensor_quant_fp8(normed, separate_output, scale, is_static=True)

    assert fused_output.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        fused_output.float(), separate_output.float(), atol=1.5e-1, rtol=1.5e-1
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
