import sys

import pytest
import torch

from sglang.kernels.ops.quantization.awq_triton import (
    awq_dequantize_decomposition,
    awq_dequantize_triton,
    awq_gemm_triton,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, suite="stage-a-test-1-gpu-small-amd")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [32, 64, 128, 512])
@pytest.mark.parametrize("split_k", [1, 8])
def test_gemm_activation_dtypes(dtype, group_size, split_k):
    torch.manual_seed(0)
    x = torch.randn((3, 512), device="cuda", dtype=dtype)
    qweight = torch.randint(
        -(2**31), 2**31 - 1, (512, 9), device="cuda", dtype=torch.int32
    )
    qzeros = torch.randint(
        -(2**31),
        2**31 - 1,
        (512 // group_size, 9),
        device="cuda",
        dtype=torch.int32,
    )
    scales = torch.rand((512 // group_size, 72), device="cuda", dtype=dtype) * 0.05
    weight = awq_dequantize_decomposition(qweight, scales, qzeros)
    torch.testing.assert_close(
        awq_dequantize_triton(qweight, scales, qzeros), weight, atol=0, rtol=0
    )
    expected = (x.float() @ weight.float()).to(dtype)
    actual = awq_gemm_triton(
        x,
        qweight,
        scales,
        qzeros,
        split_k,
        block_size_m=16,
        block_size_n=64,
    )
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.01)


@pytest.mark.parametrize("split_k", [1, 8])
def test_gemm_fp32_accumulation(split_k):
    # Each half overflows FP16, although the complete dot product is zero.
    x = torch.full((1, 4096), 512.0, dtype=torch.float16, device="cuda")
    x[:, 2048:] = -512.0
    qweight = torch.full((4096, 8), 0x11111111, dtype=torch.int32, device="cuda")
    qzeros = torch.zeros((32, 8), dtype=torch.int32, device="cuda")
    scales = torch.ones((32, 64), dtype=torch.float16, device="cuda")
    actual = awq_gemm_triton(
        x,
        qweight,
        scales,
        qzeros,
        split_k,
        block_size_m=16,
        block_size_n=64,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0.002, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
