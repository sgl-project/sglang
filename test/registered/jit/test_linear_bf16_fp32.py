"""Tests for the shared BF16 x FP32 GEMM helper."""

import sys

import pytest
import torch

from sglang.jit_kernel.dsv4.gemm import (
    _linear_bf16_fp32_hpc_ops,
    linear_bf16_fp32,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-unit-1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m", [1, 8, 64])
def test_linear_bf16_fp32_matches_fp32_reference(weight_dtype, m):
    torch.manual_seed(42)
    x = torch.randn((m, 4096), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((256, 4096), dtype=torch.float32, device="cuda").to(weight_dtype)

    out = linear_bf16_fp32(x, w)
    ref = x.float() @ w.float().t()

    assert out.shape == (m, 256)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_bf16_fp32_hpc_ops_optional_path():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("HPC-Ops public wheel is currently SM90-only")
    try:
        import hpc  # noqa: F401
    except Exception:
        pytest.skip("HPC-Ops is not installed")

    torch.manual_seed(42)
    x = torch.randn((16, 4096), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((256, 4096), dtype=torch.float32, device="cuda")

    out = _linear_bf16_fp32_hpc_ops(x, w)
    ref = x.float() @ w.t()

    assert out is not None
    assert out.shape == (16, 256)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-1, rtol=8e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
