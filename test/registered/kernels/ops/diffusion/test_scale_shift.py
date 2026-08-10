import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.triton.scale_shift import (
    try_fused_scaled_residual_add_exact,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_residual_add_is_bit_exact(dtype):
    torch.manual_seed(0)
    residual = torch.randn(2, 17, 64, device="cuda", dtype=torch.float32)
    x = torch.randn(2, 17, 64, device="cuda", dtype=dtype)
    scale = torch.randn(64, device="cuda", dtype=torch.float32)

    actual = try_fused_scaled_residual_add_exact(residual, x, scale)
    expected = residual + x * scale
    assert actual is not None
    assert torch.equal(actual, expected)


@torch.no_grad()
def test_scaled_residual_add_rejects_unsupported_inputs():
    residual = torch.empty(2, 3, 8, device="cuda", dtype=torch.float32)
    x = torch.empty_like(residual)
    scale = torch.empty(8, device="cuda", dtype=torch.float32)

    assert try_fused_scaled_residual_add_exact(residual, x, scale) is None
    assert try_fused_scaled_residual_add_exact(residual, x.half(), scale[:-1]) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
