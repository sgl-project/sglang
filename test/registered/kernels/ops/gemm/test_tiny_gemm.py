"""Tests for the JIT tiny_gemm kernels."""

import sys

import pytest
import torch

from sglang.kernels.jit.utils import (
    get_ci_test_range,
    get_jit_cuda_arch,
    is_hip_runtime,
)
from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=300, stage="nightly", runner_config="1-gpu-large")

# One kernel is built per m in [1, MAX_M], so hold max_m fixed across the sweep:
# every num_tokens of a shape then shares one JIT module.
MAX_M = 16
SHAPES = [(256, 7168), (384, 7168), (256, 4096), (896, 7168), (144, 7168), (1536, 128)]

TINY_GEMM_CASES = get_ci_test_range(
    [
        (n, k, num_tokens, dtype)
        for n, k in SHAPES
        for num_tokens in range(1, MAX_M + 1)
        for dtype in (torch.bfloat16, torch.float32)
    ],
    [
        (384, 7168, 1, torch.float32),
        (384, 7168, 4, torch.float32),
        (896, 7168, 8, torch.float32),
        (1536, 128, 16, torch.bfloat16),
    ],
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n,k,num_tokens,out_dtype", TINY_GEMM_CASES)
def test_tiny_gemm(n, k, num_tokens, out_dtype):
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        pytest.skip("SM90+ required")

    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    assert can_use_tiny_gemm(n, k, MAX_M)
    out = tiny_gemm_bf16(x, w, out_dtype=out_dtype, max_m=MAX_M)
    ref = torch.nn.functional.linear(x, w)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
