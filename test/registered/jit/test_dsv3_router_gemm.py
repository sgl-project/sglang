"""Tests for JIT dsv3_router_gemm kernel."""

import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm
from sglang.jit_kernel.utils import get_ci_test_range, get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=37, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=148, suite="nightly-kernel-1-gpu", nightly=True)

HIDDEN_DIMS = [1024, 4096, 5120, 6144, 7168]
ROUTER_GEMM_CASES = get_ci_test_range(
    list(
        itertools.product(
            [256, 384],
            HIDDEN_DIMS,
            list(range(1, 17)),
            [torch.bfloat16, torch.float32],
        )
    ),
    [
        (256, 1024, 1, torch.bfloat16),
        (256, 6144, 4, torch.float32),
        (384, 7168, 8, torch.bfloat16),
        (384, 5120, 16, torch.float32),
    ],
)
ATOL = 1e-2
RTOL = 1e-2


def _ref(hidden_states, router_weights, out_dtype):
    return (hidden_states.float() @ router_weights.float().T).to(out_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "num_experts,hidden_dim,num_tokens,out_dtype", ROUTER_GEMM_CASES
)
def test_dsv3_router_gemm(num_experts, hidden_dim, num_tokens, out_dtype):
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        pytest.skip("SM90+ required")

    mat_a = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
    mat_b = torch.randn(num_experts, hidden_dim, dtype=torch.bfloat16, device="cuda")

    ref = _ref(mat_a, mat_b, out_dtype)
    out = dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == out_dtype
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
