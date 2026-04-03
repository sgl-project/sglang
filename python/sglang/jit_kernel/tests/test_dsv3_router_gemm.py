"""Tests for JIT dsv3_router_gemm kernel."""

import pytest
import torch

from sglang.jit_kernel.dsv3_router_gemm import (
    can_use_dsv3_router_gemm,
    dsv3_router_gemm,
)

HIDDEN_DIM = 7168
ATOL = 1e-2
RTOL = 1e-2


def _ref(hidden_states, router_weights, out_dtype):
    return (hidden_states.float() @ router_weights.float().T).to(out_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("num_tokens", list(range(1, 17)))
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_dsv3_router_gemm(num_experts, num_tokens, out_dtype):
    if not can_use_dsv3_router_gemm(num_experts, HIDDEN_DIM):
        pytest.skip("SM90+ required")

    mat_a = torch.randn(num_tokens, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")
    mat_b = torch.randn(num_experts, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")

    ref = _ref(mat_a, mat_b, out_dtype)
    out = dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == out_dtype
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
