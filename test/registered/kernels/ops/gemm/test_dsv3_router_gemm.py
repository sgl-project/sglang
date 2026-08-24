"""Integration tests for SGLang's FlashInfer router GEMM adapter."""

import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.kernels.ops.gemm import dsv3_router_gemm
from sglang.kernels.ops.gemm.flashinfer_router_gemm import (
    is_flashinfer_router_gemm_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=37, stage="base-b-kernel-unit", runner_config="1-gpu-large")

SUPPORTED_DEVICE_SMS = {90, 100, 103, 107}
ROUTER_GEMM_CASES = [
    # Existing fixed-shape APIs.
    (128, 7168, 1, torch.bfloat16),
    (256, 6144, 4, torch.float32),
    (256, 7168, 16, torch.float32),
    # APIs added by flashinfer-ai/flashinfer#4630.
    (256, 7168, 2, torch.bfloat16),
    (384, 7168, 8, torch.bfloat16),
    (384, 7168, 16, torch.float32),
    (896, 7168, 2, torch.bfloat16),
    (896, 7168, 4, torch.float32),
]
ATOL = 1e-2
RTOL = 1e-2


def _ref(hidden_states, router_weights, out_dtype):
    return (hidden_states.float() @ router_weights.float().T).to(out_dtype)


def _device_sm() -> int:
    arch = get_jit_cuda_arch()
    return arch.major * 10 + arch.minor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "num_experts,hidden_dim,num_tokens,out_dtype", ROUTER_GEMM_CASES
)
def test_dsv3_router_gemm(num_experts, hidden_dim, num_tokens, out_dtype):
    device_sm = _device_sm()
    if is_hip_runtime() or device_sm not in SUPPORTED_DEVICE_SMS:
        pytest.skip("FlashInfer router GEMM requires SM90, SM100, SM103, or SM107")

    assert is_flashinfer_router_gemm_supported(
        num_tokens, hidden_dim, num_experts, out_dtype, device_sm
    ), "The installed FlashInfer must provide the router GEMM API pinned by this branch"

    hidden_states = torch.randn(
        num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )
    router_weights = torch.randn(
        num_experts, hidden_dim, dtype=torch.bfloat16, device="cuda"
    )

    ref = _ref(hidden_states, router_weights, out_dtype)
    out = dsv3_router_gemm(hidden_states, router_weights, out_dtype=out_dtype)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == out_dtype
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
