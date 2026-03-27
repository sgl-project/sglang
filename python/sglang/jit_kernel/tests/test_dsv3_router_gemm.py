"""
JIT kernel test for DeepSeek V3 router GEMM.

Adapted from sgl-kernel/tests/test_dsv3_router_gemm.py.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-b-kernel-unit-1-gpu")


def _skip_if_not_sm90():
    from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime

    if is_hip_runtime():
        pytest.skip("dsv3_router_gemm JIT kernel requires CUDA (not ROCm)")
    arch = get_jit_cuda_arch()
    if arch.major < 9:
        pytest.skip(
            f"dsv3_router_gemm JIT kernel requires SM90+ (got SM{arch.major}{arch.minor})"
        )


@pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)])
@pytest.mark.parametrize("num_experts", [256, 384])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_dsv3_router_gemm_jit(num_tokens, num_experts, out_dtype):
    _skip_if_not_sm90()

    from sglang.jit_kernel.dsv3_router_gemm import dsv3_router_gemm

    hidden_dim = 7168
    mat_a = torch.randn(
        (num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    mat_b = torch.randn(
        (num_experts, hidden_dim), dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    ref = F.linear(mat_a, mat_b).to(out_dtype)
    out = dsv3_router_gemm(mat_a, mat_b, out_dtype=out_dtype)

    assert out.shape == (num_tokens, num_experts)
    assert out.dtype == out_dtype
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


def test_can_use_dsv3_router_gemm():
    _skip_if_not_sm90()

    from sglang.jit_kernel.dsv3_router_gemm import can_use_dsv3_router_gemm

    assert can_use_dsv3_router_gemm(256, 7168) is True
    assert can_use_dsv3_router_gemm(384, 7168) is True
    assert can_use_dsv3_router_gemm(128, 7168) is False  # unsupported num_experts
    assert can_use_dsv3_router_gemm(256, 4096) is False  # unsupported hidden_dim


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
