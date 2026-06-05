"""Tests for the CuTe DSL DeepSeek-V3 fused-A GEMM kernel."""

import sys

import pytest
import torch

from sglang.jit_kernel.cutedsl_dsv3_fused_a_gemm import dsv3_fused_a_gemm
from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

HD_OUT = 2112
HD_INS = [6144, 7168]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hd_in", HD_INS)
@pytest.mark.parametrize("num_tokens", list(range(1, 17)))
def test_dsv3_fused_a_gemm(num_tokens, hd_in):
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        pytest.skip("SM90+ required")

    torch.manual_seed(num_tokens)
    weight = torch.randn(HD_OUT, hd_in, dtype=torch.bfloat16, device="cuda")
    mat_a = torch.randn(num_tokens, hd_in, dtype=torch.bfloat16, device="cuda")
    mat_b = weight.t()

    out = dsv3_fused_a_gemm(mat_a, mat_b)
    assert out.shape == (num_tokens, HD_OUT)
    assert out.dtype == torch.bfloat16

    ref = (mat_a.float() @ weight.float().T).bfloat16()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
