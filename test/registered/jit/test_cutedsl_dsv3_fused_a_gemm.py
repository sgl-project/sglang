"""Tests for the CuTe DSL DeepSeek-V3 fused-A GEMM kernel."""

import sys

import pytest
import torch

from sglang.jit_kernel.utils import get_ci_test_range, get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.jit_kernel.cutedsl_dsv3_fused_a_gemm import dsv3_fused_a_gemm  # noqa: E402

# hd_in must be a multiple of 256; 6144/7168 cover the real fused-A shapes.
HD_INS = [6144, 7168]
# hd_out must be a multiple of 16; 2112 and 2624 cover real fused-A variants.
HD_OUTS = [2112, 2624]
NUM_TOKENS = get_ci_test_range(list(range(1, 17)) + [38], [1, 8, 16, 38])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hd_out", HD_OUTS)
@pytest.mark.parametrize("hd_in", HD_INS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_dsv3_fused_a_gemm(num_tokens, hd_in, hd_out):
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        pytest.skip("SM90+ required")

    torch.manual_seed(num_tokens)
    weight = torch.randn(hd_out, hd_in, dtype=torch.bfloat16, device="cuda")
    mat_a = torch.randn(num_tokens, hd_in, dtype=torch.bfloat16, device="cuda")
    mat_b = weight.t()

    out = dsv3_fused_a_gemm(mat_a, mat_b)
    assert out.shape == (num_tokens, hd_out)
    assert out.dtype == torch.bfloat16

    ref = (mat_a.float() @ weight.float().T).bfloat16()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
