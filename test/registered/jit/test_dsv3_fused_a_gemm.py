"""Tests for JIT dsv3_fused_a_gemm kernel."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.dsv3_fused_a_gemm import dsv3_fused_a_gemm
from sglang.jit_kernel.utils import get_ci_test_range, get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# hd_in must be a multiple of 256; 2048/6144/7168 cover the real fused-A shapes
# (2048 is q_b_proj TP4/TP8, 6144/7168 are qkv_a).
HD_INS = [2048, 6144, 7168]
# hd_out must be a multiple of 16; 2048/2112/2624/4096 cover real fused-A variants
# (2048/4096 are q_b_proj TP8/TP4, 2112/2624 are qkv_a).
HD_OUTS = [2048, 2112, 2624, 4096]
NUM_TOKENS = get_ci_test_range(list(range(1, 17)), [1, 8, 16])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hd_out", HD_OUTS)
@pytest.mark.parametrize("hd_in", HD_INS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_dsv3_fused_a_gemm(num_tokens, hd_in, hd_out):
    if is_hip_runtime() or get_jit_cuda_arch().major < 9:
        pytest.skip("SM90+ required")

    mat_a = torch.randn(num_tokens, hd_in, dtype=torch.bfloat16, device="cuda")
    mat_b = torch.randn(hd_out, hd_in, dtype=torch.bfloat16, device="cuda").transpose(
        0, 1
    )

    ref = F.linear(mat_a, mat_b.T)
    out = dsv3_fused_a_gemm(mat_a, mat_b)

    assert out.shape == (num_tokens, hd_out)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
