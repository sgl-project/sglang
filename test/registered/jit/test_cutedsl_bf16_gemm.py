"""Tests for the CuTe DSL TGV BF16 GEMM kernel."""

import sys

import pytest
import torch

from sglang.kernels.jit.utils import (
    get_ci_test_range,
    get_jit_cuda_arch,
    is_hip_runtime,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.jit_kernel.cutedsl_bf16_gemm import cutedsl_bf16_gemm  # noqa: E402

N_VALUES = [1024, 2624, 6144]
K_VALUES = [2048, 6144]
NUM_TOKENS = get_ci_test_range(list(range(1, 33)), [1, 15, 16, 32])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_cutedsl_bf16_gemm(num_tokens, k, n, has_bias):
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM100/SM103 required")

    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") if has_bias else None

    out = cutedsl_bf16_gemm(x, weight, bias)
    assert out.shape == (num_tokens, n)
    assert out.dtype == torch.bfloat16

    ref = x.float() @ weight.float().T
    if bias is not None:
        ref = ref + bias.float()
    torch.testing.assert_close(out, ref.bfloat16(), rtol=2e-2, atol=2.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
