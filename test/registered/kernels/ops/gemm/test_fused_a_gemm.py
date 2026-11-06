"""Tests for the fused-A GEMM dispatcher's eligibility check."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.fused_a_gemm import (
    fused_a_gemm_weight_eligible,
    linear_with_fused_a_gemm,
)
from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HD_OUT = 2112


class _Linear(torch.nn.Linear):
    # SGLang linear layers return (output, output_bias).
    def forward(self, x):
        return super().forward(x), None


@pytest.mark.skipif(
    not is_sm120_supported(), reason="auto backend is CuTe DSL only on SM120"
)
@pytest.mark.parametrize(
    "hd_in, bias, expect_fused",
    [(3200, False, False), (3584, False, True), (3584, True, False)],
)
@pytest.mark.parametrize("num_tokens", [1, 16])
def test_fused_a_gemm_eligibility(num_tokens, hd_in, bias, expect_fused):
    """On SM120 a 256-aligned K runs the fused CuTe DSL kernel and matches F.linear;
    a K that is not 256-aligned, or a biased layer, takes the unfused path, so it
    neither asserts in the kernel nor drops the bias."""
    torch.manual_seed(num_tokens)
    layer = _Linear(hd_in, HD_OUT, bias=bias, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(num_tokens, hd_in, dtype=torch.bfloat16, device="cuda")

    eligible = fused_a_gemm_weight_eligible(layer)
    assert eligible == expect_fused
    with torch.no_grad():
        if bias:
            # Larger than the tolerance below, so a dropped bias fails the check.
            layer.bias.fill_(1.0)
        out = linear_with_fused_a_gemm(layer, x) if eligible else layer(x)[0]
        ref = F.linear(x, layer.weight, layer.bias)

    assert out.shape == (num_tokens, HD_OUT)
    # nn.Linear's init keeps outputs O(1); this allows a few bf16 ulps.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
