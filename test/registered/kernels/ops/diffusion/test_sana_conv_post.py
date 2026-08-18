"""Sana GLUMB post-processing fusions must match the eager bf16 chain."""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.triton.sana_conv_post import (
    can_use_fused_bias_glu,
    can_use_fused_bias_silu,
    fused_bias_glu,
    fused_bias_silu,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("channels", [2240, 11200])
def test_sana_bias_silu_is_bit_exact(channels):
    torch.manual_seed(0)
    x = torch.randn(
        (1, channels, 7, 5),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(memory_format=torch.channels_last)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)

    assert can_use_fused_bias_silu(x, bias)
    actual = fused_bias_silu(x, bias)
    expected = F.silu(x + bias[None, :, None, None])

    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("channels", [2240, 5600])
def test_sana_bias_glu_is_bit_exact(channels):
    torch.manual_seed(1)
    x = torch.randn(
        (1, 2 * channels, 7, 5),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(memory_format=torch.channels_last)
    bias = torch.randn(2 * channels, device="cuda", dtype=torch.bfloat16)

    assert can_use_fused_bias_glu(x, bias)
    actual = fused_bias_glu(x, bias)
    biased = x + bias[None, :, None, None]
    hidden, gate = torch.chunk(biased, 2, dim=1)
    expected = hidden * F.silu(gate)

    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert torch.equal(actual, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
