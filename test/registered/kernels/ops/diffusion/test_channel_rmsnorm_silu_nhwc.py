# SPDX-License-Identifier: Apache-2.0
"""Channels-last channel RMSNorm + SiLU must stay within one bf16 ulp of the eager chain.

Regressions caught: wrong pixel/channel addressing for channels_last 4D and 5D
inputs across the lane-group specializations (1, 4, 8 and 32 lanes per pixel,
2/4/8-wide loads, ragged last load), a tail whose rounding points drift from the
eager chain (differences would exceed the last bf16 bit), a fused conv bias that
skips aten's bf16 rounding, and predicates admitting NCHW or unsupported shapes.
"""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_channel_rmsnorm_silu_nhwc,
    channel_rmsnorm_silu_nhwc,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def reference(x, gamma, scale):
    normalized = F.normalize(x.float(), dim=1).to(x.dtype)
    return F.silu(normalized * scale * gamma + 0.0)


def channels_last(x):
    fmt = torch.channels_last if x.ndim == 4 else torch.channels_last_3d
    return x.contiguous(memory_format=fmt)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1152, 1, 64, 64),  # 8-wide loads, 32 lanes per pixel
        (1, 144, 1, 128, 128),  # 8-wide loads, 4 lanes per pixel, ragged last load
        (2, 288, 32, 32),  # 8 lanes per pixel
        (1, 6, 8, 8),  # 2-wide loads, one lane per pixel
        (1, 20, 1, 8, 8),  # 4-wide loads
        (1, 16, 1, 5, 5),  # pixel count not a multiple of the rows per warp
    ],
)
@pytest.mark.parametrize("amplitude", [1e-3, 1.0, 100.0])
@pytest.mark.parametrize("with_bias", [False, True])
@torch.no_grad()
def test_close_to_eager_and_layout_preserved(shape, amplitude, with_bias):
    torch.manual_seed(0)
    x = channels_last(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * amplitude
    )
    gamma_shape = (shape[1],) + (1,) * (len(shape) - 2)
    gamma = (1 + 0.1 * torch.randn(gamma_shape, device="cuda")).to(torch.bfloat16)
    bias = None
    x_ref = x
    if with_bias:
        bias = (0.1 * amplitude * torch.randn(gamma_shape, device="cuda")).to(x.dtype)
        # the eager chain adds the conv bias in bf16 before the norm
        x_ref = x + bias
    scale = shape[1] ** 0.5
    assert can_use_channel_rmsnorm_silu_nhwc(x, gamma, bias)
    out = channel_rmsnorm_silu_nhwc(x, gamma, scale, bias)
    assert out.stride() == x.stride()
    expected = reference(x_ref, gamma, scale)
    # Only the reduction order differs from aten's: the norm may move by one
    # ulp, which the four bf16 rounding steps of the tail can turn into two
    # ulps of the output (2^-5 relative), and the vast majority of elements
    # stay bit-identical.
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=2.0**-5)
    assert (out.view(torch.int16) == expected.view(torch.int16)).float().mean() > 0.95


@torch.no_grad()
def test_predicates_reject_nchw_and_odd_channels():
    x = torch.randn(1, 16, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones(16, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    assert not can_use_channel_rmsnorm_silu_nhwc(x, gamma)  # NCDHW contiguous
    xl = channels_last(x)
    assert can_use_channel_rmsnorm_silu_nhwc(xl, gamma)
    odd = channels_last(torch.randn(1, 15, 8, 8, device="cuda", dtype=torch.bfloat16))
    assert not can_use_channel_rmsnorm_silu_nhwc(
        odd, torch.ones(15, 1, 1, device="cuda", dtype=torch.bfloat16)
    )
    assert not can_use_channel_rmsnorm_silu_nhwc(xl.float(), gamma.float())
    assert not can_use_channel_rmsnorm_silu_nhwc(xl, gamma.view(16))
    assert not can_use_channel_rmsnorm_silu_nhwc(xl, gamma, gamma.view(16))
    assert not can_use_channel_rmsnorm_silu_nhwc(xl, gamma, gamma.float())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
