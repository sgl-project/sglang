# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.norm.vae_channel_norm_triton import (
    bias_residual_add_triton,
    can_use_bias_residual_add_triton,
    can_use_channel_rmsnorm_silu_nhwc_triton,
    channel_rmsnorm_silu_nhwc_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=30, stage="base-a-jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

# (channels, spatial) pairs at the Qwen-Image 2.1 decoder widths; 5D adds a frame dim
SHAPES = [(1152, (64, 64)), (576, (128, 128)), (144, (256, 256)), (96, (1, 32, 32))]


def _channels_last(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    fmt = torch.channels_last if x.dim() == 4 else torch.channels_last_3d
    return x.contiguous(memory_format=fmt)


def _vector(channels, ndim):
    return (
        (torch.rand(channels, device="cuda") + 0.5)
        .to(torch.bfloat16)
        .view(channels, *([1] * (ndim - 2)))
    )


@pytest.mark.parametrize("channels,spatial", SHAPES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_channel_rmsnorm_silu_matches_eager(channels, spatial, has_bias):
    torch.manual_seed(0)
    x = _channels_last((1, channels, *spatial))
    gamma = _vector(channels, x.dim())
    bias = _vector(channels, x.dim()) if has_bias else None
    scale = channels**0.5
    assert can_use_channel_rmsnorm_silu_nhwc_triton(x, gamma, bias)
    source = x if bias is None else x + bias
    expected = F.silu(
        F.normalize(source.float(), dim=1).to(x.dtype) * scale * gamma + 0.0
    )
    actual = channel_rmsnorm_silu_nhwc_triton(x, gamma, scale, bias)
    assert actual.stride() == x.stride()
    # only the channel reduction order differs from the eager chain
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize("channels,spatial", SHAPES)
@pytest.mark.parametrize("channels_last", [True, False])
def test_bias_residual_add_matches_eager_bits(channels, spatial, channels_last):
    torch.manual_seed(0)
    shape = (1, channels, *spatial)
    make = (
        _channels_last
        if channels_last
        else (lambda s: torch.randn(s, device="cuda", dtype=torch.bfloat16))
    )
    y, h = make(shape), make(shape)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    assert can_use_bias_residual_add_triton(y, bias, h)
    expected = (y + bias.view(channels, *([1] * len(spatial)))) + h
    actual = bias_residual_add_triton(y, bias, h)
    assert actual.stride() == y.stride()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_rejects_unsupported_inputs():
    x = _channels_last((1, 96, 8, 8))
    gamma = _vector(96, 4)
    assert not can_use_channel_rmsnorm_silu_nhwc_triton(x.contiguous(), gamma)
    assert not can_use_channel_rmsnorm_silu_nhwc_triton(x.float(), gamma)
    assert not can_use_channel_rmsnorm_silu_nhwc_triton(
        _channels_last((1, 4096, 2, 2)), _vector(4096, 4)
    )
    y = torch.randn(1, 96, 8, 8, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16)
    assert not can_use_bias_residual_add_triton(
        y, bias, y.contiguous(memory_format=torch.channels_last)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
