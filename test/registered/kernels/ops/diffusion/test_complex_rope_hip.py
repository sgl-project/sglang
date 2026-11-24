# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 Triton RoPE kernels on ROCm, where they match eager to rounding error."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.rope.complex_rope_triton import fused_complex_rope
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import (
    qknorm_complex_rope,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="base-a-jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm required",
)


def _rope(seq_len, half):
    angle = torch.rand(seq_len, half, device="cuda") * 6.28
    return torch.polar(torch.ones_like(angle), angle)


def _eager_rope(x, rope):
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)


def _eager_rmsnorm(x, weight, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype) * weight


@pytest.mark.parametrize("shape", [(1, 2048, 32, 128), (1, 7, 4, 128)])
def test_complex_rope_matches_eager(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    rope = _rope(shape[1], shape[-1] // 2)
    torch.testing.assert_close(
        fused_complex_rope(x, rope), _eager_rope(x, rope), atol=2e-2, rtol=1e-2
    )


@pytest.mark.parametrize("shape", [(1, 2048, 32, 128), (1, 7, 4, 128)])
def test_qknorm_complex_rope_matches_eager(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
    rope = _rope(shape[1], 64)
    expected = _eager_rope(_eager_rmsnorm(x, weight, 1e-6), rope)
    torch.testing.assert_close(
        qknorm_complex_rope(x, weight, rope, 1e-6), expected, atol=3e-2, rtol=2e-2
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
