# SPDX-License-Identifier: Apache-2.0
"""The VAE channel RMSNorm finish + SiLU kernel must match the eager chain bit for bit.

Regressions caught: a moved rounding point in the normalize/scale/gamma/bias
tail, a SiLU formula that differs from aten's, the sign of zero after the
``+ 0.0`` bias, wrong channel/norm indexing for NCHW and NCDHW inputs, and
predicates admitting layouts the kernel cannot address.
"""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_channel_rmsnorm_finish_silu,
    channel_rmsnorm_finish_silu,
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


def fused(x, gamma, scale):
    norm = x.float().norm(p=2, dim=1, keepdim=True)
    return channel_rmsnorm_finish_silu(x, norm, gamma, scale)


def assert_bits_equal(actual, expected):
    assert actual.dtype is expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "shape",
    [(1, 1152, 1, 64, 64), (1, 144, 1, 256, 256), (2, 288, 32, 32), (1, 4, 1, 8, 8)],
)
@pytest.mark.parametrize("amplitude", [1e-3, 1.0, 100.0])
@torch.no_grad()
def test_matches_eager(shape, amplitude):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * amplitude
    gamma_shape = (shape[1],) + (1,) * (len(shape) - 2)
    gamma = (1 + 0.1 * torch.randn(gamma_shape, device="cuda")).to(torch.bfloat16)
    scale = shape[1] ** 0.5
    assert can_use_channel_rmsnorm_finish_silu(x, gamma)
    assert_bits_equal(fused(x, gamma, scale), reference(x, gamma, scale))


@torch.no_grad()
def test_zero_input_and_negative_gamma_keep_eager_signs():
    x = torch.zeros(1, 16, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    x[0, :8] = -torch.rand(8, 1, 8, 8, device="cuda").to(torch.bfloat16)
    gamma = -torch.ones(16, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    assert_bits_equal(fused(x, gamma, 4.0), reference(x, gamma, 4.0))


@torch.no_grad()
def test_predicates_reject_unsupported_inputs():
    x = torch.randn(1, 16, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones(16, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    assert can_use_channel_rmsnorm_finish_silu(x, gamma)
    assert not can_use_channel_rmsnorm_finish_silu(x.float(), gamma.float())
    assert not can_use_channel_rmsnorm_finish_silu(x.transpose(1, 2), gamma)
    assert not can_use_channel_rmsnorm_finish_silu(x[..., :3], gamma)
    assert not can_use_channel_rmsnorm_finish_silu(x, gamma.view(16))
    assert not can_use_channel_rmsnorm_finish_silu(x, gamma.float())
    assert not can_use_channel_rmsnorm_finish_silu(x.cpu(), gamma.cpu())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
