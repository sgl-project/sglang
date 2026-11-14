# SPDX-License-Identifier: Apache-2.0
"""PerChannelRMSNorm must honour the channel_dim it accepts.

The constructor fixes a default dimension, and `forward` also takes a per-call override.
Before this test the override was computed and then ignored, and a value of 0 was treated
as "unset" because the assignment used `or`.
"""

import torch

from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import PerChannelRMSNorm

EPS = 1e-8


def _reference(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x / torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + EPS)


def test_default_uses_the_configured_dimension():
    x = torch.randn(2, 3, 4) + 2.0
    torch.testing.assert_close(PerChannelRMSNorm(1)(x), _reference(x, 1))


def test_forward_override_is_honoured():
    x = torch.randn(2, 3, 4) + 2.0
    torch.testing.assert_close(PerChannelRMSNorm(1)(x, channel_dim=2), _reference(x, 2))


def test_zero_override_is_honoured():
    x = torch.randn(2, 3, 4) + 2.0
    torch.testing.assert_close(PerChannelRMSNorm(2)(x, channel_dim=0), _reference(x, 0))
