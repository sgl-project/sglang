# SPDX-License-Identifier: Apache-2.0
"""PerChannelRMSNorm must honour the channel_dim it accepts.

The constructor fixes a default dimension, and `forward` also takes a per-call override.
Before this test the override was computed and then ignored, and a value of 0 was treated
as "unset" because the assignment used `or`.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import VaeFastPathGate
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or bool(torch.version.hip),
    reason="CUDA-only fast path",
)
@pytest.mark.parametrize("channel_dim", [0, 2])
@torch.no_grad()
def test_enabled_fast_path_honours_channel_dim_override(channel_dim):
    x = torch.randn(2, 3, 4, 5, 6, device="cuda") + 2.0
    norm = PerChannelRMSNorm(1)
    norm.fast_path_gate = VaeFastPathGate()
    norm.fast_path_gate.enabled = True
    activation = torch.nn.SiLU()
    torch.testing.assert_close(
        norm(x, channel_dim=channel_dim, activation=activation),
        activation(_reference(x, channel_dim)),
        rtol=0,
        atol=0,
    )
