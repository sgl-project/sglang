"""Unit tests for the gfx942 W4A16 clamped SwiGLU-OAI helper.

Covers the activation added for MiniMax-M3 / gpt-oss MXFP4 MoE on the W4A16 path.
Fails on base (the helper does not exist there).
"""

import pytest
import torch

from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    _clamped_swiglu_oai,
)


def _reference(gate_up, inter, alpha, limit):
    gu = gate_up.view(-1, 2 * inter).double()
    gate = gu[..., :inter].clamp(max=limit)
    up = gu[..., inter:].clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (up + 1)


@pytest.mark.parametrize("alpha,limit", [(1.702, 7.0), (1.0, 3.0)])
def test_clamped_swiglu_matches_reference(alpha, limit):
    torch.manual_seed(0)
    rows, inter = 8, 16
    gate_up = torch.randn(rows, 2 * inter) * 10.0  # wide range exercises the clamp
    out = _clamped_swiglu_oai(gate_up, inter, alpha, limit)
    assert out.shape == (rows, inter)
    torch.testing.assert_close(
        out.double(), _reference(gate_up, inter, alpha, limit), rtol=1e-5, atol=1e-5
    )


def test_clamped_swiglu_clamps_gate_and_up():
    inter = 4
    gate_up = torch.cat([torch.full((1, inter), 100.0), torch.zeros(1, inter)], dim=-1)
    out = _clamped_swiglu_oai(gate_up, inter, alpha=1.702, limit=7.0)
    expected = 7.0 * torch.sigmoid(torch.tensor(1.702 * 7.0)) * 1.0
    torch.testing.assert_close(
        out, torch.full((1, inter), float(expected)), rtol=1e-4, atol=1e-4
    )
