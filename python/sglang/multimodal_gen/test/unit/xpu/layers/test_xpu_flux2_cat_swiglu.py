# SPDX-License-Identifier: Apache-2.0
"""Tests for the FLUX.2 single-block SwiGLU written straight into the to_out input on XPU."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import sglang.multimodal_gen.runtime.models.dits.flux_2 as flux2
from sglang.multimodal_gen.runtime.models.dits.flux_2 import _flux2_cat_swiglu

_xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
pytestmark = pytest.mark.skipif(not _xpu_available, reason="XPU not available")


def _single_tail_inputs(hidden: int, ff: int, device: str = "xpu"):
    torch.manual_seed(2)
    attn = torch.randn(1, 19, hidden, device=device, dtype=torch.bfloat16)
    # The single block's gate/up is a strided split view of the fused projection.
    fused = torch.randn(1, 19, 3 * hidden + 2 * ff, device=device, dtype=torch.bfloat16)
    return attn, fused[..., 3 * hidden :]


@pytest.mark.parametrize("hidden,ff", [(96, 128), (3072, 9216)])
def test_cat_swiglu_is_bit_exact_vs_cat(hidden, ff):
    attn, gate_up = _single_tail_inputs(hidden, ff)
    assert not gate_up.is_contiguous()
    expected = torch.cat([attn, F.silu(gate_up[..., :ff]) * gate_up[..., ff:]], dim=-1)
    actual = _flux2_cat_swiglu(attn, gate_up)
    assert actual is not None and actual.is_contiguous()
    assert torch.equal(actual, expected)


def test_cat_swiglu_declines_where_the_cat_is_fused():
    attn, gate_up = _single_tail_inputs(96, 128)
    with patch.object(flux2, "can_use_fused_packed_silu_mul", return_value=True):
        assert _flux2_cat_swiglu(attn, gate_up) is None
    with patch("torch.compiler.is_compiling", return_value=True):
        assert _flux2_cat_swiglu(attn, gate_up) is None
    assert _flux2_cat_swiglu(attn.float(), gate_up) is None
