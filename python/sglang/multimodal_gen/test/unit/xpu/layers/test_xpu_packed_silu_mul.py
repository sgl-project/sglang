# SPDX-License-Identifier: Apache-2.0
"""Tests for the XPU backend of fused_packed_silu_mul_bitexact (FLUX.2 SwiGLU)."""

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_fused_packed_silu_mul,
    fused_packed_silu_mul_bitexact,
)

_xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
pytestmark = pytest.mark.skipif(not _xpu_available, reason="XPU not available")


@pytest.mark.parametrize("hidden", [384, 9216])
def test_dense_packed_silu_mul_is_bit_exact(hidden):
    torch.manual_seed(1)
    x = torch.randn(1, 19, 2 * hidden, device="xpu", dtype=torch.bfloat16)
    assert can_use_fused_packed_silu_mul(x)
    expected = F.silu(x[..., :hidden]) * x[..., hidden:]
    assert torch.equal(fused_packed_silu_mul_bitexact(x), expected)
