# SPDX-License-Identifier: Apache-2.0
"""Ascend fusions that back Qwen-Image 2.1's `exact` tier.

The kernels ship in sgl-kernel-npu; these cases check that the installed wheel
carries them and that each one matches the eager chain the model runs when the
fusion gate declines. Elementwise ops must reproduce every eager bf16 rounding
boundary, and RMSNorm must reproduce aten's row reduction, not an equivalent
one.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    residual_gate_add_npu,
    rmsnorm_npu_onepass,
    silu_mul_npu,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=40, suite="full-1-npu-a3")

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU required",
)

DEVICE = torch.device("npu")


@pytest.mark.parametrize("shape", [(1, 128), (1, 4096, 12288)])
def test_silu_mul_matches_eager(shape):
    torch.manual_seed(42)
    a = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    expected = torch.nn.functional.silu(a) * b
    assert torch.equal(silu_mul_npu(a, b), expected)


@pytest.mark.parametrize("per_row", [False, True])
def test_residual_gate_add_matches_eager(per_row):
    torch.manual_seed(42)
    residual = torch.randn(1, 4096, 4096, device=DEVICE, dtype=torch.bfloat16)
    update = torch.randn_like(residual)
    shape = residual.shape if per_row else (1, 1, 4096)
    gate = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    expected = residual + update * gate
    assert torch.equal(residual_gate_add_npu(residual, update, gate), expected)


def _reference_norm(shape, scale, eps):
    dim = shape[-1]
    x = (torch.randn(shape, device=DEVICE) * scale).to(torch.bfloat16)
    norm = RMSNorm(
        dim,
        eps,
        cast_x_before_out_mul=True,
        force_native=True,
        weight_dtype=torch.bfloat16,
    ).to(DEVICE)
    torch.nn.init.normal_(norm.weight, mean=1.0, std=0.1)
    return x, norm.weight, norm(x)


@pytest.mark.parametrize("shape", [(16, 128), (1, 32, 128), (1, 4096, 32, 128)])
@pytest.mark.parametrize("scale,eps", [(1e-4, 1e-6), (100.0, 1e-5)])
def test_rmsnorm_matches_native(shape, scale, eps):
    x, weight, expected = _reference_norm(shape, scale, eps)
    assert torch.equal(rmsnorm_npu_onepass(x, weight, eps), expected)


def test_rmsnorm_refuses_tiles_under_16_rows():
    # triton_ascend 3.2.2 computed these tiles incorrectly; the kernel must
    # decline so the caller keeps the eager result instead of publishing one.
    x = torch.randn(8, 128, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.ones(128, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        rmsnorm_npu_onepass(x, weight, 1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
