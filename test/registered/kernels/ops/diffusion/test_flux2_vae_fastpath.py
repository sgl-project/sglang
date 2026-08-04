"""FLUX.2 VAE decoder fast path: kernel numerics and gate dispatch contract.

The fast paths are installed once at VAE load as wrappers that dispatch on a
request-scoped gate: requests with ``quality == "high"`` run the
near-lossless fast paths with lazily synthesized folded weights; the
``"lossless"`` default must run the original module path bit-for-bit -- this
is what keeps the CI golden outputs byte-stable.

What turns these cases red:

- statistics drift in the two-pass GroupNorm kernel (the partial reduction
  must match a full fp32 reduction within rounding of the input dtype);
- a support-check regression that silently accepts unsupported inputs
  instead of returning ``None`` (callers rely on ``None`` to fall back);
- a wrapper that consults the gate incorrectly (fast kernels leaking into
  the lossless path, a fast path that does not restore bit-exactness when
  the gate is turned back off, or eager weight folding on the lossless
  path);
- a "looks equivalent" rewrite of the weight folding math (upsample tap
  summation, attention V/proj fold) that breaks the exact re-association.
"""

import math
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.upsampling import Upsample2D

from sglang.kernels.ops.diffusion.triton.group_norm_silu_twopass import (
    group_norm_silu_4d,
    group_norm_silu_rows,
)
from sglang.multimodal_gen.runtime.models.vaes.flux2_vae_cuda_opt import (
    FusedGroupNormSiLU,
    FusedUpsample2xConv2d,
    VaeFastPathGate,
    _fold_attn_vproj,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _gn_ref(x, weight, bias, num_groups, eps, apply_silu):
    y = F.group_norm(x.float(), num_groups, weight.float(), bias.float(), eps)
    return F.silu(y) if apply_silu else y


def _psnr(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref32, got32 = ref.float(), got.float()
    mse = F.mse_loss(got32, ref32).item()
    if mse == 0:
        return float("inf")
    rng = (ref32.max() - ref32.min()).item()
    return 10 * math.log10(rng * rng / mse)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_group_norm_silu_4d_channels_last_numerics(dtype):
    torch.manual_seed(0)
    n, c, h, w = 2, 256, 96, 96
    num_groups, eps = 32, 1e-6
    weight = torch.rand(c, device=DEVICE, dtype=dtype) + 0.5
    bias = torch.rand(c, device=DEVICE, dtype=dtype) - 0.5
    x = (torch.randn(n, c, h, w, device=DEVICE, dtype=dtype) * 2 + 0.3).to(
        memory_format=torch.channels_last
    )
    ref = _gn_ref(x, weight, bias, num_groups, eps, apply_silu=True)

    y = group_norm_silu_4d(x, weight, bias, num_groups, eps, apply_silu=True)
    assert y is not None
    assert y.dtype == dtype
    assert y.is_contiguous(memory_format=torch.channels_last)
    tol = 1e-4 if dtype == torch.float32 else 6e-2
    assert (y.float() - ref).abs().max().item() < tol


@torch.no_grad()
def test_group_norm_rows_numerics():
    # apply_silu=False is the attention group_norm configuration.
    torch.manual_seed(0)
    n, rows, c = 1, 5000, 512
    num_groups, eps = 32, 1e-6
    dtype = torch.bfloat16
    weight = torch.rand(c, device=DEVICE, dtype=dtype) + 0.5
    bias = torch.rand(c, device=DEVICE, dtype=dtype) - 0.5
    x3 = torch.randn(n, rows, c, device=DEVICE, dtype=dtype) * 2 + 0.3
    # GroupNorm reference over rows: normalize per (n, group) across all rows.
    ref = _gn_ref(
        x3.transpose(1, 2), weight, bias, num_groups, eps, apply_silu=False
    ).transpose(1, 2)

    y3 = group_norm_silu_rows(x3, weight, bias, num_groups, eps, apply_silu=False)
    assert y3 is not None
    assert (y3.float() - ref).abs().max().item() < 6e-2


@torch.no_grad()
def test_group_norm_silu_unsupported_returns_none():
    torch.manual_seed(0)
    eps = 1e-6

    # NCHW-contiguous (not channels_last): fall back.
    c = 128
    weight = torch.ones(c, device=DEVICE)
    bias = torch.zeros(c, device=DEVICE)
    x = torch.randn(2, c, 16, 16, device=DEVICE)
    assert group_norm_silu_4d(x, weight, bias, 32, eps) is None

    # Non-power-of-two channel count: fall back.
    c = 96
    weight = torch.ones(c, device=DEVICE)
    bias = torch.zeros(c, device=DEVICE)
    x = torch.randn(2, c, 16, 16, device=DEVICE).to(memory_format=torch.channels_last)
    assert group_norm_silu_4d(x, weight, bias, 32, eps) is None

    # Channel counts beyond the static-shape limit: fall back.
    c = 4096
    weight = torch.ones(c, device=DEVICE)
    bias = torch.zeros(c, device=DEVICE)
    x = torch.randn(1, c, 8, 8, device=DEVICE).to(memory_format=torch.channels_last)
    assert group_norm_silu_4d(x, weight, bias, 32, eps) is None


@torch.no_grad()
def test_flux2_group_norm_silu_gate_dispatch():
    torch.manual_seed(0)
    gate = VaeFastPathGate()
    c = 128
    gn = nn.GroupNorm(32, c, eps=1e-6).to(DEVICE, torch.bfloat16)
    gn.weight.data.uniform_(0.5, 1.5)
    gn.bias.data.uniform_(-0.5, 0.5)
    x = (torch.randn(1, c, 64, 64, device=DEVICE, dtype=torch.bfloat16) * 2 + 0.3).to(
        memory_format=torch.channels_last
    )
    ref = F.silu(gn(x))
    fused = FusedGroupNormSiLU(gn, gate)

    gate.enabled = False
    assert torch.equal(fused(x), ref)

    gate.enabled = True
    fast = fused(x)
    assert fast.is_contiguous(memory_format=torch.channels_last)
    assert _psnr(ref, fast) > 45.0

    gate.enabled = False
    assert torch.equal(fused(x), ref)


@torch.no_grad()
def test_flux2_fused_upsample_gate_dispatch():
    torch.manual_seed(0)
    gate = VaeFastPathGate()
    for dtype, min_psnr in [(torch.float32, 80.0), (torch.bfloat16, 45.0)]:
        up = Upsample2D(channels=32, use_conv=True).to(DEVICE, dtype)
        fused = FusedUpsample2xConv2d(up, gate)
        x = torch.randn(2, 32, 33, 29, device=DEVICE, dtype=dtype)
        ref = up(x)

        gate.enabled = False
        assert torch.equal(fused(x), ref)
        assert fused._fused_weight is None  # folding must stay lazy

        gate.enabled = True
        fast = fused(x)
        assert fused._fused_weight is not None
        assert _psnr(ref, fast) > min_psnr

        gate.enabled = False
        assert torch.equal(fused(x), ref)


@torch.no_grad()
def test_flux2_attn_vproj_fold_math():
    """Softmax rows sum to 1, so folding the output projection into V must be
    an exact re-association: A @ (X W_v^T + b_v) W_o^T + b_o == A @ (X W'^T) + b'."""
    torch.manual_seed(0)

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_v = nn.Linear(64, 64)
            self.to_out = nn.ModuleList([nn.Linear(64, 64)])

    m = _M().to(DEVICE, torch.bfloat16)
    w, b = _fold_attn_vproj(m)
    x = torch.randn(1, 128, 64, device=DEVICE, dtype=torch.bfloat16)
    attn = torch.softmax(
        torch.randn(1, 128, 128, device=DEVICE, dtype=torch.float32), dim=-1
    ).to(torch.bfloat16)
    ref = m.to_out[0](torch.bmm(attn, m.to_v(x)))
    fast = torch.bmm(attn, F.linear(x, w)) + b
    assert _psnr(ref, fast) > 40.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
