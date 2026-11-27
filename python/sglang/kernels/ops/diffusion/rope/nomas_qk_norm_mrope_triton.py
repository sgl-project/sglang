# SPDX-License-Identifier: Apache-2.0
"""Nomas PixelUMM Q/K normalization and three-axis RoPE, with FP32 products."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.numerics import mul_rn_f32
from sglang.kernels.ops.diffusion.common.fast_launch import replay_launches


@triton.jit
def _qk_norm_mrope_kernel(
    X,
    MT,
    MH,
    WT,
    WH,
    COS,
    SIN,
    OUT,
    N: tl.constexpr,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    BS: tl.constexpr,
    TS: tl.constexpr,
    HS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = off < N
    col = off % 128
    row = off // 128
    token = (row // HEADS) % TOKENS
    batch = row // (HEADS * TOKENS)
    head = row % HEADS
    src = batch * BS + token * TS + head * HS
    half = tl.where(col < 64, 32, 16)
    partner = col ^ half
    x = tl.load(X + src + col, valid, 0).to(tl.float32)
    xp = tl.load(X + src + partner, valid, 0).to(tl.float32)
    wt = tl.load(WT + col, valid & (col < 64), 0).to(tl.float32)
    wh = tl.load(WH + col - 64, valid & (col >= 64), 0).to(tl.float32)
    wtp = tl.load(WT + partner, valid & (partner < 64), 0).to(tl.float32)
    whp = tl.load(WH + partner - 64, valid & (partner >= 64), 0).to(tl.float32)
    mt = tl.load(MT + row, valid, 0)
    mh = tl.load(MH + row, valid, 0)
    rstd = libdevice.rsqrt(tl.where(col < 64, mt, mh) + EPS)
    normal = mul_rn_f32(tl.where(col < 64, wt, wh), mul_rn_f32(x, rstd))
    rotated = mul_rn_f32(tl.where(partner < 64, wtp, whp), mul_rn_f32(xp, rstd))
    rotated = tl.where((col & half) == 0, -rotated, rotated)
    c = tl.load(COS + token * 128 + col, valid, 0)
    s = tl.load(SIN + token * 128 + col, valid, 0)
    # The compiled parameter graph rounds the sine product, then contracts
    # the cosine product and sum on Blackwell. Keep that operand order.
    result = tl.fma(normal, c, mul_rn_f32(rotated, s))
    tl.store(OUT + off, result, valid)


@replay_launches
def fused_nomas_qk_norm_mrope(
    x: torch.Tensor,
    mean_t: torch.Tensor,
    mean_hw: torch.Tensor,
    eps: float,
    weight_t: torch.Tensor,
    weight_hw: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reuse ATen means and the compiled parameter graph's FP32 rounding."""
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and x.ndim == 4
        and all(size > 0 for size in x.shape)
        and x.shape[-1] == 128
        and x.stride(-1) == 1
    ):
        raise ValueError("Expected CUDA BF16 [B,S,H,128] with contiguous columns")
    batch, tokens, heads, _ = x.shape
    for mean in (mean_t, mean_hw):
        if not (
            mean.device == x.device
            and mean.dtype == torch.float32
            and mean.shape == (batch, tokens, heads, 1)
            and mean.is_contiguous()
        ):
            raise ValueError(
                "Expected contiguous FP32 [B,S,H,1] mean squares on the input device"
            )
    for weight in (weight_t, weight_hw):
        if not (
            weight.device == x.device
            and weight.dtype == torch.bfloat16
            and weight.shape == (64,)
            and weight.is_contiguous()
        ):
            raise ValueError(
                "Expected contiguous BF16 [64] weights on the input device"
            )
    for table in (cos, sin):
        if not (
            table.device == x.device
            and table.dtype == torch.float32
            and table.shape == (1, tokens, 1, 128)
            and table.is_contiguous()
        ):
            raise ValueError(
                "Expected contiguous FP32 [1,S,1,128] RoPE tables on the input device"
            )
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    with torch.cuda.device(x.device):
        _qk_norm_mrope_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            mean_t,
            mean_hw,
            weight_t,
            weight_hw,
            cos,
            sin,
            out,
            x.numel(),
            tokens,
            heads,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            eps,
            1024,
            enable_fp_fusion=False,
        )
    return out
