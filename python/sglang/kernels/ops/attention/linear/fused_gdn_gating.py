from sglang.kernels.ops.extend_fastpath import GDN_GATING_ENABLED as _Q38C_EXTEND_PATH

import os
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
# beta_output = b.sigmoid()
@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + i_b * stride_a + head_off, mask=mask)
    blk_b = tl.load(b + i_b * stride_b + head_off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


# gdn_front: 2D (tokens x heads) launch for extend-sized batches. The stock
# launch is one 32-thread program per (token, 8 heads); this one covers BT
# tokens x all heads per program with the same per-element math (bit-identical).
# Kill switch: SGLANG_X_GDN_FRONT_DISABLE=1 or SGLANG_X_GDN_FRONT_GATING_DISABLE=1.
_GDN_FRONT_GATING = _Q38C_EXTEND_PATH
# SGLANG_GDN_PREFILL_ALPHA_FUSED (default off): the FlashInfer prefill takes exp(g) from the
# gating kernel instead of a torch.exp launch per GDN layer; needs the 2D kernel above.
ALPHA_FUSED = _Q38C_EXTEND_PATH and os.environ.get(
    "SGLANG_GDN_PREFILL_ALPHA_FUSED", "0"
).strip().lower() in ("1", "true", "yes", "on")
GATING_BIG_BATCH = 512
GATING_BIG_CFG = (16, 2)  # (BT, num_warps)


@triton.jit(do_not_specialize=["T"])
def fused_gdn_gating_2d_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    T,
    stride_a,
    stride_b,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    EXP_G: tl.constexpr = False,
):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    heads = tl.arange(0, BH)
    hmask = heads < NUM_HEADS
    mask = (rows < T)[:, None] & hmask[None, :]
    rows64 = rows.to(tl.int64)[:, None]
    blk_A_log = tl.load(A_log + heads, mask=hmask)
    blk_bias = tl.load(dt_bias + heads, mask=hmask)
    blk_a = tl.load(a + rows64 * stride_a + heads[None, :], mask=mask)
    blk_b = tl.load(b + rows64 * stride_b + heads[None, :], mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32))[None, :] * softplus_x
    off = rows64 * NUM_HEADS + heads[None, :]
    if EXP_G:
        # The FlashInfer prefill's torch.exp(g): libdevice expf on the stored fp32 g.
        blk_g = libdevice.exp(blk_g.to(g.dtype.element_ty).to(tl.float32))
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


# Import-time selection keeps the disabled entry and launch body stock.
if _Q38C_EXTEND_PATH:
    def fused_gdn_gating(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
        launch_cfg=None,
        exp_g: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """exp_g: return (exp(g), beta) for the FlashInfer prefill (SGLANG_GDN_PREFILL_ALPHA_FUSED)."""
        batch, num_heads = a.shape
        seq_len = 1
        stride_a = a.stride(0)
        stride_b = b.stride(0)
        grid = (batch, seq_len, triton.cdiv(num_heads, 8))
        g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
        beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
        if exp_g or (_GDN_FRONT_GATING and batch >= GATING_BIG_BATCH) or launch_cfg is not None:
            # Both launches compute each element alike; exp_g takes the 2D one at any size.
            bt, n_warps = launch_cfg if launch_cfg is not None else GATING_BIG_CFG
            fused_gdn_gating_2d_kernel[(triton.cdiv(batch, bt),)](
                g,
                beta_output,
                A_log,
                a,
                b,
                dt_bias,
                batch,
                stride_a,
                stride_b,
                num_heads,
                beta,
                threshold,
                BT=bt,
                BH=triton.next_power_of_2(num_heads),
                num_warps=n_warps,
                **({"EXP_G": True} if exp_g else {}),
            )
            return g, beta_output
        fused_gdn_gating_kernel[grid](
            g,
            beta_output,
            A_log,
            a,
            b,
            dt_bias,
            seq_len,
            stride_a,
            stride_b,
            num_heads,
            beta,
            threshold,
            8,
            num_warps=1,
        )
        return g, beta_output
else:
    def fused_gdn_gating(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, num_heads = a.shape
        seq_len = 1
        stride_a = a.stride(0)
        stride_b = b.stride(0)
        grid = (batch, seq_len, triton.cdiv(num_heads, 8))
        g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
        beta_output = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
        fused_gdn_gating_kernel[grid](
            g,
            beta_output,
            A_log,
            a,
            b,
            dt_bias,
            seq_len,
            stride_a,
            stride_b,
            num_heads,
            beta,
            threshold,
            8,
            num_warps=1,
        )
        return g, beta_output
