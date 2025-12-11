from typing import Tuple

import torch
import triton
import triton.language as tl


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
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(b.dtype.element_ty), mask=mask)


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
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output


@triton.jit
def fused_gdn_gating_kernel_v3(
    g,
    A_log,
    a,
    dt_bias,
    batch,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    batch_off = i_b * BLK_BATCHES + tl.arange(0, BLK_BATCHES)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    head_mask = head_off < NUM_HEADS

    a_off = (
        batch_off[:, None] * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off[None, :]
    )
    a_mask = (batch_off[:, None] < batch) & head_mask[None, :]

    blk_A_log = tl.load(A_log + head_off, mask=head_mask)
    blk_bias = tl.load(dt_bias + head_off, mask=head_mask)

    blk_a = tl.load(a + a_off, mask=a_mask)

    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + a_off, blk_g.to(g.dtype.element_ty), mask=a_mask)


def fused_gdn_gating_v3(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape
    seq_len = 1
    g = torch.empty_like(a, dtype=torch.float32)

    num_cores = 48  # num_vectorcore of NPU
    NUM_BLK_BATCHES = triton.cdiv(num_cores, triton.cdiv(num_heads, 8))
    BLK_BATCHES = triton.cdiv(batch, NUM_BLK_BATCHES)
    grid = (NUM_BLK_BATCHES, seq_len, triton.cdiv(num_heads, 8))
    fused_gdn_gating_kernel_v3[grid](
        g,
        A_log,
        a,
        dt_bias,
        batch,
        seq_len,
        num_heads,
        beta,
        threshold,
        BLK_BATCHES=BLK_BATCHES,
        BLK_HEADS=8,
        num_warps=1,
    )
    g = g.unsqueeze(0)
    return g, b.sigmoid().unsqueeze(0)
