# -*- coding: utf-8 -*-
"""Fused sigmoid + gdn_gating kernel for prefill."""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_gdn_gating_sigmoid_kernel(
    g_ptr, beta_ptr, A_log_ptr, a_ptr, b_ptr, dt_bias_ptr,
    NUM_HEADS: tl.constexpr,
    softplus_beta: tl.constexpr, softplus_threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    """Fused: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)."""
    i_s = tl.program_id(0)
    
    head_off = tl.arange(0, BLK_HEADS)
    mask = head_off < NUM_HEADS
    off = i_s * NUM_HEADS + head_off
    
    blk_A_log = tl.load(A_log_ptr + head_off, mask=mask)
    blk_dt_bias = tl.load(dt_bias_ptr + head_off, mask=mask)
    blk_a = tl.load(a_ptr + off, mask=mask)
    blk_b = tl.load(b_ptr + off, mask=mask)
    
    # g = -exp(A_log) * softplus(a + dt_bias)
    x = blk_a.to(tl.float32) + blk_dt_bias.to(tl.float32)
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    
    # beta = sigmoid(b)
    blk_beta = 1.0 / (1.0 + tl.exp(-blk_b.to(tl.float32)))
    
    tl.store(g_ptr + off, blk_g.to(g_ptr.dtype.element_ty), mask=mask)
    tl.store(beta_ptr + off, blk_beta.to(beta_ptr.dtype.element_ty), mask=mask)


def fused_gdn_gating_and_sigmoid(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused g and beta computation in single kernel."""
    seq_len, num_heads = a.shape
    g = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    
    BLK_HEADS = triton.next_power_of_2(num_heads)
    grid = (seq_len,)
    
    fused_gdn_gating_sigmoid_kernel[grid](
        g, beta, A_log, a, b, dt_bias,
        num_heads, softplus_beta, softplus_threshold, BLK_HEADS,
        num_warps=2,
    )
    
    return g, beta
