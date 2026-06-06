# SPDX-License-Identifier: Apache-2.0
"""Fused MiniMax-M3 per-head Gemma Q/K RMSNorm + partial RoPE for ROCm."""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_gemma_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_stride_m,
    q_stride_d,
    k_stride_m,
    k_stride_d,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    is_neox_style: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_program = tl.program_id(1)
    cols = tl.arange(0, BLOCK_HD)
    mask = cols < head_dim
    half_rotary: tl.constexpr = rotary_dim // 2

    is_q = head_program < q_heads
    head_id = tl.where(is_q, head_program, head_program - q_heads)
    in_ptr = tl.where(is_q, q_ptr, k_ptr)
    out_ptr = tl.where(is_q, q_out_ptr, k_out_ptr)
    weight_ptr = tl.where(is_q, q_weight_ptr, k_weight_ptr)
    stride_m = tl.where(is_q, q_stride_m, k_stride_m)
    stride_d = tl.where(is_q, q_stride_d, k_stride_d)
    n_heads = tl.where(is_q, q_heads, k_heads)

    base_in = in_ptr + token_id * stride_m + head_id * head_dim * stride_d
    x = tl.load(base_in + cols * stride_d, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    rstd = tl.rsqrt(var + eps)
    normed = x * rstd * (1.0 + w)
    # Match the unfused path: GemmaRMSNorm writes bf16/fp16, then RoPE reads
    # that rounded value in the following kernel.
    normed = normed.to(q_out_ptr.dtype.element_ty).to(tl.float32)

    rotary_mask = cols < rotary_dim
    if is_neox_style:
        partner_cols = tl.where(
            cols < half_rotary, cols + half_rotary, cols - half_rotary
        )
        cos_cols = tl.where(cols < half_rotary, cols, cols - half_rotary)
        sign = tl.where(cols < half_rotary, -1.0, 1.0)
    else:
        partner_cols = tl.where((cols % 2) == 0, cols + 1, cols - 1)
        cos_cols = cols // 2
        sign = tl.where((cols % 2) == 0, -1.0, 1.0)

    partner_mask = partner_cols < head_dim
    x_partner = tl.load(
        base_in + partner_cols * stride_d,
        mask=partner_mask,
        other=0.0,
    ).to(tl.float32)
    w_partner = tl.load(
        weight_ptr + partner_cols,
        mask=partner_mask,
        other=0.0,
    ).to(tl.float32)
    partner_normed = x_partner * rstd * (1.0 + w_partner)
    partner_normed = partner_normed.to(q_out_ptr.dtype.element_ty).to(tl.float32)

    pos = tl.load(positions_ptr + token_id).to(tl.int64)
    cos_sin_base = cos_sin_cache_ptr + pos * rotary_dim
    cos = tl.load(cos_sin_base + cos_cols, mask=rotary_mask, other=1.0).to(tl.float32)
    sin = tl.load(
        cos_sin_base + half_rotary + cos_cols,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    rotated = normed * cos + sign * partner_normed * sin
    out = tl.where(rotary_mask, rotated, normed)

    base_out = out_ptr + token_id * n_heads * head_dim + head_id * head_dim
    tl.store(base_out + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def qk_gemma_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized+rotated Q/K tensors with the same shapes as ``q``/``k``."""
    assert q.dim() == 2 and k.dim() == 2
    assert positions.dim() == 1
    assert q.shape[0] == k.shape[0] == positions.shape[0]
    assert q.shape[1] % head_dim == 0
    assert k.shape[1] % head_dim == 0
    assert rotary_dim <= head_dim and rotary_dim % 2 == 0

    q_heads = q.shape[1] // head_dim
    k_heads = k.shape[1] // head_dim
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_hd = triton.next_power_of_2(head_dim)

    _qk_gemma_rmsnorm_rope_kernel[(q.shape[0], q_heads + k_heads)](
        q,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q_heads,
        k_heads,
        head_dim,
        rotary_dim,
        eps,
        is_neox_style,
        BLOCK_HD=block_hd,
        num_warps=4,
    )
    return q_out, k_out
