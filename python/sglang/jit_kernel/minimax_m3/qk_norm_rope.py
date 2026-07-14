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


@triton.jit
def _sparse_qk_index_gemma_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    idx_q_ptr,
    idx_k_ptr,
    q_out_ptr,
    k_out_ptr,
    idx_q_out_ptr,
    idx_k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    idx_q_weight_ptr,
    idx_k_weight_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_stride_m,
    q_stride_d,
    k_stride_m,
    k_stride_d,
    idx_q_stride_m,
    idx_q_stride_d,
    idx_k_stride_m,
    idx_k_stride_d,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    idx_q_heads: tl.constexpr,
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

    main_heads: tl.constexpr = q_heads + k_heads
    idx_k_program: tl.constexpr = q_heads + k_heads + idx_q_heads

    is_q = head_program < q_heads
    is_k = (head_program >= q_heads) & (head_program < main_heads)
    is_idx_q = (head_program >= main_heads) & (head_program < idx_k_program)

    head_id = tl.where(
        is_q,
        head_program,
        tl.where(
            is_k,
            head_program - q_heads,
            tl.where(is_idx_q, head_program - main_heads, 0),
        ),
    )

    in_ptr = tl.where(
        is_q,
        q_ptr,
        tl.where(is_k, k_ptr, tl.where(is_idx_q, idx_q_ptr, idx_k_ptr)),
    )
    out_ptr = tl.where(
        is_q,
        q_out_ptr,
        tl.where(is_k, k_out_ptr, tl.where(is_idx_q, idx_q_out_ptr, idx_k_out_ptr)),
    )
    weight_ptr = tl.where(
        is_q,
        q_weight_ptr,
        tl.where(
            is_k,
            k_weight_ptr,
            tl.where(is_idx_q, idx_q_weight_ptr, idx_k_weight_ptr),
        ),
    )
    stride_m = tl.where(
        is_q,
        q_stride_m,
        tl.where(
            is_k,
            k_stride_m,
            tl.where(is_idx_q, idx_q_stride_m, idx_k_stride_m),
        ),
    )
    stride_d = tl.where(
        is_q,
        q_stride_d,
        tl.where(
            is_k,
            k_stride_d,
            tl.where(is_idx_q, idx_q_stride_d, idx_k_stride_d),
        ),
    )
    out_heads = tl.where(
        is_q,
        q_heads,
        tl.where(is_k, k_heads, tl.where(is_idx_q, idx_q_heads, 1)),
    )

    base_in = in_ptr + token_id * stride_m + head_id * head_dim * stride_d
    x = tl.load(base_in + cols * stride_d, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    rstd = tl.rsqrt(var + eps)
    normed = x * rstd * (1.0 + w)
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

    base_out = out_ptr + token_id * out_heads * head_dim + head_id * head_dim
    tl.store(base_out + cols, out.to(q_out_ptr.dtype.element_ty), mask=mask)


def sparse_qk_index_gemma_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    idx_q_weight: torch.Tensor,
    idx_k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse main and sparse-index Gemma Q/K RMSNorm + RoPE into one launch."""
    assert q.dim() == k.dim() == idx_q.dim() == idx_k.dim() == 2
    assert positions.dim() == 1
    assert q.shape[0] == k.shape[0] == idx_q.shape[0] == idx_k.shape[0]
    assert q.shape[0] == positions.shape[0]
    assert q.shape[1] % head_dim == 0
    assert k.shape[1] % head_dim == 0
    assert idx_q.shape[1] % head_dim == 0
    assert idx_k.shape[1] == head_dim
    assert rotary_dim <= head_dim and rotary_dim % 2 == 0

    q_heads = q.shape[1] // head_dim
    k_heads = k.shape[1] // head_dim
    idx_q_heads = idx_q.shape[1] // head_dim
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    idx_q_out = torch.empty(idx_q.shape, dtype=idx_q.dtype, device=idx_q.device)
    idx_k_out = torch.empty(idx_k.shape, dtype=idx_k.dtype, device=idx_k.device)
    block_hd = triton.next_power_of_2(head_dim)

    _sparse_qk_index_gemma_rmsnorm_rope_kernel[
        (q.shape[0], q_heads + k_heads + idx_q_heads + 1)
    ](
        q,
        k,
        idx_q,
        idx_k,
        q_out,
        k_out,
        idx_q_out,
        idx_k_out,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        idx_q.stride(0),
        idx_q.stride(1),
        idx_k.stride(0),
        idx_k.stride(1),
        q_heads,
        k_heads,
        idx_q_heads,
        head_dim,
        rotary_dim,
        eps,
        is_neox_style,
        BLOCK_HD=block_hd,
        num_warps=4,
    )
    return q_out, k_out, idx_q_out, idx_k_out


@triton.jit
def _sparse_qk_index_gemma_rmsnorm_rope_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    idx_q_ptr,
    idx_k_ptr,
    q_out_ptr,
    k_out_ptr,
    idx_q_out_ptr,
    idx_k_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    idx_k_cache_ptr,
    loc_ptr,
    q_weight_ptr,
    k_weight_ptr,
    idx_q_weight_ptr,
    idx_k_weight_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_stride_m,
    q_stride_d,
    k_stride_m,
    k_stride_d,
    v_stride_m,
    v_stride_d,
    idx_q_stride_m,
    idx_q_stride_d,
    idx_k_stride_m,
    idx_k_stride_d,
    k_cache_stride_s,
    k_cache_stride_h,
    k_cache_stride_d,
    v_cache_stride_s,
    v_cache_stride_h,
    v_cache_stride_d,
    idx_k_cache_stride_s,
    idx_k_cache_stride_d,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    idx_q_heads: tl.constexpr,
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

    main_heads: tl.constexpr = q_heads + k_heads
    idx_k_program: tl.constexpr = q_heads + k_heads + idx_q_heads

    is_q = head_program < q_heads
    is_k = (head_program >= q_heads) & (head_program < main_heads)
    is_idx_q = (head_program >= main_heads) & (head_program < idx_k_program)

    head_id = tl.where(
        is_q,
        head_program,
        tl.where(
            is_k,
            head_program - q_heads,
            tl.where(is_idx_q, head_program - main_heads, 0),
        ),
    )

    in_ptr = tl.where(
        is_q,
        q_ptr,
        tl.where(is_k, k_ptr, tl.where(is_idx_q, idx_q_ptr, idx_k_ptr)),
    )
    out_ptr = tl.where(
        is_q,
        q_out_ptr,
        tl.where(is_k, k_out_ptr, tl.where(is_idx_q, idx_q_out_ptr, idx_k_out_ptr)),
    )
    weight_ptr = tl.where(
        is_q,
        q_weight_ptr,
        tl.where(
            is_k,
            k_weight_ptr,
            tl.where(is_idx_q, idx_q_weight_ptr, idx_k_weight_ptr),
        ),
    )
    stride_m = tl.where(
        is_q,
        q_stride_m,
        tl.where(
            is_k,
            k_stride_m,
            tl.where(is_idx_q, idx_q_stride_m, idx_k_stride_m),
        ),
    )
    stride_d = tl.where(
        is_q,
        q_stride_d,
        tl.where(
            is_k,
            k_stride_d,
            tl.where(is_idx_q, idx_q_stride_d, idx_k_stride_d),
        ),
    )
    out_heads = tl.where(
        is_q,
        q_heads,
        tl.where(is_k, k_heads, tl.where(is_idx_q, idx_q_heads, 1)),
    )

    base_in = in_ptr + token_id * stride_m + head_id * head_dim * stride_d
    x = tl.load(base_in + cols * stride_d, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    rstd = tl.rsqrt(var + eps)
    normed = x * rstd * (1.0 + w)
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
    out_typed = out.to(q_out_ptr.dtype.element_ty)

    base_out = out_ptr + token_id * out_heads * head_dim + head_id * head_dim
    tl.store(base_out + cols, out_typed, mask=mask)

    loc = tl.load(loc_ptr + token_id)
    cache_k_base = (
        k_cache_ptr
        + loc * k_cache_stride_s
        + head_id * k_cache_stride_h
        + cols * k_cache_stride_d
    )
    tl.store(cache_k_base, out_typed, mask=mask & is_k)

    v_base = v_ptr + token_id * v_stride_m + head_id * head_dim * v_stride_d
    v_val = tl.load(v_base + cols * v_stride_d, mask=mask & is_k, other=0.0)
    cache_v_base = (
        v_cache_ptr
        + loc * v_cache_stride_s
        + head_id * v_cache_stride_h
        + cols * v_cache_stride_d
    )
    tl.store(cache_v_base, v_val, mask=mask & is_k)

    is_idx_k = head_program == idx_k_program
    idx_cache_base = (
        idx_k_cache_ptr + loc * idx_k_cache_stride_s + cols * idx_k_cache_stride_d
    )
    tl.store(idx_cache_base, out_typed, mask=mask & is_idx_k)


def sparse_qk_index_gemma_rmsnorm_rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    idx_q_weight: torch.Tensor,
    idx_k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse sparse Q/K/index norm+RoPE with main KV and index-K cache stores."""
    assert q.dim() == k.dim() == v.dim() == idx_q.dim() == idx_k.dim() == 2
    assert k_cache.dim() == v_cache.dim() == idx_k_cache.dim() == 3
    assert out_cache_loc.dim() == positions.dim() == 1
    assert q.shape[0] == k.shape[0] == v.shape[0] == idx_q.shape[0] == idx_k.shape[0]
    assert q.shape[0] == positions.shape[0] == out_cache_loc.shape[0]
    assert q.shape[1] % head_dim == 0
    assert k.shape[1] % head_dim == 0
    assert v.shape[1] == k.shape[1]
    assert idx_q.shape[1] % head_dim == 0
    assert idx_k.shape[1] == head_dim
    assert rotary_dim <= head_dim and rotary_dim % 2 == 0

    q_heads = q.shape[1] // head_dim
    k_heads = k.shape[1] // head_dim
    idx_q_heads = idx_q.shape[1] // head_dim
    assert k_cache.shape[1] == v_cache.shape[1] == k_heads
    assert idx_k_cache.shape[1] == 1

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    idx_q_out = torch.empty(idx_q.shape, dtype=idx_q.dtype, device=idx_q.device)
    idx_k_out = torch.empty(idx_k.shape, dtype=idx_k.dtype, device=idx_k.device)
    block_hd = triton.next_power_of_2(head_dim)

    _sparse_qk_index_gemma_rmsnorm_rope_cache_kernel[
        (q.shape[0], q_heads + k_heads + idx_q_heads + 1)
    ](
        q,
        k,
        v,
        idx_q,
        idx_k,
        q_out,
        k_out,
        idx_q_out,
        idx_k_out,
        k_cache,
        v_cache,
        idx_k_cache,
        out_cache_loc,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        idx_q.stride(0),
        idx_q.stride(1),
        idx_k.stride(0),
        idx_k.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        idx_k_cache.stride(0),
        idx_k_cache.stride(2),
        q_heads,
        k_heads,
        idx_q_heads,
        head_dim,
        rotary_dim,
        eps,
        is_neox_style,
        BLOCK_HD=block_hd,
        num_warps=4,
    )
    return q_out, k_out, idx_q_out, idx_k_out
