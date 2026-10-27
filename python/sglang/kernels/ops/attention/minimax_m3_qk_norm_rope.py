# SPDX-License-Identifier: Apache-2.0
"""Fused MiniMax-M3 per-head Gemma Q/K RMSNorm + partial RoPE for ROCm."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import (
    Nvfp4Cache,
    nvfp4_f32_to_e2m1,
)
from sglang.srt.utils import is_gfx95_supported


@triton.jit
def _scale_cast_to_cache(x, inv_scale, cache_ptr, SCALED: tl.constexpr):
    # Same bytes as the unfused store: torch divides a bf16/fp16 tensor by a
    # Python scalar as x * (1/scale) in fp32, rounds to the input dtype, then
    # casts to the cache dtype.
    if SCALED:
        x = (x.to(tl.float32) * inv_scale).to(x.dtype)
    return x.to(cache_ptr.dtype.element_ty)


def _cache_inv_scale(
    scale: Optional[float], x: torch.Tensor, cache: torch.Tensor
) -> Optional[float]:
    # As in MHATokenToKVPool.set_kv_buffer: a scale applies only where the store
    # casts, and None means unit (no division at all).
    if scale is None or x.dtype == cache.dtype:
        return None
    # torch's x / scale multiplies by the fp32 reciprocal of the fp32 scale.
    return float(np.float32(1.0) / np.float32(scale))


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


# The fused cache kernel holds a 128-column head row (64 rotary columns first) as four
# 32-column quarters g: [tokens, heads, 2, 2, 8] tiles of column 32 g + 16 j + 8 h + m,
# that is 16-column chunk j and its 8-column half h.
@triton.jit
def _quarter_cols(g: tl.constexpr):
    j = tl.arange(0, 2)[None, None, :, None, None]
    h = tl.arange(0, 2)[None, None, None, :, None]
    m = tl.arange(0, 8)[None, None, None, None, :]
    return 32 * g + 16 * j + 8 * h + m


@triton.jit
def _load_quarter(row_ptr, stride_d, mask, g: tl.constexpr):
    cols = _quarter_cols(g)
    h = tl.arange(0, 2)[None, None, None, :, None]
    x = tl.load(row_ptr + cols * stride_d, mask=mask, other=0.0)
    # the other 8-column half of each chunk, for the sum of squares
    x_other_half = tl.load(
        row_ptr + (cols + 8 - 16 * h) * stride_d, mask=mask, other=0.0
    )
    return x, x_other_half


@triton.jit
def _chunk_pair_sum_of_squares(
    x, x_other_half, BLOCK_T: tl.constexpr, HEADS: tl.constexpr
):
    # Same add order as tl.sum over a 128-column row on gfx9 (a chunk is a wave64 DPP
    # row: row_shr 8 fused into an fma, then 4, 2, 1), so rstd is bitwise that of the
    # per-head kernels above. Both halves h hold the [BLOCK_T, HEADS] sum.
    h = tl.arange(0, 2)[None, None, None, :, None]
    lo = tl.where(h == 0, x, x_other_half).to(tl.float32)
    hi = tl.where(h == 0, x_other_half, x).to(tl.float32)
    sums = tl.reshape(tl.fma(hi, hi, lo * lo), [BLOCK_T, HEADS, 2, 2, 2, 2, 2])
    sums = tl.sum(tl.sum(tl.sum(sums, axis=4), axis=4), axis=4)
    return tl.sum(sums, axis=2)


@triton.jit
def _gemma_norm(x, rstd, weight, norm_dtype: tl.constexpr):
    normed = x.to(tl.float32) * rstd * (1.0 + weight.to(tl.float32))
    # Match the unfused path: GemmaRMSNorm writes bf16/fp16 that RoPE then reads.
    return normed.to(norm_dtype).to(tl.float32)


@triton.jit
def _norm_rope_heads(
    x_ptr,
    stride_m,
    stride_d,
    weight_ptr,
    rows,
    row_mask,
    positions,
    cos_sin_cache_ptr,
    eps: tl.constexpr,
    norm_dtype: tl.constexpr,
    HEADS: tl.constexpr,
    HEADS_P2: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    heads = tl.arange(0, HEADS_P2)[None, :, None, None, None]
    mask = row_mask[:, None, None, None, None] & (heads < HEADS)
    row_ptr = (
        x_ptr + rows[:, None, None, None, None] * stride_m + heads * 128 * stride_d
    )
    a, a_other = _load_quarter(row_ptr, stride_d, mask, 0)
    b, b_other = _load_quarter(row_ptr, stride_d, mask, 1)
    c, c_other = _load_quarter(row_ptr, stride_d, mask, 2)
    d, d_other = _load_quarter(row_ptr, stride_d, mask, 3)
    w_a = tl.load(weight_ptr + _quarter_cols(0))
    w_b = tl.load(weight_ptr + _quarter_cols(1))
    w_c = tl.load(weight_ptr + _quarter_cols(2))
    w_d = tl.load(weight_ptr + _quarter_cols(3))
    # Full tile shape, so the RoPE math stays in the layout of the head tiles.
    cs_row = cos_sin_cache_ptr + positions[:, None, None, None, None] * 64 + heads * 0
    cos = tl.load(cs_row + _quarter_cols(0)).to(tl.float32)
    sin = tl.load(cs_row + 32 + _quarter_cols(0)).to(tl.float32)

    sum_sq = (
        _chunk_pair_sum_of_squares(a, a_other, BLOCK_T, HEADS_P2)
        + _chunk_pair_sum_of_squares(b, b_other, BLOCK_T, HEADS_P2)
    ) + (
        _chunk_pair_sum_of_squares(c, c_other, BLOCK_T, HEADS_P2)
        + _chunk_pair_sum_of_squares(d, d_other, BLOCK_T, HEADS_P2)
    )
    rstd = tl.rsqrt(tl.max(sum_sq, axis=2) / 128 + eps)[:, :, None, None, None]
    a = _gemma_norm(a, rstd, w_a, norm_dtype)
    b = _gemma_norm(b, rstd, w_b, norm_dtype)
    c = _gemma_norm(c, rstd, w_c, norm_dtype)
    d = _gemma_norm(d, rstd, w_d, norm_dtype)
    # fp32 normed and neox-rotated quarters of [BLOCK_T, HEADS_P2] heads
    a, b = a * cos - b * sin, b * cos + a * sin
    return a, b, c, d, mask


@triton.jit
def _store_quarters(row_ptr, stride_d, mask, a, b, c, d):
    tl.store(row_ptr + _quarter_cols(0) * stride_d, a, mask=mask)
    tl.store(row_ptr + _quarter_cols(1) * stride_d, b, mask=mask)
    tl.store(row_ptr + _quarter_cols(2) * stride_d, c, mask=mask)
    tl.store(row_ptr + _quarter_cols(3) * stride_d, d, mask=mask)


@triton.jit
def _nvfp4_quarter(x, BLOCK_T: tl.constexpr, HEADS: tl.constexpr):
    """One 32-column quarter [BLOCK_T, HEADS, 2, 2, 8] to NVFP4: payload
    [BLOCK_T, HEADS, 2, 2, 4] and E4M3 scales [BLOCK_T, HEADS, 2] (axis 2 is the
    16-column scale group, as `_quarter_cols` lays the columns out)."""
    amax = tl.max(tl.max(tl.abs(x), axis=4), axis=3)
    scale_e4m3 = (amax / 6.0).to(tl.float8e4nv)
    inv = tl.where(amax != 0.0, 1.0 / scale_e4m3.to(tl.float32), 0.0)
    codes = nvfp4_f32_to_e2m1(x * inv[:, :, :, None, None])
    lo, hi = tl.split(tl.reshape(codes, [BLOCK_T, HEADS, 2, 2, 4, 2]))
    return lo | (hi << 4), scale_e4m3.to(tl.uint8, bitcast=True)


@triton.jit
def _store_nvfp4_quarter(
    payload_row_ptr,
    scale_row_ptr,
    mask,
    x,
    g: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HEADS: tl.constexpr,
):
    payload, scales = _nvfp4_quarter(x, BLOCK_T, HEADS)
    j = tl.arange(0, 2)[None, None, :, None, None]
    h = tl.arange(0, 2)[None, None, None, :, None]
    pair = tl.arange(0, 4)[None, None, None, None, :]
    # a byte holds columns 2i (low nibble) and 2i+1 of the row
    tl.store(payload_row_ptr + 16 * g + 8 * j + 4 * h + pair, payload, mask=mask)
    tl.store(scale_row_ptr + 2 * g + j, scales[:, :, :, None, None], mask=mask)


@triton.jit
def _store_nvfp4_quarters(
    payload_ptr,
    scale_ptr,
    loc,
    heads,
    pay_page,
    pay_head,
    pay_row,
    sc_page,
    sc_head,
    sc_row,
    mask,
    a,
    b,
    c,
    d,
    BLOCK_T: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    """Quantize a [BLOCK_T, HEADS] tile of 128-column rows to NVFP4 and scatter it
    into the MSA-v1 paged payload/scale buffers at ``loc``."""
    page = loc // PAGE_SIZE
    row = loc % PAGE_SIZE
    payload_row = payload_ptr + page * pay_page + heads * pay_head + row * pay_row
    scale_row = scale_ptr + page * sc_page + heads * sc_head + row * sc_row
    _store_nvfp4_quarter(payload_row, scale_row, mask, a, 0, BLOCK_T, HEADS)
    _store_nvfp4_quarter(payload_row, scale_row, mask, b, 1, BLOCK_T, HEADS)
    _store_nvfp4_quarter(payload_row, scale_row, mask, c, 2, BLOCK_T, HEADS)
    _store_nvfp4_quarter(payload_row, scale_row, mask, d, 3, BLOCK_T, HEADS)


@triton.jit
def _token_block(block, num_tokens, BLOCK_T: tl.constexpr):
    rows = block * BLOCK_T + tl.arange(0, BLOCK_T)
    return rows, rows < num_tokens


@triton.jit
def _q_block(
    block,
    num_tokens,
    x_ptr,
    stride_m,
    stride_d,
    weight_ptr,
    out_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    eps: tl.constexpr,
    norm_dtype: tl.constexpr,
    HEADS: tl.constexpr,
    HEADS_P2: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    rows, row_mask = _token_block(block, num_tokens, BLOCK_T)
    positions = tl.load(positions_ptr + rows, mask=row_mask, other=0)
    a, b, c, d, mask = _norm_rope_heads(
        x_ptr,
        stride_m,
        stride_d,
        weight_ptr,
        rows,
        row_mask,
        positions,
        cos_sin_cache_ptr,
        eps,
        norm_dtype,
        HEADS,
        HEADS_P2,
        BLOCK_T,
    )
    heads = tl.arange(0, HEADS_P2)[None, :, None, None, None]
    out_row = out_ptr + rows[:, None, None, None, None] * (HEADS * 128) + heads * 128
    _store_quarters(
        out_row,
        1,
        mask,
        a.to(norm_dtype),
        b.to(norm_dtype),
        c.to(norm_dtype),
        d.to(norm_dtype),
    )


@triton.jit
def _kv_block(
    block,
    num_tokens,
    k_ptr,
    k_stride_m,
    k_stride_d,
    v_ptr,
    v_stride_m,
    v_stride_d,
    weight_ptr,
    k_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_cache_stride_s,
    k_cache_stride_h,
    k_cache_stride_d,
    v_cache_stride_s,
    v_cache_stride_h,
    v_cache_stride_d,
    k_inv_scale,
    v_inv_scale,
    loc_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    nv_pay_page,
    nv_pay_head,
    nv_pay_row,
    nv_sc_page,
    nv_sc_head,
    nv_sc_row,
    eps: tl.constexpr,
    norm_dtype: tl.constexpr,
    HEADS: tl.constexpr,
    HEADS_P2: tl.constexpr,
    BLOCK_T: tl.constexpr,
    K_SCALED: tl.constexpr,
    V_SCALED: tl.constexpr,
    NVFP4: tl.constexpr,
    NV_PAGE: tl.constexpr,
):
    rows, row_mask = _token_block(block, num_tokens, BLOCK_T)
    positions = tl.load(positions_ptr + rows, mask=row_mask, other=0)
    loc = tl.load(loc_ptr + rows, mask=row_mask, other=0)[:, None, None, None, None]
    heads = tl.arange(0, HEADS_P2)[None, :, None, None, None]
    head_mask = row_mask[:, None, None, None, None] & (heads < HEADS)
    v_row = (
        v_ptr + rows[:, None, None, None, None] * v_stride_m + heads * 128 * v_stride_d
    )
    v_a = tl.load(v_row + _quarter_cols(0) * v_stride_d, mask=head_mask, other=0.0)
    v_b = tl.load(v_row + _quarter_cols(1) * v_stride_d, mask=head_mask, other=0.0)
    v_c = tl.load(v_row + _quarter_cols(2) * v_stride_d, mask=head_mask, other=0.0)
    v_d = tl.load(v_row + _quarter_cols(3) * v_stride_d, mask=head_mask, other=0.0)
    a, b, c, d, mask = _norm_rope_heads(
        k_ptr,
        k_stride_m,
        k_stride_d,
        weight_ptr,
        rows,
        row_mask,
        positions,
        cos_sin_cache_ptr,
        eps,
        norm_dtype,
        HEADS,
        HEADS_P2,
        BLOCK_T,
    )
    a = a.to(norm_dtype)
    b = b.to(norm_dtype)
    c = c.to(norm_dtype)
    d = d.to(norm_dtype)
    k_out_row = (
        k_out_ptr + rows[:, None, None, None, None] * (HEADS * 128) + heads * 128
    )
    _store_quarters(k_out_row, 1, mask, a, b, c, d)
    # Like store_cache, the main K/V store skips the reserved cuda-graph padding slot 0
    # (padding rows may hold NaN); the K-only index scatter writes it.
    kv_mask = mask & (loc != 0)
    if NVFP4:
        # quantize the rounded K the attention reads, as the pool's own store does
        _store_nvfp4_quarters(
            k_cache_ptr,
            k_scale_cache_ptr,
            loc,
            heads,
            nv_pay_page,
            nv_pay_head,
            nv_pay_row,
            nv_sc_page,
            nv_sc_head,
            nv_sc_row,
            kv_mask,
            a.to(tl.float32),
            b.to(tl.float32),
            c.to(tl.float32),
            d.to(tl.float32),
            BLOCK_T,
            HEADS_P2,
            NV_PAGE,
        )
        _store_nvfp4_quarters(
            v_cache_ptr,
            v_scale_cache_ptr,
            loc,
            heads,
            nv_pay_page,
            nv_pay_head,
            nv_pay_row,
            nv_sc_page,
            nv_sc_head,
            nv_sc_row,
            kv_mask,
            v_a.to(tl.float32),
            v_b.to(tl.float32),
            v_c.to(tl.float32),
            v_d.to(tl.float32),
            BLOCK_T,
            HEADS_P2,
            NV_PAGE,
        )
    else:
        _store_quarters(
            k_cache_ptr + loc * k_cache_stride_s + heads * k_cache_stride_h,
            k_cache_stride_d,
            kv_mask,
            _scale_cast_to_cache(a, k_inv_scale, k_cache_ptr, K_SCALED),
            _scale_cast_to_cache(b, k_inv_scale, k_cache_ptr, K_SCALED),
            _scale_cast_to_cache(c, k_inv_scale, k_cache_ptr, K_SCALED),
            _scale_cast_to_cache(d, k_inv_scale, k_cache_ptr, K_SCALED),
        )
        _store_quarters(
            v_cache_ptr + loc * v_cache_stride_s + heads * v_cache_stride_h,
            v_cache_stride_d,
            kv_mask,
            _scale_cast_to_cache(v_a, v_inv_scale, v_cache_ptr, V_SCALED),
            _scale_cast_to_cache(v_b, v_inv_scale, v_cache_ptr, V_SCALED),
            _scale_cast_to_cache(v_c, v_inv_scale, v_cache_ptr, V_SCALED),
            _scale_cast_to_cache(v_d, v_inv_scale, v_cache_ptr, V_SCALED),
        )


@triton.jit
def _idx_k_block(
    block,
    num_tokens,
    x_ptr,
    stride_m,
    stride_d,
    weight_ptr,
    out_ptr,
    cache_ptr,
    cache_stride_s,
    cache_stride_d,
    loc_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    scale_cache_ptr,
    nv_pay_page,
    nv_pay_row,
    nv_sc_page,
    nv_sc_row,
    eps: tl.constexpr,
    norm_dtype: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NVFP4: tl.constexpr,
    NV_PAGE: tl.constexpr,
):
    rows, row_mask = _token_block(block, num_tokens, BLOCK_T)
    positions = tl.load(positions_ptr + rows, mask=row_mask, other=0)
    loc = tl.load(loc_ptr + rows, mask=row_mask, other=0)[:, None, None, None, None]
    a, b, c, d, mask = _norm_rope_heads(
        x_ptr,
        stride_m,
        stride_d,
        weight_ptr,
        rows,
        row_mask,
        positions,
        cos_sin_cache_ptr,
        eps,
        norm_dtype,
        1,
        1,
        BLOCK_T,
    )
    out_row = out_ptr + rows[:, None, None, None, None] * 128
    _store_quarters(
        out_row,
        1,
        mask,
        a.to(norm_dtype),
        b.to(norm_dtype),
        c.to(norm_dtype),
        d.to(norm_dtype),
    )
    if NVFP4:
        # quantize the rounded index K, as the pool's own store does
        _store_nvfp4_quarters(
            cache_ptr,
            scale_cache_ptr,
            loc,
            0,
            nv_pay_page,
            0,
            nv_pay_row,
            nv_sc_page,
            0,
            nv_sc_row,
            mask,
            a.to(norm_dtype).to(tl.float32),
            b.to(norm_dtype).to(tl.float32),
            c.to(norm_dtype).to(tl.float32),
            d.to(norm_dtype).to(tl.float32),
            BLOCK_T,
            1,
            NV_PAGE,
        )
    else:
        # cast fp32 straight to the cache dtype (fp8 on gfx95) to avoid rounding twice
        cache_dtype: tl.constexpr = cache_ptr.dtype.element_ty
        _store_quarters(
            cache_ptr + loc * cache_stride_s,
            cache_stride_d,
            mask,
            a.to(cache_dtype),
            b.to(cache_dtype),
            c.to(cache_dtype),
            d.to(cache_dtype),
        )


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
    k_inv_scale,
    v_inv_scale,
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
    idx_k_cache_stride_h,
    idx_k_cache_stride_d,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    idx_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    is_neox_style: tl.constexpr,
    K_SCALED: tl.constexpr,
    V_SCALED: tl.constexpr,
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
    # Like store_cache, the main K/V store skips the reserved cuda-graph padding
    # slot 0 (padding rows may hold NaN); the K-only index scatter writes it.
    store_kv = mask & is_k & (loc != 0)
    cache_k_base = (
        k_cache_ptr
        + loc * k_cache_stride_s
        + head_id * k_cache_stride_h
        + cols * k_cache_stride_d
    )
    tl.store(
        cache_k_base,
        _scale_cast_to_cache(out_typed, k_inv_scale, k_cache_ptr, K_SCALED),
        mask=store_kv,
    )

    v_base = v_ptr + token_id * v_stride_m + head_id * head_dim * v_stride_d
    v_val = tl.load(v_base + cols * v_stride_d, mask=store_kv, other=0.0)
    cache_v_base = (
        v_cache_ptr
        + loc * v_cache_stride_s
        + head_id * v_cache_stride_h
        + cols * v_cache_stride_d
    )
    tl.store(
        cache_v_base,
        _scale_cast_to_cache(v_val, v_inv_scale, v_cache_ptr, V_SCALED),
        mask=store_kv,
    )

    is_idx_k = head_program == idx_k_program
    idx_cache_base = (
        idx_k_cache_ptr + loc * idx_k_cache_stride_s + cols * idx_k_cache_stride_d
    )
    # cast fp32 straight to the cache dtype (fp8 on gfx95) to avoid rounding twice via bf16
    tl.store(
        idx_cache_base,
        out.to(idx_k_cache_ptr.dtype.element_ty),
        mask=mask & is_idx_k,
    )


@triton.jit
def _sparse_qk_index_gemma_rmsnorm_rope_cache_tiled_kernel(
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
    k_inv_scale,
    v_inv_scale,
    q_weight_ptr,
    k_weight_ptr,
    idx_q_weight_ptr,
    idx_k_weight_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    num_tokens,
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
    # NVFP4 caches only: E4M3 scale buffers and the MSA-v1 (page, head, row) strides
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    idx_k_scale_cache_ptr,
    nv_pay_page,
    nv_pay_head,
    nv_pay_row,
    nv_sc_page,
    nv_sc_head,
    nv_sc_row,
    nv_idx_pay_page,
    nv_idx_pay_row,
    nv_idx_sc_page,
    nv_idx_sc_row,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    idx_q_heads: tl.constexpr,
    Q_HEADS_P2: tl.constexpr,
    K_HEADS_P2: tl.constexpr,
    IDX_Q_HEADS_P2: tl.constexpr,
    eps: tl.constexpr,
    K_SCALED: tl.constexpr,
    V_SCALED: tl.constexpr,
    HEAD_ROWS: tl.constexpr,
    NVFP4: tl.constexpr = False,
    NV_PAGE: tl.constexpr = 1,
):
    # Per HEAD_ROWS-token block: Q_HEADS_P2 Q, K_HEADS_P2 K+V, IDX_Q_HEADS_P2 index-Q
    # and one index-K program, each over HEAD_ROWS (token, head) rows.
    programs_per_block: tl.constexpr = Q_HEADS_P2 + K_HEADS_P2 + IDX_Q_HEADS_P2 + 1
    block = tl.program_id(0) // programs_per_block
    part = tl.program_id(0) % programs_per_block
    norm_dtype: tl.constexpr = q_ptr.dtype.element_ty
    if part < Q_HEADS_P2:
        _q_block(
            block * Q_HEADS_P2 + part,
            num_tokens,
            q_ptr,
            q_stride_m,
            q_stride_d,
            q_weight_ptr,
            q_out_ptr,
            positions_ptr,
            cos_sin_cache_ptr,
            eps,
            norm_dtype,
            q_heads,
            Q_HEADS_P2,
            HEAD_ROWS // Q_HEADS_P2,
        )
    elif part < Q_HEADS_P2 + K_HEADS_P2:
        _kv_block(
            block * K_HEADS_P2 + part - Q_HEADS_P2,
            num_tokens,
            k_ptr,
            k_stride_m,
            k_stride_d,
            v_ptr,
            v_stride_m,
            v_stride_d,
            k_weight_ptr,
            k_out_ptr,
            k_cache_ptr,
            v_cache_ptr,
            k_cache_stride_s,
            k_cache_stride_h,
            k_cache_stride_d,
            v_cache_stride_s,
            v_cache_stride_h,
            v_cache_stride_d,
            k_inv_scale,
            v_inv_scale,
            loc_ptr,
            positions_ptr,
            cos_sin_cache_ptr,
            k_scale_cache_ptr,
            v_scale_cache_ptr,
            nv_pay_page,
            nv_pay_head,
            nv_pay_row,
            nv_sc_page,
            nv_sc_head,
            nv_sc_row,
            eps,
            norm_dtype,
            k_heads,
            K_HEADS_P2,
            HEAD_ROWS // K_HEADS_P2,
            K_SCALED,
            V_SCALED,
            NVFP4,
            NV_PAGE,
        )
    elif part < Q_HEADS_P2 + K_HEADS_P2 + IDX_Q_HEADS_P2:
        _q_block(
            block * IDX_Q_HEADS_P2 + part - Q_HEADS_P2 - K_HEADS_P2,
            num_tokens,
            idx_q_ptr,
            idx_q_stride_m,
            idx_q_stride_d,
            idx_q_weight_ptr,
            idx_q_out_ptr,
            positions_ptr,
            cos_sin_cache_ptr,
            eps,
            norm_dtype,
            idx_q_heads,
            IDX_Q_HEADS_P2,
            HEAD_ROWS // IDX_Q_HEADS_P2,
        )
    else:
        _idx_k_block(
            block,
            num_tokens,
            idx_k_ptr,
            idx_k_stride_m,
            idx_k_stride_d,
            idx_k_weight_ptr,
            idx_k_out_ptr,
            idx_k_cache_ptr,
            idx_k_cache_stride_s,
            idx_k_cache_stride_d,
            loc_ptr,
            positions_ptr,
            cos_sin_cache_ptr,
            idx_k_scale_cache_ptr,
            nv_idx_pay_page,
            nv_idx_pay_row,
            nv_idx_sc_page,
            nv_idx_sc_row,
            eps,
            norm_dtype,
            HEAD_ROWS,
            NVFP4,
            NV_PAGE,
        )


def sparse_qk_index_gemma_rmsnorm_rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    k_cache: Union[torch.Tensor, Nvfp4Cache],
    v_cache: Union[torch.Tensor, Nvfp4Cache],
    idx_k_cache: Union[torch.Tensor, Nvfp4Cache],
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
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse sparse Q/K/index norm+RoPE with main KV and index-K cache stores.

    The caches may be fp8; K/V are then divided by their per-tensor scales
    (``None`` is unit) before the cast, as the unfused pool store does. They may
    also be `Nvfp4Cache` (all three), quantized from the rounded K/V/index K.
    """
    nvfp4 = isinstance(k_cache, Nvfp4Cache)
    if nvfp4:
        assert isinstance(v_cache, Nvfp4Cache) and isinstance(idx_k_cache, Nvfp4Cache)
        unit_kv_scales = k_scale in (None, 1.0) and v_scale in (None, 1.0)
        assert unit_kv_scales, (
            "NVFP4 caches carry block scales, not a per-tensor K/V scale"
        )
        nv_page = k_cache.page_size
        assert idx_k_cache.page_size == nv_page
        nv_args = (
            k_cache.scales,
            v_cache.scales,
            idx_k_cache.scales,
            *k_cache.payload_strides(),
            *k_cache.scale_strides(),
            idx_k_cache.payload_strides()[0],
            idx_k_cache.payload_strides()[2],
            idx_k_cache.scale_strides()[0],
            idx_k_cache.scale_strides()[2],
        )
        cache_heads = k_cache.num_heads
        k_cache, v_cache, idx_k_cache = (
            k_cache.payload,
            v_cache.payload,
            idx_k_cache.payload,
        )
        cache_strides = (0,) * 8
    else:
        nv_page = 1
        nv_args = (k_cache, v_cache, idx_k_cache) + (0,) * 10
        assert k_cache.dim() == v_cache.dim() == idx_k_cache.dim() == 3
        assert idx_k_cache.shape[1] == 1
        cache_heads = k_cache.shape[1]
        assert v_cache.shape[1] == cache_heads
        cache_strides = (
            *k_cache.stride(),
            *v_cache.stride(),
            idx_k_cache.stride(0),
            idx_k_cache.stride(2),
        )
    assert q.dim() == k.dim() == v.dim() == idx_q.dim() == idx_k.dim() == 2
    assert out_cache_loc.dim() == positions.dim() == 1
    assert q.shape[0] == k.shape[0] == v.shape[0] == idx_q.shape[0] == idx_k.shape[0]
    assert q.shape[0] == positions.shape[0] == out_cache_loc.shape[0]
    assert q.shape[1] % head_dim == 0
    assert k.shape[1] % head_dim == 0
    assert v.shape[1] == k.shape[1]
    assert idx_q.shape[1] % head_dim == 0
    assert idx_k.shape[1] == head_dim

    num_tokens = q.shape[0]
    q_heads = q.shape[1] // head_dim
    k_heads = k.shape[1] // head_dim
    idx_q_heads = idx_q.shape[1] // head_dim
    assert cache_heads == k_heads

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    idx_q_out = torch.empty(idx_q.shape, dtype=idx_q.dtype, device=idx_q.device)
    idx_k_out = torch.empty(idx_k.shape, dtype=idx_k.dtype, device=idx_k.device)
    k_inv_scale = None if nvfp4 else _cache_inv_scale(k_scale, k, k_cache)
    v_inv_scale = None if nvfp4 else _cache_inv_scale(v_scale, v, v_cache)
    # The tiled kernel is laid out for the MiniMax-M3 head (128 columns, 64 rotary)
    # and measured on gfx950; elsewhere the per-head kernel gives the same bits.
    if not (
        is_gfx95_supported() and head_dim == 128 and rotary_dim == 64 and is_neox_style
    ):
        assert not nvfp4, "the NVFP4 cache store needs the gfx950 tiled kernel"
        assert rotary_dim <= head_dim and rotary_dim % 2 == 0
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
            1.0 if k_inv_scale is None else k_inv_scale,
            1.0 if v_inv_scale is None else v_inv_scale,
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
            idx_k_cache.stride(1),
            idx_k_cache.stride(2),
            q_heads,
            k_heads,
            idx_q_heads,
            head_dim,
            rotary_dim,
            eps,
            is_neox_style,
            K_SCALED=k_inv_scale is not None,
            V_SCALED=v_inv_scale is not None,
            BLOCK_HD=block_hd,
            num_warps=4,
        )
        return q_out, k_out, idx_q_out, idx_k_out
    q_heads_p2 = triton.next_power_of_2(q_heads)
    k_heads_p2 = triton.next_power_of_2(k_heads)
    idx_q_heads_p2 = triton.next_power_of_2(idx_q_heads)
    # Measured on MI350X: 16 head rows per program, on one wave above 256 tokens; a
    # smaller batch leaves CUs idle, so four waves per program cut its latency instead.
    head_rows = max(16, q_heads_p2, k_heads_p2, idx_q_heads_p2)
    num_warps = max(head_rows // 16, 4 if num_tokens <= 256 else 1)
    programs_per_block = q_heads_p2 + k_heads_p2 + idx_q_heads_p2 + 1

    _sparse_qk_index_gemma_rmsnorm_rope_cache_tiled_kernel[
        (triton.cdiv(num_tokens, head_rows) * programs_per_block,)
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
        1.0 if k_inv_scale is None else k_inv_scale,
        1.0 if v_inv_scale is None else v_inv_scale,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        num_tokens,
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
        *cache_strides,
        *nv_args,
        q_heads,
        k_heads,
        idx_q_heads,
        q_heads_p2,
        k_heads_p2,
        idx_q_heads_p2,
        eps,
        K_SCALED=k_inv_scale is not None,
        V_SCALED=v_inv_scale is not None,
        HEAD_ROWS=head_rows,
        NVFP4=nvfp4,
        NV_PAGE=nv_page,
        num_warps=num_warps,
        # Contraction would change the roundings; the fma calls above are explicit.
        enable_fp_fusion=False,
    )
    return q_out, k_out, idx_q_out, idx_k_out
