# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Prefix tiles of extend_attention's _fwd_kernel, scored in parallel.

The online softmax's running max after tile j is the max over tiles <= j, so each
tile's exp, row sum and P.V depend only on its own scores and that max. These kernels
repeat the prefix loop's operations per tile; _fwd_kernel replays the rescaled
accumulation in tile order, so every output element sees the same operations.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import _extract_kv_strides


@triton.jit
def _prefix_tile_setup(
    Q_Extend,
    qo_indptr,
    kv_indptr,
    stride_qbs,
    stride_qh,
    Lq: tl.constexpr,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_PREFIX: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2) * BLOCK_N_PREFIX
    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < cur_seq_len_extend
    mask_d = offs_d < Lq
    offs_q = (
        (cur_seq_extend_start_idx + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
    offs_qpe = (
        (cur_seq_extend_start_idx + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_dpe[None, :]
    )
    qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    offs_n_prefix = tl.arange(0, BLOCK_N_PREFIX)
    mask_n = (start_n + offs_n_prefix) < cur_seq_len_prefix
    final_mask = mask_m[:, None] & mask_n[None, :]
    skip = tl.full((), 0, tl.int32)
    if SLIDING_WINDOW_SIZE > 0:
        window_mask = (cur_seq_len_prefix + offs_m[:, None]) <= (
            start_n + offs_n_prefix[None, :] + SLIDING_WINDOW_SIZE
        )
        final_mask &= window_mask
        skip = (tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0).to(
            tl.int32
        )
    return (
        cur_seq,
        cur_head,
        start_n,
        cur_seq_kv_start_idx,
        cur_seq_len_prefix,
        q,
        qpe,
        offs_d,
        offs_dpe,
        offs_m,
        offs_n_prefix,
        mask_d,
        mask_n,
        final_mask,
        skip,
    )


@triton.jit
def _prefix_tile_scores(
    K_Buffer,
    kv_indices,
    q,
    qpe,
    start_n,
    cur_seq_kv_start_idx,
    cur_kv_head,
    final_mask,
    offs_d,
    offs_dpe,
    offs_n_prefix,
    mask_d,
    mask_n,
    sm_scale,
    k_scale,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_kpage,
    stride_buf_ktok,
    PAGE_SIZE: tl.constexpr,
    IDENTITY_KV_INDICES: tl.constexpr,
):
    if IDENTITY_KV_INDICES:
        offs_kv_loc = cur_seq_kv_start_idx + start_n + offs_n_prefix
    else:
        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start_idx + start_n + offs_n_prefix,
            mask=mask_n,
            other=0,
        )
    if PAGE_SIZE == 1:
        offs_buf_k = (
            offs_kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        offs_kpe = (
            offs_kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_dpe[:, None]
        )
    else:
        page_id = offs_kv_loc // PAGE_SIZE
        tok_in_p = offs_kv_loc % PAGE_SIZE
        offs_buf_k = (
            page_id[None, :] * stride_buf_kpage
            + tok_in_p[None, :] * stride_buf_ktok
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        offs_kpe = (
            page_id[None, :] * stride_buf_kpage
            + tok_in_p[None, :] * stride_buf_ktok
            + cur_kv_head * stride_buf_kh
            + offs_dpe[:, None]
        )
    k = tl.load(
        K_Buffer + offs_buf_k,
        mask=(mask_n[None, :]) & (mask_d[:, None]),
        other=0.0,
    )
    qk = tl.dot(q.to(k.dtype), k)
    kpe = tl.load(
        K_Buffer + offs_kpe,
        mask=mask_n[None, :],
        other=0.0,
    )
    qk += tl.dot(qpe.to(kpe.dtype), kpe)
    qk *= sm_scale * k_scale
    qk = tl.where(final_mask, qk, float("-inf"))
    return qk, offs_kv_loc


@triton.jit
def _prefix_tile_max_kernel(
    Q_Extend,
    K_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    TileSkip,
    TileRowMax,
    sm_scale,
    k_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_kpage,
    stride_buf_ktok,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    Lq: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_PREFIX: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    IDENTITY_KV_INDICES: tl.constexpr,
):
    (
        cur_seq,
        cur_head,
        start_n,
        cur_seq_kv_start_idx,
        cur_seq_len_prefix,
        q,
        qpe,
        offs_d,
        offs_dpe,
        offs_m,
        offs_n_prefix,
        mask_d,
        mask_n,
        final_mask,
        skip,
    ) = _prefix_tile_setup(
        Q_Extend=Q_Extend,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        stride_qbs=stride_qbs,
        stride_qh=stride_qh,
        Lq=Lq,
        SLIDING_WINDOW_SIZE=SLIDING_WINDOW_SIZE,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_M=BLOCK_M,
        BLOCK_N_PREFIX=BLOCK_N_PREFIX,
    )
    if start_n >= cur_seq_len_prefix:
        return
    tile = (cur_seq * tl.num_programs(1) + cur_head) * tl.num_programs(2) + (
        start_n // BLOCK_N_PREFIX
    )
    tl.store(TileSkip + tile, skip)
    if skip == 0:
        qk, _ = _prefix_tile_scores(
            K_Buffer=K_Buffer,
            kv_indices=kv_indices,
            q=q,
            qpe=qpe,
            start_n=start_n,
            cur_seq_kv_start_idx=cur_seq_kv_start_idx,
            cur_kv_head=cur_head // kv_group_num,
            final_mask=final_mask,
            offs_d=offs_d,
            offs_dpe=offs_dpe,
            offs_n_prefix=offs_n_prefix,
            mask_d=mask_d,
            mask_n=mask_n,
            sm_scale=sm_scale,
            k_scale=k_scale,
            stride_buf_kbs=stride_buf_kbs,
            stride_buf_kh=stride_buf_kh,
            stride_buf_kpage=stride_buf_kpage,
            stride_buf_ktok=stride_buf_ktok,
            PAGE_SIZE=PAGE_SIZE,
            IDENTITY_KV_INDICES=IDENTITY_KV_INDICES,
        )
        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        tl.store(TileRowMax + tile * BLOCK_M + offs_m, row_max_fixed)


@triton.jit
def _prefix_tile_pv_kernel(
    Q_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    TileSkip,
    TileRowMax,
    TileMax,
    TileSum,
    TilePV,
    sm_scale,
    k_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N_PREFIX: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    IDENTITY_KV_INDICES: tl.constexpr,
    PRE_TILES: tl.constexpr,
    PRE_TILES_POW2: tl.constexpr,
):
    (
        cur_seq,
        cur_head,
        start_n,
        cur_seq_kv_start_idx,
        cur_seq_len_prefix,
        q,
        qpe,
        offs_d,
        offs_dpe,
        offs_m,
        offs_n_prefix,
        mask_d,
        mask_n,
        final_mask,
        skip,
    ) = _prefix_tile_setup(
        Q_Extend=Q_Extend,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        stride_qbs=stride_qbs,
        stride_qh=stride_qh,
        Lq=Lq,
        SLIDING_WINDOW_SIZE=SLIDING_WINDOW_SIZE,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_M=BLOCK_M,
        BLOCK_N_PREFIX=BLOCK_N_PREFIX,
    )
    if start_n >= cur_seq_len_prefix:
        return
    if skip != 0:
        return
    j = start_n // BLOCK_N_PREFIX
    tile0 = (cur_seq * tl.num_programs(1) + cur_head) * PRE_TILES
    # The loop's running max after tile j: the max over the unskipped tiles <= j
    # (max is exact in any order, so one masked load and reduction).
    offs_t = tl.arange(0, PRE_TILES_POW2)
    tile_mask = (offs_t <= j) & (offs_t < PRE_TILES)
    tile_skip = tl.load(TileSkip + tile0 + offs_t, mask=tile_mask, other=1)
    e_max = tl.max(
        tl.load(
            TileRowMax + (tile0 + offs_t[:, None]) * BLOCK_M + offs_m[None, :],
            mask=(tile_mask & (tile_skip == 0))[:, None],
            other=float("-inf"),
        ),
        axis=0,
    )
    cur_kv_head = cur_head // kv_group_num
    qk, offs_kv_loc = _prefix_tile_scores(
        K_Buffer=K_Buffer,
        kv_indices=kv_indices,
        q=q,
        qpe=qpe,
        start_n=start_n,
        cur_seq_kv_start_idx=cur_seq_kv_start_idx,
        cur_kv_head=cur_kv_head,
        final_mask=final_mask,
        offs_d=offs_d,
        offs_dpe=offs_dpe,
        offs_n_prefix=offs_n_prefix,
        mask_d=mask_d,
        mask_n=mask_n,
        sm_scale=sm_scale,
        k_scale=k_scale,
        stride_buf_kbs=stride_buf_kbs,
        stride_buf_kh=stride_buf_kh,
        stride_buf_kpage=stride_buf_kpage,
        stride_buf_ktok=stride_buf_ktok,
        PAGE_SIZE=PAGE_SIZE,
        IDENTITY_KV_INDICES=IDENTITY_KV_INDICES,
    )
    p = tl.exp(qk - e_max[:, None])
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv
    if PAGE_SIZE == 1:
        offs_buf_v = (
            offs_kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
    else:
        page_id = offs_kv_loc // PAGE_SIZE
        tok_in_p = offs_kv_loc % PAGE_SIZE
        offs_buf_v = (
            page_id[:, None] * stride_buf_vpage
            + tok_in_p[:, None] * stride_buf_vtok
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
    v = tl.load(
        V_Buffer + offs_buf_v,
        mask=mask_n[:, None] & mask_dv[None, :],
        other=0.0,
    )
    dot = tl.dot(p.to(v.dtype), v)
    tile = tile0 + j
    tl.store(TileMax + tile * BLOCK_M + offs_m, e_max)
    tl.store(TileSum + tile * BLOCK_M + offs_m, tl.sum(p, 1))
    tl.store(
        TilePV + (tile * BLOCK_M + offs_m[:, None]) * Lv + offs_dv[None, :],
        dot,
        mask=mask_dv[None, :],
    )


def launch_prefix_tiles(
    *,
    prefix_tiles,
    q_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    k_scale,
    kv_group_num,
    lv,
    page_size,
    identity_kv_indices,
    sliding_window_size,
    block_dmodel,
    block_dpe,
    block_m,
    block_n_prefix,
    num_warps,
):
    """Runs the two parallel prefix-tile passes; returns _fwd_kernel's replay args."""
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    tiles = batch_size * head_num * prefix_tiles
    dev = q_extend.device
    tile_skip = torch.empty(tiles, dtype=torch.int32, device=dev)
    tile_row_max = torch.empty(tiles * block_m, dtype=torch.float32, device=dev)
    tile_max = torch.empty_like(tile_row_max)
    tile_sum = torch.empty_like(tile_row_max)
    # Rows of lv (v_extend's width) columns: the replay strides by _fwd_kernel's Lv.
    tile_pv = torch.empty(tiles * block_m * lv, dtype=torch.float32, device=dev)
    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )
    grid = (batch_size, head_num, prefix_tiles)
    shared = dict(
        Q_Extend=q_extend,
        K_Buffer=k_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        TileSkip=tile_skip,
        TileRowMax=tile_row_max,
        sm_scale=sm_scale,
        k_scale=k_scale,
        kv_group_num=kv_group_num,
        stride_qbs=q_extend.stride(0),
        stride_qh=q_extend.stride(1),
        stride_buf_kbs=k_slot_stride,
        stride_buf_kh=k_head_stride,
        stride_buf_kpage=k_page_stride,
        stride_buf_ktok=k_tok_stride,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        Lq=q_extend.shape[-1],
        BLOCK_DMODEL=block_dmodel,
        BLOCK_DPE=block_dpe,
        BLOCK_M=block_m,
        BLOCK_N_PREFIX=block_n_prefix,
        PAGE_SIZE=page_size,
        IDENTITY_KV_INDICES=identity_kv_indices,
        num_warps=num_warps,
        num_stages=1,
    )
    _prefix_tile_max_kernel[grid](**shared)
    _prefix_tile_pv_kernel[grid](
        V_Buffer=v_buffer,
        TileMax=tile_max,
        TileSum=tile_sum,
        TilePV=tile_pv,
        stride_buf_vbs=v_slot_stride,
        stride_buf_vh=v_head_stride,
        stride_buf_vpage=v_page_stride,
        stride_buf_vtok=v_tok_stride,
        Lv=lv,
        BLOCK_DV=triton.next_power_of_2(lv),
        PRE_TILES=prefix_tiles,
        PRE_TILES_POW2=triton.next_power_of_2(prefix_tiles),
        **shared,
    )
    return dict(
        PRE_TILES=prefix_tiles,
        TileSkip=tile_skip,
        TileMax=tile_max,
        TileSum=tile_sum,
        TilePV=tile_pv,
        tile_heads=head_num,
    )
