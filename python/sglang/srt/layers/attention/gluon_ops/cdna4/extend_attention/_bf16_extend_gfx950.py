# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Basic (non-persistent) Gluon extend-attention kernel for gfx950 (BF16 path).

Symmetric heads only (D64, D128, D256). The following constexprs are derived
internally rather than accepted as parameters:

- BLOCK_DPE / ACTUAL_BLOCK_DPE: always 0 (no DPE for symmetric heads).
- BLOCK_DV / ACTUAL_BLOCK_DV: equal to BLOCK_DMODEL / ACTUAL_BLOCK_DMODEL.
- ENABLE_MASK_SPLIT / SKIP_PREFIX_CUSTOM_MASK: always True.
- ASYNC_PAD_K / ASYNC_PAD_V: derived from BLOCK_DMODEL at compile time.

This kernel handles the basic 3D grid dispatch (seq, head, m_tile). Persistent
CTA scheduling and split-K are in ``_bf16_extend_persistent_gfx950.py``.
"""

from ._common import *  # noqa: F403
from ._layouts import (
    make_mfma_dot_layouts, make_blocked_and_slice_layouts,
    SERIAL_KT_SMEM, SERIAL_V_SMEM, SERIAL_Q_SMEM, make_serial_kt_blocked,
    make_padded_smem, make_dll, make_offset_bases,
    make_kt_offset_bases, make_kt_dll, make_v_offset_bases, make_v_dll,
)
from ._prefix_direct import attn_fwd_inner_prefix_pipelined_rfidx

# ===-----------------------------------------------------------------------===#
# Basic Kernel (non-persistent, 3D grid)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_fwd(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,  #
    K_Buffer,
    V_Buffer,  #
    qo_indptr,
    kv_indptr,
    kv_indices,  #
    Mask,
    MaskIndptr,
    WindowKvOffsets,  #
    sm_scale,
    kv_group_num,  #
    stride_qbs,
    stride_qh,  #
    stride_kbs,
    stride_kh,  #
    stride_vbs,
    stride_vh,  #
    stride_obs,
    stride_oh,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    IS_CAUSAL: gl.constexpr,  #
    USE_CUSTOM_MASK: gl.constexpr,
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    Sinks,
    HAS_SINK: gl.constexpr,  #
    LOGIT_CAP: gl.constexpr,  #
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    v_scale,  #
):
    # Experimental flags that are always False in production after tuning.
    # Kept as constexprs here so dead-code elimination removes the unused
    # branches without bloating the kernel signature passed on every launch.
    V_PRELOAD: gl.constexpr = False
    USE_RFIDX_PREFIX: gl.constexpr = False
    UNIFY_CAUSAL_PATH: gl.constexpr = False

    num_warps: gl.constexpr = gl.num_warps()

    # layouts
    threads_per_warp: gl.constexpr = 64
    _mfma: gl.constexpr = make_mfma_dot_layouts(num_warps, 16, 16, 32, 8, 4)
    mma_layout: gl.constexpr = _mfma[0]
    q_dot_layout: gl.constexpr = _mfma[1]
    kt_dot_layout: gl.constexpr = _mfma[2]
    p_dot_layout: gl.constexpr = _mfma[3]
    v_dot_layout: gl.constexpr = _mfma[4]
    _blk: gl.constexpr = make_blocked_and_slice_layouts(num_warps, mma_layout)
    blocked_layout: gl.constexpr = _blk[0]
    offs_m_layout: gl.constexpr = _blk[1]
    offs_d_layout: gl.constexpr = _blk[2]
    mma_offs_n_col: gl.constexpr = _blk[3]
    mma_offs_m_row: gl.constexpr = _blk[4]
    mma_m_layout: gl.constexpr = _blk[5]

    BLOCK_DPE: gl.constexpr = 0
    ACTUAL_BLOCK_DPE: gl.constexpr = 0
    BLOCK_DV: gl.constexpr = BLOCK_DMODEL
    ACTUAL_BLOCK_DV: gl.constexpr = ACTUAL_BLOCK_DMODEL
    ENABLE_MASK_SPLIT: gl.constexpr = ACTUAL_BLOCK_DMODEL < 256
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr = True
    ASYNC_PAD_K: gl.constexpr = 16
    ASYNC_PAD_V: gl.constexpr = 16

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=offs_d_layout)

    USE_SERIAL: gl.constexpr = num_warps < 8
    qk_scale = sm_scale * LOG2E

    # Basic (non-persistent): one tile per CTA, 3D grid dispatch
    cur_seq = gl.program_id(0)
    cur_head = gl.program_id(1)
    cur_block_m = gl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_q_start_idx = gl.load(qo_indptr + cur_seq)
    seq_len_extend = (gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx).to(tl.int32)
    is_valid_tile = cur_block_m * BLOCK_M < seq_len_extend
    seq_len_extend = tl.where(is_valid_tile, seq_len_extend, 0)
    cur_seq_kv_start_idx = gl.load(kv_indptr + cur_seq)
    seq_len_prefix_raw = (gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx).to(tl.int32)
    seq_len_prefix = tl.where(is_valid_tile, seq_len_prefix_raw, 0)

    if USE_CUSTOM_MASK:
        mask_base_idx = gl.load(MaskIndptr + cur_seq).to(tl.int64)
        window_kv_offset = 0
        if SLIDING_WINDOW_SIZE > 0:
            window_kv_offset = gl.load(WindowKvOffsets + cur_seq)
        cur_seq_len = seq_len_prefix + seq_len_extend
        mask_row_stride = (cur_seq_len + window_kv_offset).to(tl.int64)
        mask_base_idx = mask_base_idx + window_kv_offset.to(tl.int64)
        mask_kv_col_offset = (seq_len_prefix).to(tl.int64)
    else:
        mask_base_idx = tl.cast(0, tl.int64)
        mask_row_stride = tl.cast(0, tl.int64)
        mask_kv_col_offset = tl.cast(0, tl.int64)

    # Q load
    q_ptrs = (
        Q_Extend
        + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = gl.load(q_ptrs, mask=q_mask, other=0.0)
    qpe = q
    qpe_dot = gl.convert_layout(qpe, q_dot_layout)

    # softmax state
    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

    q_abs_pos = (
        seq_len_prefix
        + cur_block_m * BLOCK_M
        + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    )
    q_extend_raw = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
    if USE_CUSTOM_MASK:
        q_extend_offs = tl.minimum(q_extend_raw, tl.maximum(seq_len_extend - 1, 0))
    else:
        q_extend_offs = q_extend_raw

    if XAI_TEMPERATURE_LEN > 0:
        inv_log2_len = 1.0 / tl.log2(float(XAI_TEMPERATURE_LEN))
        xai_temperature_reg = gl.where(
            q_abs_pos > XAI_TEMPERATURE_LEN,
            tl.log2(q_abs_pos.to(gl.float32)) * inv_log2_len,
            gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row),
        )
    else:
        xai_temperature_reg = gl.full(
            [BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row
        )

    # SWA prefix skip: jump past prefix tiles entirely outside the window.
    # For the M-tile, min q_abs_pos = seq_len_prefix + cur_block_m * BLOCK_M.
    # Any prefix block whose last key position < (min_q - SWS) is fully masked.
    pfx_kv_start = cur_seq_kv_start_idx
    pfx_seq_len = seq_len_prefix
    pfx_q_abs_pos = q_abs_pos
    pfx_mask_base = mask_base_idx
    if SLIDING_WINDOW_SIZE > 0:
        q_min_abs = seq_len_prefix + cur_block_m * BLOCK_M
        first_useful_pos = tl.maximum(q_min_abs - SLIDING_WINDOW_SIZE, 0)
        prefix_skip_n = (first_useful_pos // BLOCK_N) * BLOCK_N
        pfx_kv_start = cur_seq_kv_start_idx + prefix_skip_n
        pfx_seq_len = seq_len_prefix - prefix_skip_n
        pfx_q_abs_pos = q_abs_pos - prefix_skip_n
        if USE_CUSTOM_MASK:
            pfx_mask_base = mask_base_idx + prefix_skip_n.to(tl.int64)


    if USE_SERIAL:
        if NUM_STAGES >= 2 and BLOCK_DMODEL >= 128:
            kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
            kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
            v_offset_bases: gl.constexpr = make_v_offset_bases(BLOCK_DV, BLOCK_N)
            v_async_layout: gl.constexpr = make_v_dll(num_warps, BLOCK_DV, BLOCK_N)

            kt_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], kt_offset_bases, [[512, ASYNC_PAD_K]])
            v_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], v_offset_bases, [[512, ASYNC_PAD_V]])

            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=v_smem_layout,
            )

            kpe_smem = kt_smem
            kpe_async_layout: gl.constexpr = kt_async_layout

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix -- split into unmasked bulk + masked tail.
            if pfx_seq_len > 0:
                pfx_full_len = (pfx_seq_len // BLOCK_N) * BLOCK_N
                n_full_prefix = pfx_seq_len // BLOCK_N
                n_extend_est = (seq_len_extend + BLOCK_N - 1) // BLOCK_N
                use_pipe_prefix = n_full_prefix >= NUM_STAGES
                if LOGIT_CAP > 0:
                    use_pipe_prefix = use_pipe_prefix and (n_extend_est < NUM_STAGES)
                if use_pipe_prefix and USE_RFIDX_PREFIX:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined_rfidx(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_full_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        False,
                    )
                elif use_pipe_prefix:
                    acc, l_i, m_i = attn_fwd_inner_prefix_dma_simple(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_full_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        False,
                    )
                else:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        V_PRELOAD=V_PRELOAD,
                    )
                if use_pipe_prefix and pfx_seq_len > pfx_full_len:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        V_PRELOAD=V_PRELOAD,
                        block_start=n_full_prefix,
                    )

            cdna4_async.wait_group(0)

            # EXTEND: per-CTA dispatch (v2 change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK:
                n_full_blocks = 0
            elif SLIDING_WINDOW_SIZE > 0 and effective_end > SLIDING_WINDOW_SIZE:
                n_full_blocks = 0
            elif UNIFY_CAUSAL_PATH and IS_CAUSAL:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            # SWA fast-skip: any extend block whose last key-pos < (min_q - SWS)
            # is fully outside the causal+SWA intersection, so we skip the load
            # and compute entirely. `masked_start` (the starting block for the
            # per-block-masked loop) is clamped below by this value, which is 0
            # when SWA is inactive (no change to existing behaviour).
            if SLIDING_WINDOW_SIZE > 0 and IS_CAUSAL:
                swa_first_useful = tl.maximum(
                    cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0
                )
                swa_skip_n_blocks = swa_first_useful // BLOCK_N
            else:
                swa_skip_n_blocks = 0

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_dma(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    False,
                    SKIP_BOUNDS_CHECK=True,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )
            masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_dma(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    False,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )

        else:
            # 4-warp serial path (unchanged -- already correct for any n_extend_blocks)
            kt_blocked_layout: gl.constexpr = make_serial_kt_blocked(num_warps)
            kt_serial_smem_layout: gl.constexpr = SERIAL_KT_SMEM
            v_serial_smem_layout: gl.constexpr = SERIAL_V_SMEM
            q_smem_layout: gl.constexpr = SERIAL_Q_SMEM

            kt_serial_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_DMODEL, BLOCK_N],
                layout=kt_serial_smem_layout,
            )
            kt_dpe_serial_smem = kt_serial_smem
            v_serial_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_N, BLOCK_DV],
                layout=v_serial_smem_layout,
            )
            q_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_M, BLOCK_DMODEL],
                layout=q_smem_layout,
            )

            q_smem.store(q)
            q_dot = q_smem.load(q_dot_layout)
            qpe_dot = gl.convert_layout(qpe, q_dot_layout)

            if pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_serial(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_serial_smem,
                    kt_dpe_serial_smem,
                    v_serial_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_blocked_layout,
                    blocked_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    V_PRELOAD,
                )

            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK:
                n_full_blocks = 0
            elif SLIDING_WINDOW_SIZE > 0 and effective_end > SLIDING_WINDOW_SIZE:
                n_full_blocks = 0
            elif UNIFY_CAUSAL_PATH and IS_CAUSAL:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            # SWA fast-skip: any extend block whose last key-pos < (min_q - SWS)
            # is fully outside the causal+SWA intersection, so we skip the load
            # and compute entirely. `masked_start` (the starting block for the
            # per-block-masked loop) is clamped below by this value, which is 0
            # when SWA is inactive (no change to existing behaviour).
            if SLIDING_WINDOW_SIZE > 0 and IS_CAUSAL:
                swa_first_useful = tl.maximum(
                    cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0
                )
                swa_skip_n_blocks = swa_first_useful // BLOCK_N
            else:
                swa_skip_n_blocks = 0

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_serial(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_serial_smem,
                    kt_dpe_serial_smem,
                    v_serial_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_blocked_layout,
                    blocked_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD,
                )
            masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
            if n_extend_blocks > masked_start:
                acc, l_i, m_i = attn_fwd_inner_extend_serial(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_serial_smem,
                    kt_dpe_serial_smem,
                    v_serial_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_blocked_layout,
                    blocked_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD,
                )

    else:
        # 8-warp DMA path

        if BLOCK_DMODEL >= 128:
            kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
            kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
            v_offset_bases: gl.constexpr = make_v_offset_bases(BLOCK_DV, BLOCK_N)
            v_async_layout: gl.constexpr = make_v_dll(num_warps, BLOCK_DV, BLOCK_N)

            kt_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], kt_offset_bases, [[512, ASYNC_PAD_K]])
            v_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], v_offset_bases, [[512, ASYNC_PAD_V]])

            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=v_async_smem_layout,
            )

            kpe_smem = kt_smem
            kpe_async_layout: gl.constexpr = kt_async_layout

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix dispatch -- split unmasked bulk + masked tail.
            pfx_full_len = (pfx_seq_len // BLOCK_N) * BLOCK_N
            n_full_prefix = pfx_seq_len // BLOCK_N
            if n_full_prefix >= NUM_STAGES:
                if NUM_STAGES >= 3:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined_scalar_mask(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_full_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        False,
                    )
                elif USE_RFIDX_PREFIX:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined_rfidx(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_full_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        False,
                    )
                else:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_full_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        kpe_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        NUM_STAGES,
                        kt_async_layout,
                        kpe_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        False,
                    )
                if pfx_seq_len > pfx_full_len:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        qpe_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        ACTUAL_BLOCK_DMODEL,
                        BLOCK_DPE,
                        ACTUAL_BLOCK_DPE,
                        BLOCK_DV,
                        ACTUAL_BLOCK_DV,
                        kt_async_layout,
                        v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        V_PRELOAD=V_PRELOAD,
                        block_start=n_full_prefix,
                    )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    V_PRELOAD=V_PRELOAD,
                )

            # EXTEND: per-CTA dispatch (v2 core change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK:
                n_full_blocks = 0
            elif SLIDING_WINDOW_SIZE > 0 and effective_end > SLIDING_WINDOW_SIZE:
                n_full_blocks = 0
            elif UNIFY_CAUSAL_PATH and IS_CAUSAL:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            # SWA fast-skip: any extend block whose last key-pos < (min_q - SWS)
            # is fully outside the causal+SWA intersection, so we skip the load
            # and compute entirely. `masked_start` (the starting block for the
            # per-block-masked loop) is clamped below by this value, which is 0
            # when SWA is inactive (no change to existing behaviour).
            if SLIDING_WINDOW_SIZE > 0 and IS_CAUSAL:
                swa_first_useful = tl.maximum(
                    cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0
                )
                swa_skip_n_blocks = swa_first_useful // BLOCK_N
            else:
                swa_skip_n_blocks = 0

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    False,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )
            masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    False,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )

        else:
            # 8-warp BLOCK_DMODEL < 128
            kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
            kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
            v_offset_bases: gl.constexpr = make_v_offset_bases(BLOCK_DV, BLOCK_N)
            v_async_layout: gl.constexpr = make_v_dll(num_warps, BLOCK_DV, BLOCK_N)

            kt_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], kt_offset_bases, [[512, ASYNC_PAD_K]])
            v_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], v_offset_bases, [[512, ASYNC_PAD_V]])

            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=v_async_smem_layout,
            )

            kpe_smem = kt_smem
            kpe_async_layout: gl.constexpr = kt_async_layout

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix -- split unmasked bulk + masked tail.
            pfx_full_len = (pfx_seq_len // BLOCK_N) * BLOCK_N
            n_full_prefix = pfx_seq_len // BLOCK_N
            if n_full_prefix >= NUM_STAGES and USE_RFIDX_PREFIX:
                acc, l_i, m_i = attn_fwd_inner_prefix_pipelined_rfidx(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_full_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    False,
                )
            elif n_full_prefix >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_full_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    False,
                )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    V_PRELOAD=V_PRELOAD,
                )
            if n_full_prefix >= NUM_STAGES and pfx_seq_len > pfx_full_len:
                acc, l_i, m_i = attn_fwd_inner_prefix_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    V_PRELOAD=V_PRELOAD,
                    block_start=n_full_prefix,
                )

            # EXTEND: per-CTA dispatch (v2 core change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK:
                n_full_blocks = 0
            elif SLIDING_WINDOW_SIZE > 0 and effective_end > SLIDING_WINDOW_SIZE:
                n_full_blocks = 0
            elif UNIFY_CAUSAL_PATH and IS_CAUSAL:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            # SWA fast-skip: any extend block whose last key-pos < (min_q - SWS)
            # is fully outside the causal+SWA intersection, so we skip the load
            # and compute entirely. `masked_start` (the starting block for the
            # per-block-masked loop) is clamped below by this value, which is 0
            # when SWA is inactive (no change to existing behaviour).
            if SLIDING_WINDOW_SIZE > 0 and IS_CAUSAL:
                swa_first_useful = tl.maximum(
                    cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0
                )
                swa_skip_n_blocks = swa_first_useful // BLOCK_N
            else:
                swa_skip_n_blocks = 0

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    False,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )
            masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_extend_pipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    kpe_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    kpe_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    False,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_short(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    qpe_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=V_PRELOAD,
                )


    # sinks
    if HAS_SINK:
        cur_sink = gl.load(Sinks + cur_head)
        l_i = l_i + gl.exp2(cur_sink * LOG2E - m_i)

    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]
    acc = acc * v_scale

    o_base = O_Extend + cur_seq_q_start_idx * stride_obs + cur_head * stride_oh
    o_offsets = ((cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs + offs_dv[None, :]).to(tl.int32)
    o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
    if ACTUAL_BLOCK_DV != BLOCK_DV:
        o_mask = o_mask & (offs_dv[None, :] < ACTUAL_BLOCK_DV)
    out = gl.convert_layout(acc, blocked_layout).to(O_Extend.dtype.element_ty)
    cdna4_buffer_store(out, o_base, o_offsets, mask=o_mask)
