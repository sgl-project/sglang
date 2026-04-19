# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 KV basic (non-persistent) extend-attention kernel for symmetric head dims (Lq == Lv).

FP8 variant: prefix phase uses native FP8 MFMA on FP8 KV cache buffers,
extend phase uses BF16. Two-phase shared memory allocation.
Handles D=64, D=128, D=256 where BLOCK_DPE is always 0.
"""

from ._common import *  # noqa: F403
from ._layouts import (
    make_mfma_dot_layouts, make_fp8_dot_layouts, make_blocked_and_slice_layouts,
    SERIAL_KT_SMEM, SERIAL_V_SMEM, SERIAL_Q_SMEM, make_serial_kt_blocked,
    make_padded_smem, make_dll, make_offset_bases,
    make_kt_offset_bases, make_kt_dll,
    make_fp8_bf16_v_offset_bases, make_fp8_bf16_v_dll,
)


# ===-----------------------------------------------------------------------===#
# FP8 KV basic extend-attention (non-persistent), symmetric head dims
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_fwd_basic(
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
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    ENABLE_MASK_SPLIT: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    EXT_BLOCK_N: gl.constexpr,  #
    EXT_NUM_STAGES: gl.constexpr,  #
    ASYNC_PAD_K: gl.constexpr,
    ASYNC_PAD_V: gl.constexpr,  #
    Sinks,
    HAS_SINK: gl.constexpr,  #
    LOGIT_CAP: gl.constexpr,  #
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    v_scale,  #
):
    num_warps: gl.constexpr = gl.num_warps()
    PFX_SMEM_TY: gl.constexpr = K_Buffer.dtype.element_ty

    # layouts
    threads_per_warp: gl.constexpr = 64
    _mfma: gl.constexpr = make_mfma_dot_layouts(num_warps, 16, 16, 32, 8, 4)
    mma_layout: gl.constexpr = _mfma[0]
    q_dot_layout: gl.constexpr = _mfma[1]
    kt_dot_layout: gl.constexpr = _mfma[2]
    p_dot_layout: gl.constexpr = _mfma[3]
    v_dot_layout: gl.constexpr = _mfma[4]
    _fp8: gl.constexpr = make_fp8_dot_layouts(mma_layout, 16, 8)
    fp8_q_dot_layout: gl.constexpr = _fp8[0]
    fp8_kt_dot_layout: gl.constexpr = _fp8[1]
    fp8_p_dot_layout: gl.constexpr = _fp8[2]
    fp8_v_dot_layout: gl.constexpr = _fp8[3]
    _blk: gl.constexpr = make_blocked_and_slice_layouts(num_warps, mma_layout)
    blocked_layout: gl.constexpr = _blk[0]
    offs_m_layout: gl.constexpr = _blk[1]
    offs_d_layout: gl.constexpr = _blk[2]
    mma_offs_n_col: gl.constexpr = _blk[3]
    mma_offs_m_row: gl.constexpr = _blk[4]
    mma_m_layout: gl.constexpr = _blk[5]

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=offs_d_layout)

    USE_SERIAL: gl.constexpr = num_warps < 8
    qk_scale = sm_scale * LOG2E

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
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + gl.arange(0, BLOCK_DPE, layout=offs_d_layout)
        qpe_ptrs = (
            Q_Extend
            + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
        if ACTUAL_BLOCK_DPE != BLOCK_DPE:
            qpe_mask = qpe_mask & (offs_dpe[None, :] < (BLOCK_DMODEL + ACTUAL_BLOCK_DPE))
        qpe = gl.load(qpe_ptrs, mask=qpe_mask, other=0.0)
    else:
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

    orig_seq_len_extend = seq_len_extend

    # Compile-time guard: the 4-warp DMA path's layout bases assume
    # BLOCK_DMODEL >= 128. Using NUM_STAGES >= 2 with D<128 produces a
    # broken `v_async_layout` (register base [64,0] exceeds tensor row
    # count of BLOCK_DMODEL) and fails LLVM translation with
    # `unrealized_conversion_cast`. Mirror the persistent kernel's
    # gate and require callers with D<128 FP8 to use NUM_STAGES==1 on
    # basic, or route to the persistent (symmetric) kernel which has a
    # dedicated D<128 DMA branch.
    tl.static_assert(
        not (num_warps == 4 and NUM_STAGES >= 2 and BLOCK_DMODEL < 128),
        "FP8 basic: NUM_STAGES>=2 requires BLOCK_DMODEL>=128 on 4-warp path. "
        "Use NUM_STAGES=1 or route to persistent kernel for D<128.",
    )
    tl.static_assert(
        not (num_warps == 8 and BLOCK_DMODEL < 128),
        "FP8 basic: 8-warp DMA path is incomplete for BLOCK_DMODEL<128. "
        "Route to persistent kernel which has a dedicated D<128 branch.",
    )

    if USE_SERIAL:
        if NUM_STAGES >= 2:
            # 4-warp DMA path
            if BLOCK_DMODEL >= 256:
                kt_offset_bases: gl.constexpr = make_offset_bases(128, [16], [1, 2, 4, 8], 0)
                kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]], [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16], ], [[0, 1], [0, 2]])
            elif BLOCK_DMODEL >= 128:
                if BLOCK_N >= 128:
                    kt_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32, 64], [1, 2, 4, 8], 0)
                    kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]], [[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]], [[0, 1], [0, 2]])
                else:
                    kt_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32], [1, 2, 4, 8], 0)
                    kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], [[16, 0], [32, 0], [64, 0], [0, 1], [0, 2], [0, 4]], [[0, 16], [0, 32]])
            else:
                if BLOCK_N >= 128:
                    kt_offset_bases: gl.constexpr = make_offset_bases(32, [16, 32, 64], [1, 2, 4, 8], 0)
                    kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 8], [0, 64]], [[8, 0], [16, 0], [32, 0], [0, 1], [0, 16], [0, 32]], [[0, 2], [0, 4]])
                else:
                    kt_offset_bases: gl.constexpr = make_offset_bases(32, [16, 32], [1, 2, 4, 8], 0)
                    kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 8]], [[8, 0], [16, 0], [32, 0], [0, 1], [0, 16], [0, 32]], [[0, 2], [0, 4]])

            if BLOCK_DV >= 256:
                if BLOCK_N >= 128:
                    v_offset_bases: gl.constexpr = make_offset_bases(128, [16, 32, 64], [1, 2, 4, 8], 1)
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [64, 0]], [[0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [32, 0]], [[1, 0], [2, 0]])
                else:
                    v_offset_bases: gl.constexpr = make_offset_bases(128, [16], [1, 2, 4, 8], 1)
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]], [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]], [[1, 0], [2, 0]])
            elif BLOCK_DV >= 128:
                if BLOCK_N >= 128:
                    v_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32, 64], [1, 2, 4, 8], 1)
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]], [[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]], [[1, 0], [2, 0]])
                else:
                    v_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32], [1, 2, 4, 8], 1)
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]], [[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]], [[16, 0], [32, 0]])
            else:
                if BLOCK_N >= 128:
                    v_offset_bases: gl.constexpr = make_offset_bases(32, [16, 32, 64], [1, 2, 4, 8], 1)
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [8, 0], [64, 0]], [[0, 8], [0, 16], [0, 32], [1, 0], [16, 0], [32, 0]], [[2, 0], [4, 0]])
                else:
                    v_offset_bases: gl.constexpr = make_offset_bases(32, [16, 32], [1, 2, 4, 8], 1)
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [8, 0]], [[0, 8], [0, 16], [0, 32], [1, 0], [16, 0], [32, 0]], [[2, 0], [4, 0]])

            bf16_kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
            if BLOCK_DMODEL >= 128:
                bf16_kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
                bf16_v_offset_bases: gl.constexpr = make_fp8_bf16_v_offset_bases(num_warps, BLOCK_DV, BLOCK_N)
                bf16_v_async_layout: gl.constexpr = make_fp8_bf16_v_dll(num_warps, BLOCK_DV, BLOCK_N)
            else:
                bf16_v_offset_bases: gl.constexpr = make_fp8_bf16_v_offset_bases(num_warps, BLOCK_DV, BLOCK_N)
                if BLOCK_N >= 128:
                    bf16_kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 8], [0, 64]], [[8, 0], [16, 0], [32, 0], [0, 1], [0, 16], [0, 32]], [[0, 2], [0, 4]])
                    bf16_v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [8, 0], [64, 0]], [[0, 8], [0, 16], [0, 32], [1, 0], [16, 0], [32, 0]], [[2, 0], [4, 0]])
                else:
                    bf16_kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]], [[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]], [[0, 2], [0, 4]])
                    bf16_v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]], [[0, 16], [0, 32], [1, 0], [2, 0], [4, 0], [16, 0]], [[32, 0], [64, 0]])

            if BLOCK_DPE == 0:
                bf16_kpe_async_layout: gl.constexpr = bf16_kt_async_layout
                bf16_kpe_offset_bases: gl.constexpr = bf16_kt_offset_bases
            elif BLOCK_DPE >= 64 and BLOCK_N >= 128:
                bf16_kpe_offset_bases: gl.constexpr = make_offset_bases(32, [8, 16, 32, 64], [1, 2, 4], 0)
                bf16_kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 4], [0, 64]], [[8, 0], [16, 0], [32, 0], [0, 8], [0, 16], [0, 32]], [[0, 1], [0, 2]])
            elif BLOCK_DPE >= 64:
                bf16_kpe_offset_bases: gl.constexpr = make_offset_bases(32, [8, 16, 32], [1, 2, 4], 0)
                bf16_kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 4]], [[8, 0], [16, 0], [32, 0], [0, 8], [0, 16], [0, 32]], [[0, 1], [0, 2]])
            elif BLOCK_DPE >= 32 and BLOCK_N >= 128:
                bf16_kpe_offset_bases: gl.constexpr = make_offset_bases(16, [8, 16, 32, 64], [1, 2, 4], 0)
                bf16_kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [2, 0], [0, 4], [0, 64]], [[4, 0], [8, 0], [16, 0], [0, 8], [0, 16], [0, 32]], [[0, 1], [0, 2]])
            else:
                bf16_kpe_offset_bases: gl.constexpr = make_offset_bases(16, [8, 16, 32], [1, 2, 4], 0)
                bf16_kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [2, 0], [0, 4]], [[4, 0], [8, 0], [16, 0], [0, 8], [0, 16], [0, 32]], [[0, 1], [0, 2]])

            # FP8 prefix smem: 1024-byte interval for 128-bit direct-to-LDS
            fp8_kt_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], kt_offset_bases, [[1024, ASYNC_PAD_K], [2048, 32]])
            fp8_v_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], v_offset_bases, [[1024, ASYNC_PAD_V], [2048, 32]])
            # BF16 extend smem: standard 512-byte interval, using BF16 offset bases
            # When EXT_BLOCK_N != BLOCK_N, define separate extend layouts
            if EXT_BLOCK_N == BLOCK_N:
                ext_kt_offset_bases: gl.constexpr = bf16_kt_offset_bases
                ext_kt_async_layout: gl.constexpr = bf16_kt_async_layout
                ext_v_offset_bases: gl.constexpr = bf16_v_offset_bases
                ext_v_async_layout: gl.constexpr = bf16_v_async_layout
                ext_kpe_offset_bases: gl.constexpr = bf16_kpe_offset_bases
                ext_kpe_async_layout: gl.constexpr = bf16_kpe_async_layout
            else:
                if BLOCK_DMODEL >= 256:
                    ext_kt_offset_bases: gl.constexpr = make_offset_bases(128, [16, 32], [1, 2, 4, 8], 0)
                    ext_kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, EXT_BLOCK_N], [[1, 0], [2, 0], [4, 0], [128, 0], [0, 4], [0, 8]], [[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]], [[0, 1], [0, 2]])
                else:
                    ext_kt_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32], [1, 2, 4, 8], 0)
                    ext_kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, EXT_BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]], [[8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32]], [[0, 1], [0, 2]])
                if BLOCK_DV >= 256:
                    ext_v_offset_bases: gl.constexpr = make_offset_bases(128, [16, 32], [1, 2, 4, 8], 1)
                    ext_v_async_layout: gl.constexpr = make_dll([EXT_BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [0, 128], [4, 0], [8, 0]], [[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]], [[1, 0], [2, 0]])
                else:
                    ext_v_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32], [1, 2, 4, 8], 1)
                    ext_v_async_layout: gl.constexpr = make_dll([EXT_BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]], [[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]], [[1, 0], [2, 0]])
                ext_kpe_offset_bases: gl.constexpr = bf16_kpe_offset_bases
                ext_kpe_async_layout: gl.constexpr = bf16_kpe_async_layout

            bf16_kt_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, EXT_BLOCK_N], ext_kt_offset_bases, [[512, ASYNC_PAD_K]])
            bf16_v_smem_layout: gl.constexpr = make_padded_smem([EXT_BLOCK_N, BLOCK_DV], ext_v_offset_bases, [[512, ASYNC_PAD_V]])

            kt_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=fp8_kt_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=fp8_v_smem_layout,
            )

            ASYNC_KPE: gl.constexpr = (
                BLOCK_DPE > 0
                and Q_Extend.dtype.element_ty != tl.float32
            )

            if ASYNC_KPE:
                if BLOCK_DPE >= 64 and BLOCK_N >= 128:
                    kpe_offset_bases: gl.constexpr = make_offset_bases(32, [4, 8, 16, 32], [1, 2, 64], 0)
                    kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [2, 0], [4, 0], [32, 0], [0, 64]], [[0, 4], [0, 8], [0, 16], [0, 32], [8, 0], [16, 0]], [[0, 1], [0, 2]])
                elif BLOCK_DPE >= 64:
                    kpe_offset_bases: gl.constexpr = make_offset_bases(32, [4, 8, 16], [1, 2, 32], 0)
                    kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 32]], [[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]], [[0, 1], [0, 2]])
                elif BLOCK_DPE >= 32:
                    kpe_offset_bases: gl.constexpr = make_offset_bases(16, [16, 32], [1, 2, 4, 8], 0)
                    kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [2, 0], [0, 4]], [[4, 0], [8, 0], [16, 0], [0, 8], [0, 16], [0, 32]], [[0, 1], [0, 2]])
                else:
                    kpe_offset_bases: gl.constexpr = make_offset_bases(8, [16, 32], [1, 2, 4, 8], 0)
                    kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [0, 4]], [[2, 0], [4, 0], [8, 0], [0, 8], [0, 16], [0, 32]], [[0, 1], [0, 2]])
                fp8_kpe_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DPE, BLOCK_N], kpe_offset_bases, [[1024, ASYNC_PAD_K], [2048, 32]])
                kpe_smem = gl.allocate_shared_memory(
                    PFX_SMEM_TY,
                    [NUM_STAGES, BLOCK_DPE, BLOCK_N],
                    layout=fp8_kpe_smem_layout,
                )
            else:
                kpe_smem = kt_smem
                kpe_async_layout: gl.constexpr = kt_async_layout

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=PFX_SMEM_TY,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            fp8_q_dot = gl.convert_layout(q.to(tl.float8e4nv), fp8_q_dot_layout)
            fp8_qpe_dot = gl.convert_layout(qpe.to(tl.float8e4nv), fp8_q_dot_layout)
            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix phase uses FP8 dot layouts for native FP8 MFMA
            if pfx_seq_len > 0:
                pfx_full_len = (pfx_seq_len // BLOCK_N) * BLOCK_N
                n_full_prefix = pfx_seq_len // BLOCK_N
                n_extend_est = (seq_len_extend + BLOCK_N - 1) // BLOCK_N
                use_pipe_prefix = n_full_prefix >= NUM_STAGES
                if LOGIT_CAP > 0:
                    use_pipe_prefix = use_pipe_prefix and (n_extend_est < NUM_STAGES)
                if use_pipe_prefix:
                    acc, l_i, m_i = attn_fwd_inner_prefix_dma_simple(
                        acc,
                        l_i,
                        m_i,
                        fp8_q_dot,
                        fp8_qpe_dot,
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
                        fp8_kt_dot_layout,
                        fp8_p_dot_layout,
                        fp8_v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        ASYNC_KPE,
                    )
                    if pfx_seq_len > pfx_full_len:
                        acc, l_i, m_i = attn_fwd_inner_prefix_short(
                            acc,
                            l_i,
                            m_i,
                            fp8_q_dot,
                            fp8_qpe_dot,
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
                            fp8_kt_dot_layout,
                            fp8_p_dot_layout,
                            fp8_v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                            V_PRELOAD=False,
                            block_start=n_full_prefix,
                        )
                else:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        fp8_q_dot,
                        fp8_qpe_dot,
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
                        fp8_kt_dot_layout,
                        fp8_p_dot_layout,
                        fp8_v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        V_PRELOAD=False,
                    )

            # Transition FP8 prefix smem -> BF16 extend smem
            kt_smem._keep_alive()
            v_smem._keep_alive()
            if ASYNC_KPE:
                kpe_smem._keep_alive()
            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [EXT_NUM_STAGES, BLOCK_DMODEL, EXT_BLOCK_N],
                layout=bf16_kt_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [EXT_NUM_STAGES, EXT_BLOCK_N, BLOCK_DV],
                layout=bf16_v_smem_layout,
            )
            if ASYNC_KPE:
                bf16_kpe_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DPE, EXT_BLOCK_N], ext_kpe_offset_bases, [[512, ASYNC_PAD_K]])
                kpe_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [EXT_NUM_STAGES, BLOCK_DPE, EXT_BLOCK_N],
                    layout=bf16_kpe_smem_layout,
                )
            for _s in gl.static_range(EXT_NUM_STAGES):
                v_zero = gl.zeros(
                    [EXT_BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=ext_v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            # EXTEND: per-CTA dispatch (v2 change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + EXT_BLOCK_N - 1) // EXT_BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % EXT_BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + EXT_BLOCK_N - 1) // EXT_BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

            k_extend_base = (
                K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
            )
            v_extend_base = (
                V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
            )

            if n_full_blocks >= EXT_NUM_STAGES:
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
                    EXT_BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    EXT_NUM_STAGES,
                    ext_kt_async_layout,
                    ext_kpe_async_layout,
                    ext_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
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
                    EXT_BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    ext_kt_async_layout,
                    ext_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=False,
                )
            if SLIDING_WINDOW_SIZE > 0 and IS_CAUSAL:
                swa_first_useful = tl.maximum(
                    cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0
                )
                swa_skip_n_blocks = swa_first_useful // EXT_BLOCK_N
            else:
                swa_skip_n_blocks = 0
            masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= EXT_NUM_STAGES:
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
                    EXT_BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    EXT_NUM_STAGES,
                    ext_kt_async_layout,
                    ext_kpe_async_layout,
                    ext_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
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
                    EXT_BLOCK_N,
                    BLOCK_DMODEL,
                    ACTUAL_BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    BLOCK_DV,
                    ACTUAL_BLOCK_DV,
                    ext_kt_async_layout,
                    ext_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=False,
                )

        else:
            # 4-warp serial path (unchanged -- already correct for any n_extend_blocks)
            kt_blocked_layout: gl.constexpr = make_serial_kt_blocked(num_warps)
            kt_serial_smem_layout: gl.constexpr = SERIAL_KT_SMEM
            v_serial_smem_layout: gl.constexpr = SERIAL_V_SMEM
            q_smem_layout: gl.constexpr = SERIAL_Q_SMEM

            kt_serial_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
                [BLOCK_DMODEL, BLOCK_N],
                layout=kt_serial_smem_layout,
            )
            if BLOCK_DPE > 0:
                kt_dpe_serial_smem = gl.allocate_shared_memory(
                    PFX_SMEM_TY,
                    [BLOCK_DPE, BLOCK_N],
                    layout=kt_serial_smem_layout,
                )
            else:
                kt_dpe_serial_smem = kt_serial_smem
            v_serial_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
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
            fp8_q_dot = gl.convert_layout(q.to(tl.float8e4nv), fp8_q_dot_layout)
            fp8_qpe_dot = gl.convert_layout(qpe.to(tl.float8e4nv), fp8_q_dot_layout)
            qpe_dot = gl.convert_layout(qpe, q_dot_layout)

            if pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_serial(
                    acc,
                    l_i,
                    m_i,
                    fp8_q_dot,
                    fp8_qpe_dot,
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
                    fp8_kt_dot_layout,
                    fp8_p_dot_layout,
                    fp8_v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    False,
                )

            # Transition serial smem from FP8 (prefix) to BF16 (extend),
            # mirroring the DMA path's smem transition.
            kt_serial_smem._keep_alive()
            v_serial_smem._keep_alive()
            if BLOCK_DPE > 0:
                kt_dpe_serial_smem._keep_alive()
            kt_serial_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_DMODEL, BLOCK_N],
                layout=kt_serial_smem_layout,
            )
            if BLOCK_DPE > 0:
                kt_dpe_serial_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_DPE, BLOCK_N],
                    layout=kt_serial_smem_layout,
                )
            else:
                kt_dpe_serial_smem = kt_serial_smem
            v_serial_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [BLOCK_N, BLOCK_DV],
                layout=v_serial_smem_layout,
            )

            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

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
                    False,
                )
            if SLIDING_WINDOW_SIZE > 0 and IS_CAUSAL:
                swa_first_useful = tl.maximum(
                    cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0
                )
                swa_skip_n_blocks = swa_first_useful // BLOCK_N
            else:
                swa_skip_n_blocks = 0
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
                    False,
                )

    else:
        # 8-warp DMA path

        if BLOCK_DMODEL >= 128:
            if BLOCK_DMODEL >= 256:
                kt_offset_bases: gl.constexpr = make_offset_bases(128, [16], [1, 2, 4, 8], 0)
                kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 8]], [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16], ], [[0, 1], [0, 2], [0, 4]])
                if BLOCK_DV >= 256:
                    v_offset_bases: gl.constexpr = make_offset_bases(128, [16], [1, 2, 4, 8], 1)
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [8, 0]], [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], ], [[1, 0], [2, 0], [4, 0]])
                else:
                    v_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32], [1, 2, 4, 8], 1)
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [8, 0]], [[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]], [[1, 0], [2, 0], [4, 0]])
            else:
                if BLOCK_N >= 128:
                    kt_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32, 64], [1, 2, 4, 8], 0)
                    v_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32, 64], [1, 2, 4, 8], 1)
                    kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], [[16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64]], [[0, 1], [0, 2], [0, 4]])
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]], [[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]], [[1, 0], [2, 0], [4, 0]])
                else:
                    kt_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32], [1, 2, 4, 8], 0)
                    v_offset_bases: gl.constexpr = make_offset_bases(64, [16, 32], [1, 2, 4, 8], 1)
                    kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [8, 0]], [[16, 0], [32, 0], [64, 0], [0, 1], [0, 2], [0, 4]], [[0, 8], [0, 16], [0, 32]])
                    v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [0, 8]], [[0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0]], [[8, 0], [16, 0], [32, 0]])

            # FP8 prefix smem: 1024-byte interval for 128-bit direct-to-LDS
            fp8_kt_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], kt_offset_bases, [[1024, ASYNC_PAD_K], [2048, 32]])
            fp8_v_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], v_offset_bases, [[1024, ASYNC_PAD_V], [2048, 32]])
            bf16_kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
            bf16_kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
            bf16_v_offset_bases: gl.constexpr = make_fp8_bf16_v_offset_bases(num_warps, BLOCK_DV, BLOCK_N)
            bf16_v_async_layout: gl.constexpr = make_fp8_bf16_v_dll(num_warps, BLOCK_DV, BLOCK_N)
            # BF16 extend smem: standard 512-byte interval, using BF16 offset bases
            bf16_kt_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], bf16_kt_offset_bases, [[512, ASYNC_PAD_K]])
            bf16_v_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], bf16_v_offset_bases, [[512, ASYNC_PAD_V]])
            if BLOCK_DPE == 0:
                bf16_kpe_async_layout: gl.constexpr = bf16_kt_async_layout
                bf16_kpe_offset_bases: gl.constexpr = bf16_kt_offset_bases

            kt_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=fp8_kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=fp8_v_async_smem_layout,
            )

            ASYNC_KPE: gl.constexpr = (
                BLOCK_DPE > 0
                and Q_Extend.dtype.element_ty != tl.float32
            )

            if ASYNC_KPE:
                if BLOCK_DPE >= 64:
                    kpe_offset_bases: gl.constexpr = make_offset_bases(32, [4, 8, 16], [1, 2, 32], 0)
                    kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [2, 0], [4, 0]], [[8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16]], [[0, 1], [0, 2], [0, 32]])
                elif BLOCK_DPE >= 32:
                    kpe_offset_bases: gl.constexpr = make_offset_bases(16, [4, 8, 16], [1, 2, 32], 0)
                    kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0], [2, 0]], [[4, 0], [8, 0], [16, 0], [0, 4], [0, 8], [0, 16]], [[0, 1], [0, 2], [0, 32]])
                else:
                    kpe_offset_bases: gl.constexpr = make_offset_bases(8, [4, 8, 16], [1, 2, 32], 0)
                    kpe_async_layout: gl.constexpr = make_dll([BLOCK_DPE, BLOCK_N], [[1, 0]], [[2, 0], [4, 0], [8, 0], [0, 4], [0, 8], [0, 16]], [[0, 1], [0, 2], [0, 32]])
                fp8_kpe_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DPE, BLOCK_N], kpe_offset_bases, [[1024, ASYNC_PAD_K], [2048, 32]])
                kpe_smem = gl.allocate_shared_memory(
                    PFX_SMEM_TY,
                    [NUM_STAGES, BLOCK_DPE, BLOCK_N],
                    layout=fp8_kpe_async_smem_layout,
                )
            else:
                kpe_smem = kt_smem
                kpe_async_layout: gl.constexpr = kt_async_layout

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=PFX_SMEM_TY,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            fp8_q_dot = gl.convert_layout(q.to(tl.float8e4nv), fp8_q_dot_layout)
            fp8_qpe_dot = gl.convert_layout(qpe.to(tl.float8e4nv), fp8_q_dot_layout)
            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix dispatch -- FP8 dot layouts for native FP8 MFMA.
            # Split into unmasked bulk + masked tail.
            pfx_full_len = (pfx_seq_len // BLOCK_N) * BLOCK_N
            n_full_prefix = pfx_seq_len // BLOCK_N
            if n_full_prefix >= NUM_STAGES:
                if NUM_STAGES >= 3:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined_scalar_mask(
                        acc,
                        l_i,
                        m_i,
                        fp8_q_dot,
                        fp8_qpe_dot,
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
                        fp8_kt_dot_layout,
                        fp8_p_dot_layout,
                        fp8_v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        ASYNC_KPE,
                    )
                else:
                    acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                        acc,
                        l_i,
                        m_i,
                        fp8_q_dot,
                        fp8_qpe_dot,
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
                        fp8_kt_dot_layout,
                        fp8_p_dot_layout,
                        fp8_v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        ASYNC_KPE,
                    )
                if pfx_seq_len > pfx_full_len:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        fp8_q_dot,
                        fp8_qpe_dot,
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
                        fp8_kt_dot_layout,
                        fp8_p_dot_layout,
                        fp8_v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        V_PRELOAD=False,
                        block_start=n_full_prefix,
                    )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_short(
                    acc,
                    l_i,
                    m_i,
                    fp8_q_dot,
                    fp8_qpe_dot,
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
                    fp8_kt_dot_layout,
                    fp8_p_dot_layout,
                    fp8_v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    V_PRELOAD=False,
                )

            # After prefix, before extend - transition FP8 smem -> BF16 smem
            kt_smem._keep_alive()
            v_smem._keep_alive()
            if ASYNC_KPE:
                kpe_smem._keep_alive()
            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=bf16_kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=bf16_v_async_smem_layout,
            )
            if ASYNC_KPE:
                bf16_kpe_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DPE, BLOCK_N], bf16_kpe_offset_bases, [[512, ASYNC_PAD_K]])
                kpe_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DPE, BLOCK_N],
                    layout=bf16_kpe_async_smem_layout,
                )
            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=bf16_v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            # EXTEND: per-CTA dispatch (v2 core change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

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
                    bf16_kt_async_layout,
                    bf16_kpe_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
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
                    bf16_kt_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=False,
                )
            if SLIDING_WINDOW_SIZE > 0 and IS_CAUSAL:
                swa_first_useful = tl.maximum(
                    cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0
                )
                swa_skip_n_blocks = swa_first_useful // BLOCK_N
            else:
                swa_skip_n_blocks = 0
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
                    bf16_kt_async_layout,
                    bf16_kpe_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
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
                    bf16_kt_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=False,
                )

        else:
            # 8-warp BLOCK_DMODEL < 128
            if BLOCK_N >= 128:
                kt_offset_bases: gl.constexpr = make_offset_bases(32, [16, 32, 64], [1, 2, 4, 8], 0)
                v_offset_bases: gl.constexpr = make_offset_bases(32, [16, 32, 64], [1, 2, 4, 8], 1)
                kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0], [0, 64]], [[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]], [[0, 2], [0, 4], [0, 8]])
                v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4], [64, 0]], [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]], [[2, 0], [4, 0], [8, 0]])
            else:
                kt_offset_bases: gl.constexpr = make_offset_bases(32, [16, 32], [1, 2, 4, 8], 0)
                v_offset_bases: gl.constexpr = make_offset_bases(32, [16, 32], [1, 2, 4, 8], 1)
                kt_async_layout: gl.constexpr = make_dll([BLOCK_DMODEL, BLOCK_N], [[1, 0], [2, 0], [4, 0]], [[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 1]], [[0, 2], [0, 4], [0, 8]])
                v_async_layout: gl.constexpr = make_dll([BLOCK_N, BLOCK_DV], [[0, 1], [0, 2], [0, 4]], [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [1, 0]], [[2, 0], [4, 0], [8, 0]])

            # FP8 prefix smem: 1024-byte interval for 128-bit direct-to-LDS
            fp8_kt_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], kt_offset_bases, [[1024, ASYNC_PAD_K], [2048, 32]])
            fp8_v_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], v_offset_bases, [[1024, ASYNC_PAD_V], [2048, 32]])
            bf16_kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
            bf16_kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
            bf16_v_offset_bases: gl.constexpr = make_fp8_bf16_v_offset_bases(num_warps, BLOCK_DV, BLOCK_N)
            bf16_v_async_layout: gl.constexpr = make_fp8_bf16_v_dll(num_warps, BLOCK_DV, BLOCK_N)
            # BF16 extend smem: standard 512-byte interval, using BF16 offset bases
            bf16_kt_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], bf16_kt_offset_bases, [[512, ASYNC_PAD_K]])
            bf16_v_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], bf16_v_offset_bases, [[512, ASYNC_PAD_V]])
            if BLOCK_DPE == 0:
                bf16_kpe_async_layout: gl.constexpr = bf16_kt_async_layout
                bf16_kpe_offset_bases: gl.constexpr = bf16_kt_offset_bases

            kt_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=fp8_kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=fp8_v_async_smem_layout,
            )

            ASYNC_KPE: gl.constexpr = False
            kpe_smem = kt_smem
            kpe_async_layout: gl.constexpr = kt_async_layout

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=PFX_SMEM_TY,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            fp8_q_dot = gl.convert_layout(q.to(tl.float8e4nv), fp8_q_dot_layout)
            fp8_qpe_dot = gl.convert_layout(qpe.to(tl.float8e4nv), fp8_q_dot_layout)
            q_dot = gl.convert_layout(q, q_dot_layout)

            # prefix -- FP8 dot layouts for native FP8 MFMA.
            # Split into unmasked bulk + masked tail.
            pfx_full_len = (pfx_seq_len // BLOCK_N) * BLOCK_N
            n_full_prefix = pfx_seq_len // BLOCK_N
            if n_full_prefix >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_prefix_pipelined(
                    acc,
                    l_i,
                    m_i,
                    fp8_q_dot,
                    fp8_qpe_dot,
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
                    fp8_kt_dot_layout,
                    fp8_p_dot_layout,
                    fp8_v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    ASYNC_KPE,
                )
                if pfx_seq_len > pfx_full_len:
                    acc, l_i, m_i = attn_fwd_inner_prefix_short(
                        acc,
                        l_i,
                        m_i,
                        fp8_q_dot,
                        fp8_qpe_dot,
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
                        fp8_kt_dot_layout,
                        fp8_p_dot_layout,
                        fp8_v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        V_PRELOAD=False,
                        block_start=n_full_prefix,
                    )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_short(
                    acc,
                    l_i,
                    m_i,
                    fp8_q_dot,
                    fp8_qpe_dot,
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
                    fp8_kt_dot_layout,
                    fp8_p_dot_layout,
                    fp8_v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    V_PRELOAD=False,
                )

            # After prefix, before extend - transition FP8 smem -> BF16 smem
            kt_smem._keep_alive()
            v_smem._keep_alive()
            if ASYNC_KPE:
                kpe_smem._keep_alive()
            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=bf16_kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=bf16_v_async_smem_layout,
            )
            if ASYNC_KPE:
                bf16_kpe_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DPE, BLOCK_N], bf16_kpe_offset_bases, [[512, ASYNC_PAD_K]])
                kpe_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DPE, BLOCK_N],
                    layout=bf16_kpe_async_smem_layout,
                )
            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=bf16_v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            # EXTEND: per-CTA dispatch (v2 core change)
            if IS_CAUSAL:
                causal_kv_end = (cur_block_m + 1) * BLOCK_M
                effective_end = tl.minimum(seq_len_extend, causal_kv_end)
            else:
                effective_end = seq_len_extend
            n_extend_blocks = (effective_end + BLOCK_N - 1) // BLOCK_N
            if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                n_full_blocks = 0
            else:
                partial_block = ((effective_end % BLOCK_N) != 0).to(tl.int32)
                if IS_CAUSAL:
                    masked_blocks = ((BLOCK_M + BLOCK_N - 1) // BLOCK_N) + partial_block
                else:
                    masked_blocks = partial_block
                masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
                n_full_blocks = n_extend_blocks - masked_blocks

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
                    bf16_kt_async_layout,
                    bf16_kpe_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
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
                    bf16_kt_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=False,
                )
            if SLIDING_WINDOW_SIZE > 0 and IS_CAUSAL:
                swa_first_useful = tl.maximum(
                    cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0
                )
                swa_skip_n_blocks = swa_first_useful // BLOCK_N
            else:
                swa_skip_n_blocks = 0
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
                    bf16_kt_async_layout,
                    bf16_kpe_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    ASYNC_KPE,
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
                    bf16_kt_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                    V_PRELOAD=False,
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
