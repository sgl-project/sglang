# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prefix loop variant: existing DMA pipeline + RF-double-buffered kv_indices.

Based on attn_fwd_inner_prefix_pipelined.  All K/V data still flows through
the cdna4_async DMA pipeline (global → LDS via buffer_load_to_shared).

Only change: kv_indices are loaded via cdna4_buffer_load (non-blocking
global → RF) instead of gl.load, and held in two register sets (A/B) that
alternate each unrolled iteration pair.
"""

import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import (
    buffer_load as cdna4_buffer_load,
)
from triton.experimental.gluon.language.amd import warp_pipeline_stage

from ._common import (
    do_mma, compute_softmax_prefix,
    issue_dma_k_prefix_from_locs, issue_dma_v_prefix_from_locs,
    issue_async_load_kpe_prefix, add_qk_dpe_from_shared,
    add_qk_dpe_prefix_from_kv,
)


@gluon.jit
def attn_fwd_inner_prefix_pipelined_rfidx(
    acc,
    l_i,
    m_i,
    q_dot,
    qpe_dot,
    K_Buffer,
    V_Buffer,
    kv_indices,
    kv_start,
    cur_kv_head,
    seq_len_prefix,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    kt_smem,
    kpe_smem,
    v_smem,
    qk_scale,
    LOGIT_CAP: gl.constexpr,
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    kt_async_layout: gl.constexpr,
    kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    ASYNC_KPE: gl.constexpr = False,
):
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    USE_ASYNC_KPE: gl.constexpr = HAS_DPE and ASYNC_KPE
    STREAMS: gl.constexpr = 3 if USE_ASYNC_KPE else 2
    # warp_pipeline_stage requires >=2 physical LDS buffers. With NS=1 the
    # stage_idx = block_n % NUM_STAGES collapses to 0, and iter N's DMA
    # write to smem[0] races iter N+1's relaxed read from smem[0] because
    # membarFilter skips barriers between BufferLoadToLocalOp and
    # syncedViaAsyncWait LocalLoadOp. Callers wanting NS=1 must route to
    # the non-pipelined DMA path (attn_fwd_inner_prefix_dma_simple).
    tl.static_assert(
        NUM_STAGES >= 2,
        "attn_fwd_inner_prefix_pipelined_rfidx requires NUM_STAGES>=2 for "
        "determinism (warp_pipeline_stage needs multiple LDS buffers).",
    )

    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh
    kv_indices_base = kv_indices + kv_start

    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)

    # ------------------------------------------------------------------ #
    # Prologue: fill SMEM stages 0..NUM_STAGES-1.
    # Uses cdna4_buffer_load for kv_indices (non-blocking RF load)
    # then issue_dma_*_from_locs for K/V data DMA to LDS.
    # ------------------------------------------------------------------ #
    for stage in gl.static_range(NUM_STAGES):
        pf_init_n = stage * BLOCK_N
        n_idx_k = pf_init_n + kt_offs_n
        mask_n_k = n_idx_k < seq_len_prefix
        kv_locs_k = cdna4_buffer_load(
            kv_indices_base, n_idx_k.to(tl.int32), mask=mask_n_k, other=0
        ).to(tl.int32)
        n_idx_v = pf_init_n + v_offs_n
        mask_n_v = n_idx_v < seq_len_prefix
        kv_locs_v = cdna4_buffer_load(
            kv_indices_base, n_idx_v.to(tl.int32), mask=mask_n_v, other=0
        ).to(tl.int32)
        issue_dma_k_prefix_from_locs(
            kt_smem.index(stage), k_prefix_base, kv_locs_k, mask_n_k,
            stride_buf_kbs, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_prefix(
                kpe_smem.index(stage), k_prefix_base, kv_indices, kv_start,
                pf_init_n, seq_len_prefix, stride_buf_kbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                kpe_async_layout,
            )
        issue_dma_v_prefix_from_locs(
            v_smem.index(stage), v_prefix_base, kv_locs_v, mask_n_v,
            stride_buf_vbs, BLOCK_N, BLOCK_DV, ACTUAL_BLOCK_DV,
            v_async_layout,
        )

    # ------------------------------------------------------------------ #
    # RF register set A: indices for tile NUM_STAGES
    # (first tile the main loop will issue DMA for).
    # ------------------------------------------------------------------ #
    pf_start_n = NUM_STAGES * BLOCK_N
    n_idx_kt_pf = pf_start_n + kt_offs_n
    mask_n_kt_pf = n_idx_kt_pf < seq_len_prefix
    kv_locs_kt_pf = cdna4_buffer_load(
        kv_indices_base, n_idx_kt_pf.to(tl.int32), mask=mask_n_kt_pf, other=0
    ).to(tl.int32)
    n_idx_v_pf = pf_start_n + v_offs_n
    mask_n_v_pf = n_idx_v_pf < seq_len_prefix
    kv_locs_v_pf = cdna4_buffer_load(
        kv_indices_base, n_idx_v_pf.to(tl.int32), mask=mask_n_v_pf, other=0
    ).to(tl.int32)

    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    # ------------------------------------------------------------------ #
    # Main loop — identical structure to attn_fwd_inner_prefix_pipelined
    # but gl.load for kv_indices → cdna4_buffer_load.
    # ------------------------------------------------------------------ #
    main_loop_end = n_prefix_blocks - NUM_STAGES
    for block_n in tl.range(0, main_loop_end, loop_unroll_factor=2):

        with warp_pipeline_stage("compute0", priority=0):
            stage_idx = (block_n % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = do_mma(q_dot, kt_dot, qk)
            if USE_ASYNC_KPE:
                qk = add_qk_dpe_from_shared(
                    qk, qpe_dot, kpe_smem.index(stage_idx), kt_dot_layout,
                    BLOCK_DPE,
                )
            elif HAS_DPE:
                qk = add_qk_dpe_prefix_from_kv(
                    qk, qpe_dot, K_Buffer, kv_indices, kv_start, start_n,
                    cur_kv_head, seq_len_prefix, stride_buf_kbs, stride_buf_kh,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                    kt_dot_layout,
                )

        cdna4_async.wait_group(WAIT_V)

        with warp_pipeline_stage("memory0", priority=1):
            v_dot = cdna4_async.load_shared_relaxed(
                v_smem.index(stage_idx), v_dot_layout
            )
            issue_dma_k_prefix_from_locs(
                kt_smem.index(stage_idx),
                k_prefix_base,
                kv_locs_kt_pf,
                mask_n_kt_pf,
                stride_buf_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL,
                kt_async_layout,
            )
            if USE_ASYNC_KPE:
                pf_start_kpe = (block_n + NUM_STAGES) * BLOCK_N
                issue_async_load_kpe_prefix(
                    kpe_smem.index(stage_idx),
                    k_prefix_base,
                    kv_indices,
                    kv_start,
                    pf_start_kpe,
                    seq_len_prefix,
                    stride_buf_kbs,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    kpe_async_layout,
                )

        with warp_pipeline_stage("compute1", priority=0):
            acc, l_i, m_i, p = compute_softmax_prefix(
                acc, l_i, m_i, qk, start_n, seq_len_prefix,
                qk_scale, LOGIT_CAP,
                xai_temperature_reg, XAI_TEMPERATURE_LEN,
                q_abs_pos, SLIDING_WINDOW_SIZE,
                Mask, mask_base_idx, mask_row_stride, q_extend_offs,
                USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK, ENABLE_PREFIX_UNMASKED,
                BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col,
            )

        with warp_pipeline_stage("memory1", priority=1):
            issue_dma_v_prefix_from_locs(
                v_smem.index(stage_idx),
                v_prefix_base,
                kv_locs_v_pf,
                mask_n_v_pf,
                stride_buf_vbs,
                BLOCK_N,
                BLOCK_DV,
                ACTUAL_BLOCK_DV,
                v_async_layout,
            )
            # RF set B: prefetch K indices for tile (block_n + NUM_STAGES + 1)
            nf_start_n = ((block_n + NUM_STAGES + 1) * BLOCK_N).to(tl.int32)
            n_idx_kt_nf = nf_start_n + kt_offs_n
            mask_n_kt_pf = n_idx_kt_nf < seq_len_prefix
            kv_locs_kt_pf = cdna4_buffer_load(
                kv_indices_base, n_idx_kt_nf.to(tl.int32),
                mask=mask_n_kt_pf, other=0,
            ).to(tl.int32)

        with warp_pipeline_stage("compute2", priority=0):
            p_cast = p.to(v_dot.dtype)
            p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
            acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_K)

        with warp_pipeline_stage("memory2", priority=1):
            next_stage_idx = ((block_n + 1) % NUM_STAGES).to(tl.int32)
            kt_dot = cdna4_async.load_shared_relaxed(
                kt_smem.index(next_stage_idx), kt_dot_layout
            )
            # RF set B: prefetch V indices for tile (block_n + NUM_STAGES + 1)
            n_idx_v_nf = nf_start_n + v_offs_n
            mask_n_v_pf = n_idx_v_nf < seq_len_prefix
            kv_locs_v_pf = cdna4_buffer_load(
                kv_indices_base, n_idx_v_nf.to(tl.int32),
                mask=mask_n_v_pf, other=0,
            ).to(tl.int32)

    # ------------------------------------------------------------------ #
    # Tail: drain remaining NUM_STAGES tiles already in SMEM.
    # ------------------------------------------------------------------ #
    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - (STREAMS - 1))
        stage_idx = ((main_loop_end + tail_i) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(stage_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)
        if USE_ASYNC_KPE:
            qk = add_qk_dpe_from_shared(
                qk, qpe_dot, kpe_smem.index(stage_idx), kt_dot_layout,
                BLOCK_DPE,
            )
        elif HAS_DPE:
            qk = add_qk_dpe_prefix_from_kv(
                qk, qpe_dot, K_Buffer, kv_indices, kv_start, start_n,
                cur_kv_head, seq_len_prefix, stride_buf_kbs, stride_buf_kh,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                kt_dot_layout,
            )

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc, l_i, m_i, qk, start_n, seq_len_prefix,
            qk_scale, LOGIT_CAP,
            xai_temperature_reg, XAI_TEMPERATURE_LEN,
            q_abs_pos, SLIDING_WINDOW_SIZE,
            Mask, mask_base_idx, mask_row_stride, q_extend_offs,
            USE_CUSTOM_MASK, SKIP_PREFIX_CUSTOM_MASK, ENABLE_PREFIX_UNMASKED,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - STREAMS)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i
