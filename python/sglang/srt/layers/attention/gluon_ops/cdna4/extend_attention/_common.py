# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon extend-attention kernel for gfx950 (MI350X / CDNA4).

Three dispatch paths -- basic request-centric, mask-split, and persistent-CTA
(work-centric scheduling) -- are selected at launch time based on workload shape.
Supports D=64/128/256, causal/non-causal, logit cap, sliding window, custom mask,
FP8 KV dequant (k_scale/v_scale), and XAI temperature.
"""

import math
import os

import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, warp_pipeline_stage
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async
from triton.experimental.gluon.language.amd.cdna4 import (
    buffer_load as cdna4_buffer_load,
    buffer_store as cdna4_buffer_store,
)
from triton.experimental.gluon.language.amd.cdna4 import mfma as mfma_cdna4
from triton.experimental.gluon.language._layouts import (
    DistributedLinearLayout,
    DotOperandLayout,
    PaddedSharedLayout,
)

LOG2E = tl.constexpr(1.4426950408889634)


# ===-----------------------------------------------------------------------===#
# Primitives (inlined from f16_extend_attention_gfx950)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def _nan_propagating_max(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def nan_propagating_max(x, axis):
    return gl.reduce(x, axis, _nan_propagating_max)


@gluon.jit
def do_mma(a, b, c):
    if b.dtype == tl.float8e4b8 or b.dtype == tl.float8e4nv:
        a_fp8 = tl.cast(a, tl.float8e4nv, bitcast=(a.dtype != tl.bfloat16 and a.dtype != tl.float16))
        b_fp8 = tl.cast(b, tl.float8e4nv, bitcast=True)
        return mfma_cdna4(a_fp8, b_fp8, c)
    return mfma_cdna4(a.to(tl.bfloat16), b.to(tl.bfloat16), c)


@gluon.jit
def _buffer_load_to_shared_cast_safe(dst_smem, src_base, offsets, mask, other):
    """Type-safe global->shared load for mixed fp8/bf16 buffers."""
    if dst_smem.dtype == src_base.dtype.element_ty:
        if mask is not None:
            cdna4_async.buffer_load_to_shared(
                dst_smem, src_base, offsets, mask=mask, other=other
            )
        else:
            cdna4_async.buffer_load_to_shared(
                dst_smem, src_base, offsets
            )
    else:
        if mask is not None:
            reg = cdna4_buffer_load(src_base, offsets, mask=mask, other=other)
        else:
            reg = cdna4_buffer_load(src_base, offsets)
        dst_smem.store(reg.to(dst_smem.dtype))
    cdna4_async.commit_group()


@gluon.jit
def issue_async_load_k_prefix(
    kt_smem,
    k_prefix_base,  #
    kv_indices,
    kv_start,
    start_n,
    seq_len_prefix,  #
    stride_buf_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)

    n_idx = start_n + kt_offs_n
    mask_n = n_idx < seq_len_prefix
    kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)

    kt_offsets = (kt_offs_d[:, None] + kv_locs[None, :] * stride_buf_kbs).to(tl.int32)

    kt_mask = mask_n[None, :]
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
    _buffer_load_to_shared_cast_safe(
        kt_smem, k_prefix_base, kt_offsets, mask=kt_mask, other=0.0
    )


@gluon.jit
def issue_async_load_v_prefix(
    v_smem,
    v_prefix_base,  #
    kv_indices,
    kv_start,
    start_n,
    seq_len_prefix,  #
    stride_buf_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    v_async_layout: gl.constexpr,  #
):
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)

    n_idx = start_n + v_offs_n
    mask_n = n_idx < seq_len_prefix
    kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)

    v_offsets = (kv_locs[:, None] * stride_buf_vbs + v_offs_d[None, :]).to(tl.int32)

    v_mask = mask_n[:, None]
    if ACTUAL_BLOCK_DV != BLOCK_DV:
        v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DV)
    _buffer_load_to_shared_cast_safe(
        v_smem, v_prefix_base, v_offsets, mask=v_mask, other=0.0
    )


@gluon.jit
def issue_dma_k_prefix_from_locs(
    kt_smem,
    k_prefix_base,  #
    kv_locs,
    mask_n,
    stride_buf_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offsets = (kt_offs_d[:, None] + kv_locs[None, :] * stride_buf_kbs).to(tl.int32)
    kt_mask = mask_n[None, :]
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
    _buffer_load_to_shared_cast_safe(
        kt_smem, k_prefix_base, kt_offsets, mask=kt_mask, other=0.0
    )


@gluon.jit
def issue_dma_v_prefix_from_locs(
    v_smem,
    v_prefix_base,  #
    kv_locs,
    mask_n,
    stride_buf_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    v_async_layout: gl.constexpr,  #
):
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    v_offsets = (kv_locs[:, None] * stride_buf_vbs + v_offs_d[None, :]).to(tl.int32)
    v_mask = mask_n[:, None]
    if ACTUAL_BLOCK_DV != BLOCK_DV:
        v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DV)
    _buffer_load_to_shared_cast_safe(
        v_smem, v_prefix_base, v_offsets, mask=v_mask, other=0.0
    )


@gluon.jit
def issue_dma_k_prefix_from_locs_hot(
    kt_smem,
    k_prefix_base,  #
    kv_locs,
    scalar_mask,
    stride_buf_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,  #
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offsets = (kt_offs_d[:, None] + kv_locs[None, :] * stride_buf_kbs).to(tl.int32)
    if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
        kt_mask = kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL
        _buffer_load_to_shared_cast_safe(
            kt_smem, k_prefix_base, kt_offsets, mask=kt_mask, other=0.0
        )
    else:
        _buffer_load_to_shared_cast_safe(
            kt_smem, k_prefix_base, kt_offsets, mask=scalar_mask, other=0.0
        )


@gluon.jit
def issue_dma_v_prefix_from_locs_hot(
    v_smem,
    v_prefix_base,  #
    kv_locs,
    scalar_mask,
    stride_buf_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    v_async_layout: gl.constexpr,  #
):
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    v_offsets = (kv_locs[:, None] * stride_buf_vbs + v_offs_d[None, :]).to(tl.int32)
    if ACTUAL_BLOCK_DV != BLOCK_DV:
        v_mask = v_offs_d[None, :] < ACTUAL_BLOCK_DV
        _buffer_load_to_shared_cast_safe(
            v_smem, v_prefix_base, v_offsets, mask=v_mask, other=0.0
        )
    else:
        _buffer_load_to_shared_cast_safe(
            v_smem, v_prefix_base, v_offsets, mask=scalar_mask, other=0.0
        )


@gluon.jit
def issue_async_load_k_extend(
    kt_smem,
    k_base,
    start_n,
    seq_len_extend,  #
    stride_kbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    kt_async_layout: gl.constexpr,  #
    SKIP_BOUNDS_CHECK: gl.constexpr = False,
):
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    kt_offsets = (kt_offs_d[:, None] + (start_n + kt_offs_n[None, :]) * stride_kbs).to(
        tl.int32
    )

    if SKIP_BOUNDS_CHECK:
        kt_mask = None
    else:
        kt_mask = (start_n + kt_offs_n[None, :]) < seq_len_extend
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
    _buffer_load_to_shared_cast_safe(
        kt_smem, k_base, kt_offsets, mask=kt_mask, other=0.0
    )


@gluon.jit
def issue_async_load_v_extend(
    v_smem,
    v_base,
    start_n,
    seq_len_extend,  #
    stride_vbs,  #
    BLOCK_N: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    v_async_layout: gl.constexpr,  #
    SKIP_BOUNDS_CHECK: gl.constexpr = False,
):
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DV, layout=v_offs_d_layout)
    v_offsets = ((start_n + v_offs_n[:, None]) * stride_vbs + v_offs_d[None, :]).to(
        tl.int32
    )

    if SKIP_BOUNDS_CHECK:
        v_mask = None
    else:
        v_mask = (start_n + v_offs_n[:, None]) < seq_len_extend
        if ACTUAL_BLOCK_DV != BLOCK_DV:
            v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DV)
    _buffer_load_to_shared_cast_safe(
        v_smem, v_base, v_offsets, mask=v_mask, other=0.0
    )


@gluon.jit
def add_qk_dpe_prefix_from_kv(
    qk,
    qpe_dot,
    K_Buffer,
    kv_indices,
    kv_start,
    start_n,
    cur_kv_head,
    seq_len_prefix,
    stride_buf_kbs,
    stride_buf_kh,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,
    kt_dot_layout: gl.constexpr,
):
    if BLOCK_DPE > 0:
        offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_dot_layout))
        offs_dpe = gl.arange(0, BLOCK_DPE)
        n_idx = start_n + offs_n
        mask_n = n_idx < seq_len_prefix
        kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)
        kpe_ptrs = (
            K_Buffer
            + kv_locs[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + BLOCK_DMODEL
            + offs_dpe[:, None]
        )
        kpe_mask = mask_n[None, :]
        if ACTUAL_BLOCK_DPE != BLOCK_DPE:
            kpe_mask = kpe_mask & (offs_dpe[:, None] < ACTUAL_BLOCK_DPE)
        kpe = gl.load(kpe_ptrs, mask=kpe_mask, other=0.0)
        kpe_dot = gl.convert_layout(kpe, kt_dot_layout)
        qk = do_mma(qpe_dot, kpe_dot, qk)
    return qk


@gluon.jit
def add_qk_dpe_extend_from_base(
    qk,
    qpe_dot,
    k_base,
    start_n,
    seq_len_extend,
    stride_kbs,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,
    kt_dot_layout: gl.constexpr,
):
    if BLOCK_DPE > 0:
        offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_dot_layout))
        offs_dpe = gl.arange(0, BLOCK_DPE)
        kpe_ptrs = (
            k_base
            + (start_n + offs_n[None, :]) * stride_kbs
            + BLOCK_DMODEL
            + offs_dpe[:, None]
        )
        kpe_mask = (start_n + offs_n[None, :]) < seq_len_extend
        if ACTUAL_BLOCK_DPE != BLOCK_DPE:
            kpe_mask = kpe_mask & (offs_dpe[:, None] < ACTUAL_BLOCK_DPE)
        kpe = gl.load(kpe_ptrs, mask=kpe_mask, other=0.0)
        kpe_dot = gl.convert_layout(kpe, kt_dot_layout)
        qk = do_mma(qpe_dot, kpe_dot, qk)
    return qk


@gluon.jit
def issue_async_load_kpe_extend(
    kpe_smem,
    k_base,
    start_n,
    seq_len_extend,
    stride_kbs,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,
    kpe_async_layout: gl.constexpr,
):
    """Async DMA load of transposed K_DPE [BLOCK_DPE, BLOCK_N] into shared memory."""
    if BLOCK_DPE > 0:
        kpe_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
        kpe_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kpe_async_layout)
        kpe_offs_d = gl.arange(0, BLOCK_DPE, layout=kpe_offs_d_layout)
        kpe_offs_n = gl.arange(0, BLOCK_N, layout=kpe_offs_n_layout)
        kpe_offsets = (
            (kpe_offs_d[:, None] + BLOCK_DMODEL)
            + (start_n + kpe_offs_n[None, :]) * stride_kbs
        ).to(tl.int32)
        kpe_mask = (start_n + kpe_offs_n[None, :]) < seq_len_extend
        if ACTUAL_BLOCK_DPE != BLOCK_DPE:
            kpe_mask = kpe_mask & (kpe_offs_d[:, None] < ACTUAL_BLOCK_DPE)
        _buffer_load_to_shared_cast_safe(
            kpe_smem, k_base, kpe_offsets, mask=kpe_mask, other=0.0
        )


@gluon.jit
def issue_async_load_kpe_prefix(
    kpe_smem,
    k_prefix_base,
    kv_indices,
    kv_start,
    start_n,
    seq_len_prefix,
    stride_buf_kbs,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,
    kpe_async_layout: gl.constexpr,
):
    """Async DMA load of transposed K_DPE [BLOCK_DPE, BLOCK_N] from prefix (scattered)."""
    if BLOCK_DPE > 0:
        kpe_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
        kpe_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kpe_async_layout)
        kpe_offs_d = gl.arange(0, BLOCK_DPE, layout=kpe_offs_d_layout)
        kpe_offs_n = gl.arange(0, BLOCK_N, layout=kpe_offs_n_layout)
        n_idx = start_n + kpe_offs_n
        mask_n = n_idx < seq_len_prefix
        kv_locs = gl.load(kv_indices + kv_start + n_idx, mask=mask_n, other=0).to(tl.int32)
        kpe_offsets = (
            (kpe_offs_d[:, None] + BLOCK_DMODEL)
            + kv_locs[None, :] * stride_buf_kbs
        ).to(tl.int32)
        kpe_mask = mask_n[None, :]
        if ACTUAL_BLOCK_DPE != BLOCK_DPE:
            kpe_mask = kpe_mask & (kpe_offs_d[:, None] < ACTUAL_BLOCK_DPE)
        _buffer_load_to_shared_cast_safe(
            kpe_smem, k_prefix_base, kpe_offsets, mask=kpe_mask, other=0.0
        )


@gluon.jit
def issue_dma_kpe_prefix_from_locs(
    kpe_smem,
    k_prefix_base,
    kv_locs,
    mask_n,
    stride_buf_kbs,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,
    kpe_async_layout: gl.constexpr,
):
    """DMA variant for prefix KPE with pre-loaded kv_locs."""
    if BLOCK_DPE > 0:
        kpe_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kpe_async_layout)
        kpe_offs_d = gl.arange(0, BLOCK_DPE, layout=kpe_offs_d_layout)
        kpe_offsets = (
            (kpe_offs_d[:, None] + BLOCK_DMODEL)
            + kv_locs[None, :] * stride_buf_kbs
        ).to(tl.int32)
        kpe_mask = mask_n[None, :]
        if ACTUAL_BLOCK_DPE != BLOCK_DPE:
            kpe_mask = kpe_mask & (kpe_offs_d[:, None] < ACTUAL_BLOCK_DPE)
        _buffer_load_to_shared_cast_safe(
            kpe_smem, k_prefix_base, kpe_offsets, mask=kpe_mask, other=0.0
        )


@gluon.jit
def add_qk_dpe_from_shared(
    qk,
    qpe_dot,
    kpe_smem,
    kt_dot_layout: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
):
    """Accumulate Q_DPE x K_DPE from pipelined shared memory into qk."""
    if BLOCK_DPE > 0:
        kpe_dot = kpe_smem.load(kt_dot_layout)
        qk = do_mma(qpe_dot, kpe_dot, qk)
    return qk


@gluon.jit
def compute_softmax_prefix(
    acc,
    l_i,
    m_i,
    qk,
    start_n,
    seq_len_prefix,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
):
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)
    if XAI_TEMPERATURE_LEN > 0:
        qk_scaled = qk_scaled * xai_temperature_reg[:, None]
    bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
    use_custom_prefix_mask = USE_CUSTOM_MASK and (not SKIP_PREFIX_CUSTOM_MASK)
    is_partial_tail = (start_n + BLOCK_N) > seq_len_prefix
    if SLIDING_WINDOW_SIZE > 0:
        swa_safe = (tl.max(q_abs_pos) <= start_n + SLIDING_WINDOW_SIZE)
    else:
        swa_safe = True
    use_unmasked_path = (
        ENABLE_PREFIX_UNMASKED
        and swa_safe
        and (not use_custom_prefix_mask)
        and (not is_partial_tail)
    )

    if use_unmasked_path:
        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk_scaled - m_new[:, None])
    else:
        bound_mask = (q_abs_pos[:, None] >= 0) & (bound_offs[None, :] < seq_len_prefix)
        if SLIDING_WINDOW_SIZE > 0:
            bound_mask = bound_mask & (
                q_abs_pos[:, None] <= bound_offs[None, :] + SLIDING_WINDOW_SIZE
            )
        if use_custom_prefix_mask:
            mask_ptrs = (
                Mask
                + mask_base_idx
                + q_extend_offs[:, None] * mask_row_stride
                + start_n
                + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)[None, :]
            )
            mask_vals = gl.load(mask_ptrs, mask=bound_mask, other=0)
            bound_mask = bound_mask & (mask_vals != 0)
        qk_scaled = gl.where(
            bound_mask,
            qk_scaled,
            gl.full(
                [BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout
            ),
        )

        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        if SLIDING_WINDOW_SIZE > 0 or use_custom_prefix_mask:
            m_new = gl.maximum(
                m_new,
                gl.full(
                    [BLOCK_M],
                    -1e20,
                    dtype=gl.float32,
                    layout=gl.SliceLayout(dim=1, parent=mma_layout),
                ),
            )
        p = gl.exp2(qk_scaled - m_new[:, None])
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


@gluon.jit
def compute_softmax_extend(
    acc,
    l_i,
    m_i,
    qk,
    start_n,
    cur_block_m,
    seq_len_extend,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)
    if XAI_TEMPERATURE_LEN > 0:
        qk_scaled = qk_scaled * xai_temperature_reg[:, None]
    if MASK_STEPS:
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
        q_offs = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        valid_mask = q_offs[:, None] < seq_len_extend
        valid_mask = valid_mask & (bound_offs[None, :] < seq_len_extend)
        if USE_CUSTOM_MASK:
            mask_ptrs = (
                Mask
                + mask_base_idx
                + q_offs[:, None] * mask_row_stride
                + mask_kv_col_offset
                + start_n
                + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)[None, :]
            )
            mask_vals = gl.load(mask_ptrs, mask=valid_mask, other=0)
            valid_mask = valid_mask & (mask_vals != 0)
        elif IS_CAUSAL:
            valid_mask = valid_mask & (q_offs[:, None] >= bound_offs[None, :])
        if SLIDING_WINDOW_SIZE > 0:
            valid_mask = valid_mask & (
                q_offs[:, None] <= bound_offs[None, :] + SLIDING_WINDOW_SIZE
            )
        qk_scaled = gl.where(
            valid_mask,
            qk_scaled,
            gl.full(
                [BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout
            ),
        )

        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        if SLIDING_WINDOW_SIZE > 0 or USE_CUSTOM_MASK:
            m_new = gl.maximum(
                m_new,
                gl.full(
                    [BLOCK_M],
                    -1e20,
                    dtype=gl.float32,
                    layout=gl.SliceLayout(dim=1, parent=mma_layout),
                ),
            )
        p = gl.exp2(qk_scaled - m_new[:, None])
    else:
        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk_scaled - m_new[:, None])
    l_ij = gl.sum(p, axis=1)
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i, p


@gluon.jit
def compute_softmax_extend_part0(
    m_i,
    qk,
    start_n,
    cur_block_m,
    seq_len_extend,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
):
    """Scale, optional mask, max reduction, exp2, alpha.

    Produces (p, alpha, m_new) without touching acc or l_i.
    Intended to run between QK MMA and V wait so the VALU work
    overlaps with the V global→LDS DMA.
    """
    qk_scaled = qk * qk_scale
    if LOGIT_CAP > 0:
        log2_cap: gl.constexpr = LOGIT_CAP * LOG2E
        inv_cap: gl.constexpr = 2.0 / LOGIT_CAP
        e_neg = tl.math.exp2(-qk_scaled * inv_cap)
        sig = 1.0 / (1.0 + e_neg)
        qk_scaled = log2_cap * (2.0 * sig - 1.0)
    if XAI_TEMPERATURE_LEN > 0:
        qk_scaled = qk_scaled * xai_temperature_reg[:, None]
    if MASK_STEPS:
        bound_offs = start_n + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)
        q_offs = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        valid_mask = q_offs[:, None] < seq_len_extend
        valid_mask = valid_mask & (bound_offs[None, :] < seq_len_extend)
        if USE_CUSTOM_MASK:
            mask_ptrs = (
                Mask
                + mask_base_idx
                + q_offs[:, None] * mask_row_stride
                + mask_kv_col_offset
                + start_n
                + gl.arange(0, BLOCK_N, layout=mma_offs_n_col)[None, :]
            )
            mask_vals = gl.load(mask_ptrs, mask=valid_mask, other=0)
            valid_mask = valid_mask & (mask_vals != 0)
        elif IS_CAUSAL:
            valid_mask = valid_mask & (q_offs[:, None] >= bound_offs[None, :])
        if SLIDING_WINDOW_SIZE > 0:
            valid_mask = valid_mask & (
                q_offs[:, None] <= bound_offs[None, :] + SLIDING_WINDOW_SIZE
            )
        qk_scaled = gl.where(
            valid_mask,
            qk_scaled,
            gl.full(
                [BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=mma_layout
            ),
        )

        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        if SLIDING_WINDOW_SIZE > 0 or USE_CUSTOM_MASK:
            m_new = gl.maximum(
                m_new,
                gl.full(
                    [BLOCK_M],
                    -1e20,
                    dtype=gl.float32,
                    layout=gl.SliceLayout(dim=1, parent=mma_layout),
                ),
            )
        p = gl.exp2(qk_scaled - m_new[:, None])
    else:
        m_ij = nan_propagating_max(qk_scaled, axis=1)
        m_new = gl.maximum(m_i, m_ij, propagate_nan=tl.PropagateNan.ALL)
        p = gl.exp2(qk_scaled - m_new[:, None])
    alpha = gl.exp2(m_i - m_new)
    return p, alpha, m_new


@gluon.jit
def compute_softmax_extend_part1(acc, l_i, p, alpha, m_new):
    """Row-sum, rescale acc and l_i, update m_i.

    Runs after V is available so the acc rescale feeds directly
    into the P*V MMA that follows.
    """
    l_ij = gl.sum(p, axis=1)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    m_i = m_new
    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Prefix inner loops
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_prefix_pipelined(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    kpe_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
    ASYNC_KPE: gl.constexpr = False,
):
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    USE_ASYNC_KPE: gl.constexpr = HAS_DPE and ASYNC_KPE
    STREAMS: gl.constexpr = 3 if USE_ASYNC_KPE else 2
    # warp_pipeline_stage requires >=2 physical LDS buffers. With NS=1 the
    # stage_idx collapses to 0, and iter N's DMA write to smem[0] races
    # iter N+1's relaxed read from smem[0] because membarFilter skips
    # barriers between BufferLoadToLocalOp and syncedViaAsyncWait loads.
    # Callers wanting NS=1 must route to the non-pipelined DMA path
    # (attn_fwd_inner_prefix_dma_simple).
    tl.static_assert(
        NUM_STAGES >= 2,
        "attn_fwd_inner_prefix_pipelined requires NUM_STAGES>=2 for "
        "determinism (warp_pipeline_stage needs multiple LDS buffers).",
    )

    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh

    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    kt_offs_n_pf = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    v_offs_n_pf = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)

    for stage in gl.static_range(NUM_STAGES):
        pf_init_n = stage * BLOCK_N
        issue_async_load_k_prefix(
            kt_smem.index(stage),
            k_prefix_base,
            kv_indices,
            kv_start,
            pf_init_n,
            seq_len_prefix,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_prefix(
                kpe_smem.index(stage),
                k_prefix_base,
                kv_indices,
                kv_start,
                pf_init_n,
                seq_len_prefix,
                stride_buf_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DPE,
                ACTUAL_BLOCK_DPE,
                kpe_async_layout,
            )
        issue_async_load_v_prefix(
            v_smem.index(stage),
            v_prefix_base,
            kv_indices,
            kv_start,
            pf_init_n,
            seq_len_prefix,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DV,
            ACTUAL_BLOCK_DV,
            v_async_layout,
        )

    pf_start_n = NUM_STAGES * BLOCK_N
    n_idx_kt_pf = pf_start_n + kt_offs_n_pf
    mask_n_kt_pf = n_idx_kt_pf < seq_len_prefix
    kv_locs_kt_pf = gl.load(
        kv_indices + kv_start + n_idx_kt_pf, mask=mask_n_kt_pf, other=0
    ).to(tl.int32)
    n_idx_v_pf = pf_start_n + v_offs_n_pf
    mask_n_v_pf = n_idx_v_pf < seq_len_prefix
    kv_locs_v_pf = gl.load(
        kv_indices + kv_start + n_idx_v_pf, mask=mask_n_v_pf, other=0
    ).to(tl.int32)

    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

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
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                seq_len_prefix,
                qk_scale,
                LOGIT_CAP,
                xai_temperature_reg,
                XAI_TEMPERATURE_LEN,
                q_abs_pos,
                SLIDING_WINDOW_SIZE,
                Mask,
                mask_base_idx,
                mask_row_stride,
                q_extend_offs,
                USE_CUSTOM_MASK,
                SKIP_PREFIX_CUSTOM_MASK,
                ENABLE_PREFIX_UNMASKED,
                BLOCK_M,
                BLOCK_N,
                mma_layout,
                mma_offs_n_col,
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
            nf_start_n = ((block_n + NUM_STAGES + 1) * BLOCK_N).to(tl.int32)
            n_idx_kt_nf = nf_start_n + kt_offs_n_pf
            mask_n_kt_pf = n_idx_kt_nf < seq_len_prefix
            kv_locs_kt_pf = gl.load(
                kv_indices + kv_start + n_idx_kt_nf, mask=mask_n_kt_pf, other=0
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
            n_idx_v_nf = nf_start_n + v_offs_n_pf
            mask_n_v_pf = n_idx_v_nf < seq_len_prefix
            kv_locs_v_pf = gl.load(
                kv_indices + kv_start + n_idx_v_nf, mask=mask_n_v_pf, other=0
            ).to(tl.int32)

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
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - STREAMS)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_prefix_pipelined_scalar_mask(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    kpe_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
    ASYNC_KPE: gl.constexpr = False,
):
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    USE_ASYNC_KPE: gl.constexpr = HAS_DPE and ASYNC_KPE
    STREAMS: gl.constexpr = 3 if USE_ASYNC_KPE else 2
    # warp_pipeline_stage requires >=2 physical LDS buffers. See
    # attn_fwd_inner_prefix_pipelined docstring above for the full race
    # analysis. Callers wanting NS=1 must route to attn_fwd_inner_prefix_dma_simple.
    tl.static_assert(
        NUM_STAGES >= 2,
        "attn_fwd_inner_prefix_pipelined_scalar_mask requires NUM_STAGES>=2 "
        "for determinism (warp_pipeline_stage needs multiple LDS buffers).",
    )

    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh

    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    kt_offs_n_pf = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    v_offs_n_pf = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)

    for stage in gl.static_range(NUM_STAGES):
        pf_init_n = stage * BLOCK_N
        issue_async_load_k_prefix(
            kt_smem.index(stage),
            k_prefix_base,
            kv_indices,
            kv_start,
            pf_init_n,
            seq_len_prefix,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_prefix(
                kpe_smem.index(stage),
                k_prefix_base,
                kv_indices,
                kv_start,
                pf_init_n,
                seq_len_prefix,
                stride_buf_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DPE,
                ACTUAL_BLOCK_DPE,
                kpe_async_layout,
            )
        issue_async_load_v_prefix(
            v_smem.index(stage),
            v_prefix_base,
            kv_indices,
            kv_start,
            pf_init_n,
            seq_len_prefix,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DV,
            ACTUAL_BLOCK_DV,
            v_async_layout,
        )

    pf_start_n = NUM_STAGES * BLOCK_N
    n_idx_kt_pf = pf_start_n + kt_offs_n_pf
    mask_n_kt_pf = n_idx_kt_pf < seq_len_prefix
    kv_locs_kt_pf = gl.load(
        kv_indices + kv_start + n_idx_kt_pf, mask=mask_n_kt_pf, other=0
    ).to(tl.int32)
    n_idx_v_pf = pf_start_n + v_offs_n_pf
    mask_n_v_pf = n_idx_v_pf < seq_len_prefix
    kv_locs_v_pf = gl.load(
        kv_indices + kv_start + n_idx_v_pf, mask=mask_n_v_pf, other=0
    ).to(tl.int32)

    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    dma_mask = seq_len_prefix > 0

    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

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
            issue_dma_k_prefix_from_locs_hot(
                kt_smem.index(stage_idx),
                k_prefix_base,
                kv_locs_kt_pf,
                dma_mask,
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
                acc,
                l_i,
                m_i,
                qk,
                start_n,
                seq_len_prefix,
                qk_scale,
                LOGIT_CAP,
                xai_temperature_reg,
                XAI_TEMPERATURE_LEN,
                q_abs_pos,
                SLIDING_WINDOW_SIZE,
                Mask,
                mask_base_idx,
                mask_row_stride,
                q_extend_offs,
                USE_CUSTOM_MASK,
                SKIP_PREFIX_CUSTOM_MASK,
                ENABLE_PREFIX_UNMASKED,
                BLOCK_M,
                BLOCK_N,
                mma_layout,
                mma_offs_n_col,
            )

        with warp_pipeline_stage("memory1", priority=1):
            issue_dma_v_prefix_from_locs_hot(
                v_smem.index(stage_idx),
                v_prefix_base,
                kv_locs_v_pf,
                dma_mask,
                stride_buf_vbs,
                BLOCK_N,
                BLOCK_DV,
                ACTUAL_BLOCK_DV,
                v_async_layout,
            )
            nf_start_n = ((block_n + NUM_STAGES + 1) * BLOCK_N).to(tl.int32)
            n_idx_kt_nf = nf_start_n + kt_offs_n_pf
            mask_n_kt_pf = n_idx_kt_nf < seq_len_prefix
            kv_locs_kt_pf = gl.load(
                kv_indices + kv_start + n_idx_kt_nf, mask=mask_n_kt_pf, other=0
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
            n_idx_v_nf = nf_start_n + v_offs_n_pf
            mask_n_v_pf = n_idx_v_nf < seq_len_prefix
            kv_locs_v_pf = gl.load(
                kv_indices + kv_start + n_idx_v_nf, mask=mask_n_v_pf, other=0
            ).to(tl.int32)

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
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - STREAMS)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_prefix_short(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
    V_PRELOAD: gl.constexpr = False,  #
    block_start=0,
):
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh

    for block_n in tl.range(block_start, n_prefix_blocks):
        start_n = block_n * BLOCK_N

        issue_async_load_k_prefix(
            kt_smem.index(0),
            k_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        issue_async_load_v_prefix(
            v_smem.index(0),
            v_prefix_base,
            kv_indices,
            kv_start,
            start_n,
            seq_len_prefix,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DV,
            ACTUAL_BLOCK_DV,
            v_async_layout,
        )

        if V_PRELOAD:
            cdna4_async.wait_group(0)
        else:
            cdna4_async.wait_group(1)
        kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)
        if V_PRELOAD:
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)
        if BLOCK_DPE > 0:
            qk = add_qk_dpe_prefix_from_kv(
                qk,
                qpe_dot,
                K_Buffer,
                kv_indices,
                kv_start,
                start_n,
                cur_kv_head,
                seq_len_prefix,
                stride_buf_kbs,
                stride_buf_kh,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DPE,
                ACTUAL_BLOCK_DPE,
                kt_dot_layout,
            )

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        if not V_PRELOAD:
            cdna4_async.wait_group(0)
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(0), v_dot_layout)
        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_prefix_dma_simple(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    cur_kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_smem,
    kpe_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
    ASYNC_KPE: gl.constexpr = True,
):
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    USE_ASYNC_KPE: gl.constexpr = HAS_DPE and ASYNC_KPE
    STREAMS: gl.constexpr = 3 if USE_ASYNC_KPE else 2

    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N
    k_prefix_base = K_Buffer + cur_kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + cur_kv_head * stride_buf_vh
    kv_indices_base = kv_indices + kv_start

    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)

    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - STREAMS

    for stage in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(0)
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
            kt_smem.index(stage),
            k_prefix_base,
            kv_locs_k,
            mask_n_k,
            stride_buf_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_prefix(
                kpe_smem.index(stage),
                k_prefix_base,
                kv_indices,
                kv_start,
                pf_init_n,
                seq_len_prefix,
                stride_buf_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DPE,
                ACTUAL_BLOCK_DPE,
                kpe_async_layout,
            )
        issue_dma_v_prefix_from_locs(
            v_smem.index(stage),
            v_prefix_base,
            kv_locs_v,
            mask_n_v,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DV,
            ACTUAL_BLOCK_DV,
            v_async_layout,
        )

    cdna4_async.wait_group(0)
    pf_start_n = NUM_STAGES * BLOCK_N
    n_idx_k_pf = pf_start_n + kt_offs_n
    mask_k_pf = n_idx_k_pf < seq_len_prefix
    kv_locs_k_pf = cdna4_buffer_load(
        kv_indices_base, n_idx_k_pf.to(tl.int32), mask=mask_k_pf, other=0
    ).to(tl.int32)
    n_idx_v_pf = pf_start_n + v_offs_n
    mask_v_pf = n_idx_v_pf < seq_len_prefix
    kv_locs_v_pf = cdna4_buffer_load(
        kv_indices_base, n_idx_v_pf.to(tl.int32), mask=mask_v_pf, other=0
    ).to(tl.int32)

    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = n_prefix_blocks - NUM_STAGES
    for block_n in tl.range(0, main_loop_end):
        stage_idx = (block_n % NUM_STAGES).to(tl.int32)
        start_n = (block_n * BLOCK_N).to(tl.int32)

        nf_start_n = ((block_n + NUM_STAGES + 1) * BLOCK_N).to(tl.int32)
        n_idx_k_nf = nf_start_n + kt_offs_n
        mask_k_nf = n_idx_k_nf < seq_len_prefix
        kv_locs_k_nf = cdna4_buffer_load(
            kv_indices_base, n_idx_k_nf.to(tl.int32), mask=mask_k_nf, other=0
        ).to(tl.int32)
        n_idx_v_nf = nf_start_n + v_offs_n
        mask_v_nf = n_idx_v_nf < seq_len_prefix
        kv_locs_v_nf = cdna4_buffer_load(
            kv_indices_base, n_idx_v_nf.to(tl.int32), mask=mask_v_nf, other=0
        ).to(tl.int32)

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
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(stage_idx), v_dot_layout)

        # NOTE: issue K_future AFTER v_dot read; writing back into kt_smem slot
        # after the load avoids the DMA race that was observable once the
        # ``_buffer_load_to_shared_cast_safe`` dtype check was fixed (d1a61f1).
        issue_dma_k_prefix_from_locs(
            kt_smem.index(stage_idx),
            k_prefix_base,
            kv_locs_k_pf,
            mask_k_pf,
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

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        p_cast = p.to(v_dot.dtype)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_reg, v_dot, acc)

        next_stage_idx = ((block_n + 1) % NUM_STAGES).to(tl.int32)
        cdna4_async.wait_group(WAIT_K)
        kt_dot = cdna4_async.load_shared_relaxed(
            kt_smem.index(next_stage_idx), kt_dot_layout
        )

        # V_future must be issued after the p*V MMA has consumed v_dot and
        # after the next kt_dot has been read.
        issue_dma_v_prefix_from_locs(
            v_smem.index(stage_idx),
            v_prefix_base,
            kv_locs_v_pf,
            mask_v_pf,
            stride_buf_vbs,
            BLOCK_N,
            BLOCK_DV,
            ACTUAL_BLOCK_DV,
            v_async_layout,
        )

        kv_locs_k_pf = kv_locs_k_nf
        kv_locs_v_pf = kv_locs_v_nf
        mask_k_pf = mask_k_nf
        mask_v_pf = mask_v_nf

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
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - STREAMS)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Extend inner loops
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_extend_dma(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    kpe_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
    ASYNC_KPE: gl.constexpr = True,
    SKIP_BOUNDS_CHECK: gl.constexpr = False,
):
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    USE_ASYNC_KPE: gl.constexpr = HAS_DPE and ASYNC_KPE
    STREAMS: gl.constexpr = 3 if USE_ASYNC_KPE else 2
    cdna4_async.wait_group(0)

    for stage in gl.static_range(NUM_STAGES):
        pf_start_n = (block_start + stage) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(stage),
            k_base,
            pf_start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
            SKIP_BOUNDS_CHECK,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_extend(
                kpe_smem.index(stage),
                k_base,
                pf_start_n,
                seq_len_extend,
                stride_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DPE,
                ACTUAL_BLOCK_DPE,
                kpe_async_layout,
            )
        issue_async_load_v_extend(
            v_smem.index(stage),
            v_base,
            pf_start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DV,
            ACTUAL_BLOCK_DV,
            v_async_layout,
            SKIP_BOUNDS_CHECK,
        )

    WAIT_K: gl.constexpr = STREAMS * NUM_STAGES - 1
    WAIT_V: gl.constexpr = STREAMS * NUM_STAGES - 2
    cdna4_async.wait_group(WAIT_K)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = block_end - NUM_STAGES
    for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=2):
        stage_idx = ((block_n - block_start) % NUM_STAGES).to(tl.int32)
        start_n = (block_n * BLOCK_N).to(tl.int32)
        future_start_n = ((block_n + NUM_STAGES) * BLOCK_N).to(tl.int32)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)
        if USE_ASYNC_KPE:
            qk = add_qk_dpe_from_shared(
                qk, qpe_dot, kpe_smem.index(stage_idx), kt_dot_layout,
                BLOCK_DPE,
            )
        elif HAS_DPE:
            qk = add_qk_dpe_extend_from_base(
                qk, qpe_dot, k_base, start_n, seq_len_extend, stride_kbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                kt_dot_layout,
            )

        cdna4_async.wait_group(WAIT_V)
        v_dot = cdna4_async.load_shared_relaxed(v_smem.index(stage_idx), v_dot_layout)
        issue_async_load_k_extend(
            kt_smem.index(stage_idx),
            k_base,
            future_start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
            SKIP_BOUNDS_CHECK,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_extend(
                kpe_smem.index(stage_idx),
                k_base,
                future_start_n,
                seq_len_extend,
                stride_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DPE,
                ACTUAL_BLOCK_DPE,
                kpe_async_layout,
            )

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
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
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )
        p_cast = p.to(v_dot.dtype)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_V)
        next_stage_idx = ((block_n + 1 - block_start) % NUM_STAGES).to(tl.int32)
        kt_dot = cdna4_async.load_shared_relaxed(
            kt_smem.index(next_stage_idx), kt_dot_layout
        )
        issue_async_load_v_extend(
            v_smem.index(stage_idx),
            v_base,
            future_start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DV,
            ACTUAL_BLOCK_DV,
            v_async_layout,
            SKIP_BOUNDS_CHECK,
        )

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - 1)
        stage_idx = ((main_loop_end + tail_i - block_start) % NUM_STAGES).to(tl.int32)
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
            qk = add_qk_dpe_extend_from_base(
                qk, qpe_dot, k_base, start_n, seq_len_extend, stride_kbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                kt_dot_layout,
            )

        p, alpha, m_new = compute_softmax_extend_part0(
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
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
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - 2)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        acc, l_i, m_i = compute_softmax_extend_part1(acc, l_i, p, alpha, m_new)
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_extend_pipelined(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    kpe_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
    ASYNC_KPE: gl.constexpr = False,
    SKIP_BOUNDS_CHECK: gl.constexpr = False,
):
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    USE_ASYNC_KPE: gl.constexpr = HAS_DPE and ASYNC_KPE
    STREAMS: gl.constexpr = 3 if USE_ASYNC_KPE else 2
    # warp_pipeline_stage requires >=2 physical LDS buffers. With NS=1 the
    # stage_idx collapses to 0, and iter N's DMA write to smem[0] races
    # iter N+1's relaxed read from smem[0] because membarFilter skips
    # barriers between BufferLoadToLocalOp and syncedViaAsyncWait loads.
    # Callers wanting NS=1 must route to the non-pipelined DMA path
    # (attn_fwd_inner_extend_dma).
    tl.static_assert(
        NUM_STAGES >= 2,
        "attn_fwd_inner_extend_pipelined requires NUM_STAGES>=2 for "
        "determinism (warp_pipeline_stage needs multiple LDS buffers).",
    )
    cdna4_async.wait_group(0)

    for stage in gl.static_range(NUM_STAGES):
        pf_start_n = (block_start + stage) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(stage),
            k_base,
            pf_start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
            SKIP_BOUNDS_CHECK,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_extend(
                kpe_smem.index(stage),
                k_base,
                pf_start_n,
                seq_len_extend,
                stride_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DPE,
                ACTUAL_BLOCK_DPE,
                kpe_async_layout,
            )
        issue_async_load_v_extend(
            v_smem.index(stage),
            v_base,
            pf_start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DV,
            ACTUAL_BLOCK_DV,
            v_async_layout,
            SKIP_BOUNDS_CHECK,
        )

    WAIT_INIT: gl.constexpr = STREAMS * NUM_STAGES - (STREAMS - 1)
    WAIT_LOOP: gl.constexpr = STREAMS * NUM_STAGES - STREAMS
    cdna4_async.wait_group(WAIT_INIT)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = block_end - NUM_STAGES
    for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=2):

        with warp_pipeline_stage("dot1", priority=0):
            stage_idx = ((block_n - block_start) % NUM_STAGES).to(tl.int32)
            start_n = (block_n * BLOCK_N).to(tl.int32)
            future_start_n = ((block_n + NUM_STAGES) * BLOCK_N).to(tl.int32)
            qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
            qk = do_mma(q_dot, kt_dot, qk)
            if USE_ASYNC_KPE:
                qk = add_qk_dpe_from_shared(
                    qk, qpe_dot, kpe_smem.index(stage_idx), kt_dot_layout,
                    BLOCK_DPE,
                )
            elif HAS_DPE:
                qk = add_qk_dpe_extend_from_base(
                    qk, qpe_dot, k_base, start_n, seq_len_extend, stride_kbs,
                    BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                    kt_dot_layout,
                )
            p, alpha, m_new = compute_softmax_extend_part0(
                m_i,
                qk,
                start_n,
                cur_block_m,
                seq_len_extend,
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
                MASK_STEPS,
                BLOCK_M,
                BLOCK_N,
                mma_layout,
                mma_offs_n_col,
                mma_offs_m_row,
            )

        cdna4_async.wait_group(WAIT_LOOP)

        with warp_pipeline_stage("mem1", priority=1):
            v_dot = cdna4_async.load_shared_relaxed(
                v_smem.index(stage_idx), v_dot_layout
            )
            issue_async_load_k_extend(
                kt_smem.index(stage_idx),
                k_base,
                future_start_n,
                seq_len_extend,
                stride_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL,
                kt_async_layout,
                SKIP_BOUNDS_CHECK,
            )
            if USE_ASYNC_KPE:
                issue_async_load_kpe_extend(
                    kpe_smem.index(stage_idx),
                    k_base,
                    future_start_n,
                    seq_len_extend,
                    stride_kbs,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DPE,
                    ACTUAL_BLOCK_DPE,
                    kpe_async_layout,
                )

        with warp_pipeline_stage("dot2a", priority=0):
            acc, l_i, m_i = compute_softmax_extend_part1(
                acc, l_i, p, alpha, m_new
            )

        with warp_pipeline_stage("dot2b", priority=0):
            p_cast = p.to(v_dot.dtype)
            p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
            acc = do_mma(p_dot_reg, v_dot, acc)

        cdna4_async.wait_group(WAIT_LOOP)

        with warp_pipeline_stage("mem2", priority=1):
            next_stage_idx = ((block_n + 1 - block_start) % NUM_STAGES).to(tl.int32)
            kt_dot = cdna4_async.load_shared_relaxed(
                kt_smem.index(next_stage_idx), kt_dot_layout
            )
            issue_async_load_v_extend(
                v_smem.index(stage_idx),
                v_base,
                future_start_n,
                seq_len_extend,
                stride_vbs,
                BLOCK_N,
                BLOCK_DV,
                ACTUAL_BLOCK_DV,
                v_async_layout,
                SKIP_BOUNDS_CHECK,
            )

    for tail_i in gl.static_range(NUM_STAGES):
        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - (STREAMS - 1))
        stage_idx = ((main_loop_end + tail_i - block_start) % NUM_STAGES).to(tl.int32)
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
            qk = add_qk_dpe_extend_from_base(
                qk, qpe_dot, k_base, start_n, seq_len_extend, stride_kbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                kt_dot_layout,
            )

        p, alpha, m_new = compute_softmax_extend_part0(
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
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
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        cdna4_async.wait_group(STREAMS * (NUM_STAGES - tail_i) - STREAMS)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(stage_idx), v_dot_layout
        )
        acc, l_i, m_i = compute_softmax_extend_part1(acc, l_i, p, alpha, m_new)
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_extend_pipelined_n2(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    kpe_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    NUM_STAGES: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    kpe_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
    ASYNC_KPE: gl.constexpr = False,
):
    """N-subtiled variant: processes 2 BLOCK_N sub-tiles per outer iteration.

    Requires kt_smem/v_smem with 2*NUM_STAGES physical buffers, but
    QK/PV accumulators stay at BLOCK_N width (no register pressure increase).
    block_end - block_start must be >= 2*NUM_STAGES.
    """
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    USE_ASYNC_KPE: gl.constexpr = HAS_DPE and ASYNC_KPE
    STREAMS: gl.constexpr = 3 if USE_ASYNC_KPE else 2
    PHYS: gl.constexpr = 2 * NUM_STAGES
    cdna4_async.wait_group(0)

    for stage in gl.static_range(PHYS):
        pf_start_n = (block_start + stage) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(stage), k_base, pf_start_n, seq_len_extend,
            stride_kbs, BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_extend(
                kpe_smem.index(stage), k_base, pf_start_n, seq_len_extend,
                stride_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                kpe_async_layout,
            )
        issue_async_load_v_extend(
            v_smem.index(stage), v_base, pf_start_n, seq_len_extend,
            stride_vbs, BLOCK_N, BLOCK_DV, ACTUAL_BLOCK_DV, v_async_layout,
        )

    WAIT_INIT: gl.constexpr = STREAMS * PHYS - (STREAMS - 1)
    WAIT_LOOP: gl.constexpr = STREAMS * PHYS - STREAMS
    cdna4_async.wait_group(WAIT_INIT)
    kt_dot = cdna4_async.load_shared_relaxed(kt_smem.index(0), kt_dot_layout)

    main_loop_end = block_end - PHYS
    for block_n in tl.range(block_start, main_loop_end, 2):
        for sub in gl.static_range(2):
            actual_block = block_n + sub
            phys_idx = ((actual_block - block_start) % PHYS).to(tl.int32)
            start_n = (actual_block * BLOCK_N).to(tl.int32)
            future_start_n = ((actual_block + PHYS) * BLOCK_N).to(tl.int32)

            with warp_pipeline_stage("dot1", priority=0):
                qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
                qk = do_mma(q_dot, kt_dot, qk)
                if USE_ASYNC_KPE:
                    qk = add_qk_dpe_from_shared(
                        qk, qpe_dot, kpe_smem.index(phys_idx), kt_dot_layout,
                        BLOCK_DPE,
                    )
                elif HAS_DPE:
                    qk = add_qk_dpe_extend_from_base(
                        qk, qpe_dot, k_base, start_n, seq_len_extend, stride_kbs,
                        BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                        kt_dot_layout,
                    )
                p, alpha, m_new = compute_softmax_extend_part0(
                    m_i, qk, start_n, cur_block_m, seq_len_extend,
                    qk_scale, LOGIT_CAP, xai_temperature_reg, XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE, IS_CAUSAL,
                    Mask, mask_base_idx, mask_row_stride, mask_kv_col_offset,
                    USE_CUSTOM_MASK, MASK_STEPS,
                    BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
                )

            cdna4_async.wait_group(WAIT_LOOP)

            with warp_pipeline_stage("mem1", priority=1):
                v_dot = cdna4_async.load_shared_relaxed(
                    v_smem.index(phys_idx), v_dot_layout
                )
                issue_async_load_k_extend(
                    kt_smem.index(phys_idx), k_base, future_start_n,
                    seq_len_extend, stride_kbs,
                    BLOCK_N, BLOCK_DMODEL, ACTUAL_BLOCK_DMODEL, kt_async_layout,
                )
                if USE_ASYNC_KPE:
                    issue_async_load_kpe_extend(
                        kpe_smem.index(phys_idx), k_base, future_start_n,
                        seq_len_extend, stride_kbs,
                        BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                        kpe_async_layout,
                    )

            with warp_pipeline_stage("dot2a", priority=0):
                acc, l_i, m_i = compute_softmax_extend_part1(
                    acc, l_i, p, alpha, m_new
                )

            with warp_pipeline_stage("dot2b", priority=0):
                p_cast = p.to(v_dot.dtype)
                p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
                acc = do_mma(p_dot_reg, v_dot, acc)

            cdna4_async.wait_group(WAIT_LOOP)

            with warp_pipeline_stage("mem2", priority=1):
                next_phys = ((actual_block + 1 - block_start) % PHYS).to(tl.int32)
                kt_dot = cdna4_async.load_shared_relaxed(
                    kt_smem.index(next_phys), kt_dot_layout
                )
                issue_async_load_v_extend(
                    v_smem.index(phys_idx), v_base, future_start_n,
                    seq_len_extend, stride_vbs,
                    BLOCK_N, BLOCK_DV, ACTUAL_BLOCK_DV, v_async_layout,
                )

    for tail_i in gl.static_range(PHYS):
        cdna4_async.wait_group(STREAMS * (PHYS - tail_i) - (STREAMS - 1))
        phys_idx = ((main_loop_end + tail_i - block_start) % PHYS).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        kt_dot_tail = cdna4_async.load_shared_relaxed(
            kt_smem.index(phys_idx), kt_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot_tail, qk)
        if USE_ASYNC_KPE:
            qk = add_qk_dpe_from_shared(
                qk, qpe_dot, kpe_smem.index(phys_idx), kt_dot_layout,
                BLOCK_DPE,
            )
        elif HAS_DPE:
            qk = add_qk_dpe_extend_from_base(
                qk, qpe_dot, k_base, start_n, seq_len_extend, stride_kbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                kt_dot_layout,
            )

        p, alpha, m_new = compute_softmax_extend_part0(
            m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, xai_temperature_reg, XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE, IS_CAUSAL,
            Mask, mask_base_idx, mask_row_stride, mask_kv_col_offset,
            USE_CUSTOM_MASK, MASK_STEPS,
            BLOCK_M, BLOCK_N, mma_layout, mma_offs_n_col, mma_offs_m_row,
        )

        cdna4_async.wait_group(STREAMS * (PHYS - tail_i) - STREAMS)
        v_dot_tail = cdna4_async.load_shared_relaxed(
            v_smem.index(phys_idx), v_dot_layout
        )
        acc, l_i, m_i = compute_softmax_extend_part1(acc, l_i, p, alpha, m_new)
        p_cast = p.to(v_dot_tail.dtype)
        p_dot_tail = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot_tail, v_dot_tail, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Serial inner loops (4-warp path)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def attn_fwd_inner_prefix_serial(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    K_Buffer,
    V_Buffer,
    kv_indices,  #
    kv_start,
    kv_head,
    seq_len_prefix,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    kt_serial_smem,
    kt_dpe_serial_smem,
    v_serial_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    q_abs_pos,
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    q_extend_offs,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    kt_blocked_layout: gl.constexpr,
    blocked_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,  #
    V_PRELOAD: gl.constexpr = False,  #
):
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    kt_offs_d = gl.arange(
        0, BLOCK_DMODEL, layout=gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    )
    if HAS_DPE:
        kt_dpe_offs_d = gl.arange(
            0, BLOCK_DPE, layout=gl.SliceLayout(dim=1, parent=kt_blocked_layout)
        )
    kt_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    )
    v_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=blocked_layout)
    )
    v_offs_d = gl.arange(0, BLOCK_DV, layout=gl.SliceLayout(dim=0, parent=blocked_layout))

    k_prefix_base = K_Buffer + kv_head * stride_buf_kh
    v_prefix_base = V_Buffer + kv_head * stride_buf_vh
    n_prefix_blocks = (seq_len_prefix + BLOCK_N - 1) // BLOCK_N

    for block_n in tl.range(0, n_prefix_blocks):
        start_n = block_n * BLOCK_N

        n_idx_k = start_n + kt_offs_n
        mask_n_k = n_idx_k < seq_len_prefix
        kv_locs_k = gl.load(kv_indices + kv_start + n_idx_k, mask=mask_n_k, other=0).to(
            tl.int32
        )

        kt_ptrs = (
            k_prefix_base + kt_offs_d[:, None] + kv_locs_k[None, :] * stride_buf_kbs
        )
        kt_mask = mask_n_k[None, :]
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
        kt_global = gl.load(kt_ptrs, mask=kt_mask, other=0.0)
        kt_serial_smem.store(kt_global)
        kt_dot = kt_serial_smem.load(kt_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)
        if BLOCK_DPE > 0:
            kt_dpe_ptrs = (
                k_prefix_base
                + BLOCK_DMODEL
                + kt_dpe_offs_d[:, None]
                + kv_locs_k[None, :] * stride_buf_kbs
            )
            kt_dpe_mask = mask_n_k[None, :]
            if ACTUAL_BLOCK_DPE != BLOCK_DPE:
                kt_dpe_mask = kt_dpe_mask & (kt_dpe_offs_d[:, None] < ACTUAL_BLOCK_DPE)
            kt_dpe_global = gl.load(kt_dpe_ptrs, mask=kt_dpe_mask, other=0.0)
            kt_dpe_serial_smem.store(kt_dpe_global)
            kt_dpe_dot = kt_dpe_serial_smem.load(kt_dot_layout)
            qk = do_mma(qpe_dot, kt_dpe_dot, qk)

        if V_PRELOAD:
            n_idx_v = start_n + v_offs_n
            mask_n_v = n_idx_v < seq_len_prefix
            kv_locs_v = gl.load(
                kv_indices + kv_start + n_idx_v, mask=mask_n_v, other=0
            ).to(tl.int32)
            v_ptrs = (
                v_prefix_base + kv_locs_v[:, None] * stride_buf_vbs + v_offs_d[None, :]
            )
            v_mask = mask_n_v[:, None]
            if ACTUAL_BLOCK_DV != BLOCK_DV:
                v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DV)
            v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)

        acc, l_i, m_i, p = compute_softmax_prefix(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            seq_len_prefix,
            qk_scale,
            LOGIT_CAP,
            xai_temperature_reg,
            XAI_TEMPERATURE_LEN,
            q_abs_pos,
            SLIDING_WINDOW_SIZE,
            Mask,
            mask_base_idx,
            mask_row_stride,
            q_extend_offs,
            USE_CUSTOM_MASK,
            SKIP_PREFIX_CUSTOM_MASK,
            ENABLE_PREFIX_UNMASKED,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
        )

        if not V_PRELOAD:
            n_idx_v = start_n + v_offs_n
            mask_n_v = n_idx_v < seq_len_prefix
            kv_locs_v = gl.load(
                kv_indices + kv_start + n_idx_v, mask=mask_n_v, other=0
            ).to(tl.int32)
            v_ptrs = (
                v_prefix_base + kv_locs_v[:, None] * stride_buf_vbs + v_offs_d[None, :]
            )
            v_mask = mask_n_v[:, None]
            if ACTUAL_BLOCK_DV != BLOCK_DV:
                v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DV)
            v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
        v_serial_smem.store(v_global)
        v_dot = v_serial_smem.load(v_dot_layout)

        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


@gluon.jit
def attn_fwd_inner_extend_serial(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    k_extend_base,
    v_extend_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_serial_smem,
    kt_dpe_serial_smem,
    v_serial_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    kt_blocked_layout: gl.constexpr,
    blocked_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
    V_PRELOAD: gl.constexpr = False,  #
):
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    kt_offs_d = gl.arange(
        0, BLOCK_DMODEL, layout=gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    )
    if HAS_DPE:
        kt_dpe_offs_d = gl.arange(
            0, BLOCK_DPE, layout=gl.SliceLayout(dim=1, parent=kt_blocked_layout)
        )
    kt_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    )
    v_offs_n = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=blocked_layout)
    )
    v_offs_d = gl.arange(0, BLOCK_DV, layout=gl.SliceLayout(dim=0, parent=blocked_layout))

    for block_n in tl.range(block_start, block_end):
        start_n = block_n * BLOCK_N

        kt_ptrs = (
            k_extend_base
            + kt_offs_d[:, None]
            + (start_n + kt_offs_n[None, :]) * stride_kbs
        )
        kt_mask = (start_n + kt_offs_n[None, :]) < seq_len_extend
        if ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL:
            kt_mask = kt_mask & (kt_offs_d[:, None] < ACTUAL_BLOCK_DMODEL)
        kt_global = gl.load(kt_ptrs, mask=kt_mask, other=0.0)
        kt_serial_smem.store(kt_global)
        kt_dot = kt_serial_smem.load(kt_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)
        if BLOCK_DPE > 0:
            kt_dpe_ptrs = (
                k_extend_base
                + BLOCK_DMODEL
                + kt_dpe_offs_d[:, None]
                + (start_n + kt_offs_n[None, :]) * stride_kbs
            )
            kt_dpe_mask = (start_n + kt_offs_n[None, :]) < seq_len_extend
            if ACTUAL_BLOCK_DPE != BLOCK_DPE:
                kt_dpe_mask = kt_dpe_mask & (kt_dpe_offs_d[:, None] < ACTUAL_BLOCK_DPE)
            kt_dpe_global = gl.load(kt_dpe_ptrs, mask=kt_dpe_mask, other=0.0)
            kt_dpe_serial_smem.store(kt_dpe_global)
            kt_dpe_dot = kt_dpe_serial_smem.load(kt_dot_layout)
            qk = do_mma(qpe_dot, kt_dpe_dot, qk)

        if V_PRELOAD:
            v_ptrs = (
                v_extend_base
                + (start_n + v_offs_n[:, None]) * stride_vbs
                + v_offs_d[None, :]
            )
            v_mask = (start_n + v_offs_n[:, None]) < seq_len_extend
            if ACTUAL_BLOCK_DV != BLOCK_DV:
                v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DV)
            v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
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
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        if not V_PRELOAD:
            v_ptrs = (
                v_extend_base
                + (start_n + v_offs_n[:, None]) * stride_vbs
                + v_offs_d[None, :]
            )
            v_mask = (start_n + v_offs_n[:, None]) < seq_len_extend
            if ACTUAL_BLOCK_DV != BLOCK_DV:
                v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DV)
            v_global = gl.load(v_ptrs, mask=v_mask, other=0.0)
        v_serial_smem.store(v_global)
        v_dot = v_serial_smem.load(v_dot_layout)

        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i


# ===-----------------------------------------------------------------------===#
# Extend short path (preload-all, defined in this file)
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# D-dimension subtiled extend inner loop
# Splits K^T [BLOCK_DMODEL, BLOCK_N] and V [BLOCK_N, BLOCK_DV] along the
# model dimension into two halves (each 256-wide). This halves LDS per tile,
# allowing BLOCK_N=64 with double-buffering to fit within 160KB on MI350X.
#
# Smem layout: kt_lo/kt_hi/v_lo/v_hi each [NUM_STAGES, half_d, BLOCK_N].
# Stage 0 and 1 double-buffer across N-blocks. Within each N-block:
#   QK = Q_lo @ K_lo^T + Q_hi @ K_hi^T (+ Q_dpe @ K_dpe^T)
#   acc_lo += P @ V_lo,  acc_hi += P @ V_hi
# ===-----------------------------------------------------------------------===#


@gluon.jit
def issue_async_load_k_extend_half(
    kt_smem,
    k_base,
    start_n,
    seq_len_extend,
    stride_kbs,
    d_offset,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL_HALF: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    kt_async_layout: gl.constexpr,
):
    """Load one half of K^T [BLOCK_DMODEL_HALF, BLOCK_N] starting at d_offset."""
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_async_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_async_layout)
    kt_offs_d = gl.arange(0, BLOCK_DMODEL_HALF, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    kt_offsets = (
        (kt_offs_d[:, None] + d_offset)
        + (start_n + kt_offs_n[None, :]) * stride_kbs
    ).to(tl.int32)
    kt_mask = (start_n + kt_offs_n[None, :]) < seq_len_extend
    if d_offset + BLOCK_DMODEL_HALF > ACTUAL_BLOCK_DMODEL:
        kt_mask = kt_mask & ((kt_offs_d[:, None] + d_offset) < ACTUAL_BLOCK_DMODEL)
    _buffer_load_to_shared_cast_safe(
        kt_smem, k_base, kt_offsets, mask=kt_mask, other=0.0
    )


@gluon.jit
def issue_async_load_v_extend_half(
    v_smem,
    v_base,
    start_n,
    seq_len_extend,
    stride_vbs,
    d_offset,
    BLOCK_N: gl.constexpr,
    BLOCK_DV_HALF: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,
    v_async_layout: gl.constexpr,
):
    """Load one half of V [BLOCK_N, BLOCK_DV_HALF] starting at d_offset."""
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=v_async_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=v_async_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DV_HALF, layout=v_offs_d_layout)
    v_offsets = (
        (start_n + v_offs_n[:, None]) * stride_vbs + v_offs_d[None, :] + d_offset
    ).to(tl.int32)
    v_mask = (start_n + v_offs_n[:, None]) < seq_len_extend
    if d_offset + BLOCK_DV_HALF > ACTUAL_BLOCK_DV:
        v_mask = v_mask & ((v_offs_d[None, :] + d_offset) < ACTUAL_BLOCK_DV)
    _buffer_load_to_shared_cast_safe(
        v_smem, v_base, v_offsets, mask=v_mask, other=0.0
    )


@gluon.jit
def attn_fwd_inner_extend_subtiled(
    acc_lo,
    acc_hi,
    l_i,
    m_i,
    q_dot_lo,
    q_dot_hi,
    qpe_dot,
    k_base,
    v_base,
    cur_block_m,
    seq_len_extend,
    stride_kbs,
    stride_vbs,
    block_start,
    block_end,
    kt_half_smem,
    kpe_smem,
    v_half_smem,
    qk_scale,
    LOGIT_CAP: gl.constexpr,
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,
    SLIDING_WINDOW_SIZE: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,
    USE_CUSTOM_MASK: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    kt_half_async_layout: gl.constexpr,
    kpe_async_layout: gl.constexpr,
    v_half_async_layout: gl.constexpr,
    kt_half_dot_layout: gl.constexpr,
    kpe_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_half_dot_layout: gl.constexpr,
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,
    ASYNC_KPE: gl.constexpr = True,
):
    """D-subtiled extend inner loop for D512 on MI350X.

    K^T and V are each split into two 256-wide halves along d-dimension.
    A single kt_half_smem and v_half_smem are shared between lo/hi halves.
    Within each N-block, halves are loaded and consumed sequentially:
      K_lo → QK_lo, K_hi → QK_hi, DPE, softmax, V_lo → PV_lo, V_hi → PV_hi.
    Across N-blocks, K_lo of the next block is prefetched while computing
    the current block (using NUM_STAGES pipeline slots).
    """
    HAS_DPE: gl.constexpr = BLOCK_DPE > 0
    USE_ASYNC_KPE: gl.constexpr = HAS_DPE and ASYNC_KPE
    BLOCK_DMODEL_HALF: gl.constexpr = BLOCK_DMODEL // 2
    BLOCK_DV_HALF: gl.constexpr = BLOCK_DV // 2

    cdna4_async.wait_group(0)

    # --- Prologue: prefetch K_lo + KPE for first NUM_STAGES blocks ---
    for stage in gl.static_range(NUM_STAGES):
        pf_n = (block_start + stage) * BLOCK_N
        issue_async_load_k_extend_half(
            kt_half_smem.index(stage), k_base, pf_n, seq_len_extend,
            stride_kbs, 0, BLOCK_N, BLOCK_DMODEL_HALF,
            ACTUAL_BLOCK_DMODEL, kt_half_async_layout,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_extend(
                kpe_smem.index(stage), k_base, pf_n, seq_len_extend,
                stride_kbs, BLOCK_N, BLOCK_DMODEL, BLOCK_DPE,
                ACTUAL_BLOCK_DPE, kpe_async_layout,
            )

    STREAMS_PREFETCH: gl.constexpr = 2 if USE_ASYNC_KPE else 1

    # --- Main loop ---
    main_loop_end = block_end - NUM_STAGES
    for block_n in tl.range(block_start, main_loop_end, loop_unroll_factor=1):
        stage_idx = ((block_n - block_start) % NUM_STAGES).to(tl.int32)
        start_n = (block_n * BLOCK_N).to(tl.int32)
        future_n = ((block_n + NUM_STAGES) * BLOCK_N).to(tl.int32)

        # Wait K_lo, compute QK_lo
        cdna4_async.wait_group(STREAMS_PREFETCH * NUM_STAGES - 1)
        kt_lo_dot = cdna4_async.load_shared_relaxed(
            kt_half_smem.index(stage_idx), kt_half_dot_layout
        )
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot_lo, kt_lo_dot, qk)

        # Load K_hi into same slot, wait, compute QK_hi
        issue_async_load_k_extend_half(
            kt_half_smem.index(stage_idx), k_base, start_n, seq_len_extend,
            stride_kbs, BLOCK_DMODEL_HALF, BLOCK_N, BLOCK_DMODEL_HALF,
            ACTUAL_BLOCK_DMODEL, kt_half_async_layout,
        )
        cdna4_async.wait_group(0)
        kt_hi_dot = cdna4_async.load_shared_relaxed(
            kt_half_smem.index(stage_idx), kt_half_dot_layout
        )
        qk = do_mma(q_dot_hi, kt_hi_dot, qk)

        # DPE contribution
        if USE_ASYNC_KPE:
            qk = add_qk_dpe_from_shared(
                qk, qpe_dot, kpe_smem.index(stage_idx), kpe_dot_layout,
                BLOCK_DPE,
            )
        elif HAS_DPE:
            qk = add_qk_dpe_extend_from_base(
                qk, qpe_dot, k_base, start_n, seq_len_extend, stride_kbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                kpe_dot_layout,
            )

        # Softmax — rescales acc_lo, we manually rescale acc_hi
        m_i_old = m_i
        acc_lo, l_i, m_i, p = compute_softmax_extend(
            acc_lo, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, xai_temperature_reg, XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE, IS_CAUSAL, Mask, mask_base_idx,
            mask_row_stride, mask_kv_col_offset, USE_CUSTOM_MASK,
            MASK_STEPS, BLOCK_M, BLOCK_N, mma_layout,
            mma_offs_n_col, mma_offs_m_row,
        )
        acc_hi = acc_hi * gl.exp2(m_i_old - m_i)[:, None]

        p_cast = p.to(q_dot_lo.dtype)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)

        # Load V_lo, wait, compute PV_lo
        issue_async_load_v_extend_half(
            v_half_smem.index(stage_idx), v_base, start_n, seq_len_extend,
            stride_vbs, 0, BLOCK_N, BLOCK_DV_HALF,
            ACTUAL_BLOCK_DV, v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_lo_dot = cdna4_async.load_shared_relaxed(
            v_half_smem.index(stage_idx), v_half_dot_layout
        )
        acc_lo = do_mma(p_dot_reg, v_lo_dot, acc_lo)

        # Load V_hi into same slot, wait, compute PV_hi
        issue_async_load_v_extend_half(
            v_half_smem.index(stage_idx), v_base, start_n, seq_len_extend,
            stride_vbs, BLOCK_DV_HALF, BLOCK_N, BLOCK_DV_HALF,
            ACTUAL_BLOCK_DV, v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_hi_dot = cdna4_async.load_shared_relaxed(
            v_half_smem.index(stage_idx), v_half_dot_layout
        )
        acc_hi = do_mma(p_dot_reg, v_hi_dot, acc_hi)

        # Prefetch K_lo + KPE for next N-block
        issue_async_load_k_extend_half(
            kt_half_smem.index(stage_idx), k_base, future_n, seq_len_extend,
            stride_kbs, 0, BLOCK_N, BLOCK_DMODEL_HALF,
            ACTUAL_BLOCK_DMODEL, kt_half_async_layout,
        )
        if USE_ASYNC_KPE:
            issue_async_load_kpe_extend(
                kpe_smem.index(stage_idx), k_base, future_n,
                seq_len_extend, stride_kbs, BLOCK_N, BLOCK_DMODEL,
                BLOCK_DPE, ACTUAL_BLOCK_DPE, kpe_async_layout,
            )

    # --- Tail: drain remaining NUM_STAGES blocks ---
    for tail_i in gl.static_range(NUM_STAGES):
        si = ((main_loop_end + tail_i - block_start) % NUM_STAGES).to(tl.int32)
        start_n = (main_loop_end + tail_i) * BLOCK_N

        cdna4_async.wait_group(STREAMS_PREFETCH * (NUM_STAGES - tail_i) - 1)
        kt_lo_d = cdna4_async.load_shared_relaxed(kt_half_smem.index(si), kt_half_dot_layout)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot_lo, kt_lo_d, qk)

        issue_async_load_k_extend_half(
            kt_half_smem.index(si), k_base, start_n, seq_len_extend,
            stride_kbs, BLOCK_DMODEL_HALF, BLOCK_N, BLOCK_DMODEL_HALF,
            ACTUAL_BLOCK_DMODEL, kt_half_async_layout,
        )
        cdna4_async.wait_group(0)
        kt_hi_d = cdna4_async.load_shared_relaxed(kt_half_smem.index(si), kt_half_dot_layout)
        qk = do_mma(q_dot_hi, kt_hi_d, qk)

        if USE_ASYNC_KPE:
            qk = add_qk_dpe_from_shared(
                qk, qpe_dot, kpe_smem.index(si), kpe_dot_layout, BLOCK_DPE,
            )
        elif HAS_DPE:
            qk = add_qk_dpe_extend_from_base(
                qk, qpe_dot, k_base, start_n, seq_len_extend, stride_kbs,
                BLOCK_N, BLOCK_DMODEL, BLOCK_DPE, ACTUAL_BLOCK_DPE,
                kpe_dot_layout,
            )

        m_i_old = m_i
        acc_lo, l_i, m_i, p = compute_softmax_extend(
            acc_lo, l_i, m_i, qk, start_n, cur_block_m, seq_len_extend,
            qk_scale, LOGIT_CAP, xai_temperature_reg, XAI_TEMPERATURE_LEN,
            SLIDING_WINDOW_SIZE, IS_CAUSAL, Mask, mask_base_idx,
            mask_row_stride, mask_kv_col_offset, USE_CUSTOM_MASK,
            MASK_STEPS, BLOCK_M, BLOCK_N, mma_layout,
            mma_offs_n_col, mma_offs_m_row,
        )
        acc_hi = acc_hi * gl.exp2(m_i_old - m_i)[:, None]

        p_cast = p.to(q_dot_lo.dtype)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)

        issue_async_load_v_extend_half(
            v_half_smem.index(si), v_base, start_n, seq_len_extend,
            stride_vbs, 0, BLOCK_N, BLOCK_DV_HALF,
            ACTUAL_BLOCK_DV, v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_lo_d = cdna4_async.load_shared_relaxed(v_half_smem.index(si), v_half_dot_layout)
        acc_lo = do_mma(p_dot_reg, v_lo_d, acc_lo)

        issue_async_load_v_extend_half(
            v_half_smem.index(si), v_base, start_n, seq_len_extend,
            stride_vbs, BLOCK_DV_HALF, BLOCK_N, BLOCK_DV_HALF,
            ACTUAL_BLOCK_DV, v_half_async_layout,
        )
        cdna4_async.wait_group(0)
        v_hi_d = cdna4_async.load_shared_relaxed(v_half_smem.index(si), v_half_dot_layout)
        acc_hi = do_mma(p_dot_reg, v_hi_d, acc_hi)

    return acc_lo, acc_hi, l_i, m_i


@gluon.jit
def attn_fwd_inner_extend_short(
    acc,
    l_i,
    m_i,
    q_dot,  #
    qpe_dot,  #
    k_base,
    v_base,  #
    cur_block_m,
    seq_len_extend,  #
    stride_kbs,
    stride_vbs,  #
    block_start,
    block_end,  #
    kt_smem,
    v_smem,  #
    qk_scale,
    LOGIT_CAP: gl.constexpr,  #
    xai_temperature_reg,
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    IS_CAUSAL: gl.constexpr,  #
    Mask,
    mask_base_idx,
    mask_row_stride,
    mask_kv_col_offset,  #
    USE_CUSTOM_MASK: gl.constexpr,  #
    MASK_STEPS: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    ACTUAL_BLOCK_DMODEL: gl.constexpr,  #
    BLOCK_DPE: gl.constexpr,
    ACTUAL_BLOCK_DPE: gl.constexpr,  #
    BLOCK_DV: gl.constexpr,
    ACTUAL_BLOCK_DV: gl.constexpr,  #
    kt_async_layout: gl.constexpr,
    v_async_layout: gl.constexpr,  #
    kt_dot_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,  #
    mma_layout: gl.constexpr,
    mma_offs_n_col: gl.constexpr,
    mma_offs_m_row: gl.constexpr,  #
    V_PRELOAD: gl.constexpr = False,  #
    SKIP_BOUNDS_CHECK: gl.constexpr = False,  #
):
    """Preload all K/V blocks into distinct SMEM slots, then compute.

    Requires (block_end - block_start) <= NUM_STAGES so each block gets its own buffer.
    Phase 1: issue K[i]->smem[i] and V[i]->smem[i] for all i.
    Phase 2: wait_group(0) -- everything fully drained.
    Phase 3: compute QK.softmax.PV block-by-block from resident SMEM.

    When V_PRELOAD is True, V is loaded from SMEM to registers before softmax
    so the smem->register transfer overlaps with softmax ALU work.
    """
    cdna4_async.wait_group(0)

    # Phase 1: bulk-issue all DMAs
    n_local_blocks = block_end - block_start
    for local_idx in tl.range(0, n_local_blocks):
        buf_idx = local_idx.to(gl.int32)
        start_n = (block_start + local_idx) * BLOCK_N
        issue_async_load_k_extend(
            kt_smem.index(buf_idx),
            k_base,
            start_n,
            seq_len_extend,
            stride_kbs,
            BLOCK_N,
            BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL,
            kt_async_layout,
            SKIP_BOUNDS_CHECK,
        )
        issue_async_load_v_extend(
            v_smem.index(buf_idx),
            v_base,
            start_n,
            seq_len_extend,
            stride_vbs,
            BLOCK_N,
            BLOCK_DV,
            ACTUAL_BLOCK_DV,
            v_async_layout,
            SKIP_BOUNDS_CHECK,
        )

    # Phase 2: drain every outstanding DMA
    cdna4_async.wait_group(0)

    # Phase 3: compute from fully-resident SMEM (no DMA during this loop)
    for local_idx in tl.range(0, n_local_blocks):
        buf_idx = local_idx.to(gl.int32)
        start_n = (block_start + local_idx) * BLOCK_N

        kt_dot = cdna4_async.load_shared_relaxed(
            kt_smem.index(buf_idx), kt_dot_layout
        )
        if V_PRELOAD:
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(buf_idx), v_dot_layout)

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mma_layout)
        qk = do_mma(q_dot, kt_dot, qk)
        if BLOCK_DPE > 0:
            qk = add_qk_dpe_extend_from_base(
                qk,
                qpe_dot,
                k_base,
                start_n,
                seq_len_extend,
                stride_kbs,
                BLOCK_N,
                BLOCK_DMODEL,
                BLOCK_DPE,
                ACTUAL_BLOCK_DPE,
                kt_dot_layout,
            )

        acc, l_i, m_i, p = compute_softmax_extend(
            acc,
            l_i,
            m_i,
            qk,
            start_n,
            cur_block_m,
            seq_len_extend,
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
            MASK_STEPS,
            BLOCK_M,
            BLOCK_N,
            mma_layout,
            mma_offs_n_col,
            mma_offs_m_row,
        )

        if not V_PRELOAD:
            v_dot = cdna4_async.load_shared_relaxed(v_smem.index(buf_idx), v_dot_layout)
        p_cast = p.to(v_dot.dtype)
        p_dot = gl.convert_layout(p_cast, p_dot_layout)
        acc = do_mma(p_dot, v_dot, acc)

    return acc, l_i, m_i
