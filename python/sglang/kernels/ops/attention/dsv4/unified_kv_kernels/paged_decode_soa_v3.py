# SPDX-License-Identifier: MIT
# Stage-3 PoC v3: SoA fp8 unified_kv decode with COMPACT scale load.
#
# Diagnostic finding (MI355X, jobs 27646/27653): both the packed-AoS dequant AND
# the existing "tuned" clean-SoA QUANT_KV path are 1.3-13x SLOWER than the bf16
# baseline at large batch/context, despite reading fewer bytes. Root cause: the
# fp8 decode is COMPUTE/ISSUE bound, dominated by the per-tile 512-wide scale
# BROADCAST load (only D/GROUP=8 distinct scales per token, read 512-wide -> 64x
# over-read) plus fp8->f32 casts.
#
# v3 fixes the scale traffic: scales are loaded COMPACT as [BLOCK_K, 8] (fp32,
# pre-exponentiated at store so no in-kernel exp2) and broadcast to [BLOCK_K,512]
# IN REGISTERS. RoPE stays bf16 (phase-1 requirement) in its own contiguous
# buffer. Layout (Structure-of-Arrays, all coalesced):
#     nope_fp8  : [C, 448]  fp8    (7 groups x 64, 1x64 block scale)
#     rope_bf16 : [C,  64]  bf16   (kept exactly bf16)
#     scale_f32 : [C,   8]  fp32   (7 real group scales + 1 pad=1.0)
#
# Two-segment single-source (SWA bf16 | compressed SoA-fp8), one online-softmax
# accumulator; each tile issues exactly one source's loads.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _paged_decode_reduce_kernel,
    _kv_splits_heuristic,
    _cu_count,
)

LOG2E = 1.4426950408889634
_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

_DIM_NOPE = 448
_DIM_ROPE = 64
_GROUP = 64
_NUM_G = _DIM_NOPE // _GROUP  # 7
_NUM_G_PAD = 8


@triton.jit
def _decode_soa_v3_kernel(
    q_ptr,  # [N,H,D]
    swa_ptr,  # [swa_pages, D] bf16
    nope_ptr,  # [C, 448] fp8
    rope_ptr,  # [C, 64] bf16
    scale_ptr,  # [C, 8] fp32
    kv_indices_ptr,  # [tot] int32
    kv_indptr_ptr,  # [N+1] int32
    swa_len_ptr,  # [N] int32
    attn_sink_ptr,  # [H] fp32
    out_ptr,  # [N,H,D]
    q_st, q_sh, q_sd,
    swa_sn, swa_sd,
    nope_sn, rope_sn, scale_sn,
    o_st, o_sh, o_sd,
    qk_scale, log2e, swa_pages,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_KC: tl.constexpr,
    DIM_NOPE: tl.constexpr, DIM_ROPE: tl.constexpr,
    GROUP: tl.constexpr, NUM_G_PAD: tl.constexpr,
    NS_A: tl.constexpr, NS_C: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr + t * q_st + h_offs[:, None] * q_sh + d_offs[None, :] * q_sd,
        mask=h_mask[:, None] & d_mask[None, :], other=0.0,
    )

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    swa_len = tl.load(swa_len_ptr + t)
    swa_len = tl.minimum(tl.maximum(swa_len, 0), kv_len)

    neg = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    # ---- LOOP A: SWA bf16 ----
    k_offs = tl.arange(0, BLOCK_K)
    na = tl.cdiv(swa_len, BLOCK_K)
    for j in tl.range(0, na, num_stages=NS_A):
        k_pos = j * BLOCK_K + k_offs
        valid = k_pos < swa_len
        slot = tl.load(kv_indices_ptr + kv_start + k_pos, mask=valid, other=0)
        kv = tl.load(
            swa_ptr + slot[:, None] * swa_sn + d_offs[None, :] * swa_sd,
            mask=valid[:, None] & d_mask[None, :], other=0.0,
        )
        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg)
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    # ---- LOOP B: compressed SoA fp8 ----
    kc_offs = tl.arange(0, BLOCK_KC)
    g_of_d = d_offs // GROUP  # [BLOCK_D] group index per column (0..7)
    nope_mask = d_offs < DIM_NOPE
    rope_mask = (d_offs >= DIM_NOPE) & (d_offs < DIM_NOPE + DIM_ROPE)
    g_cols = tl.arange(0, NUM_G_PAD)  # 0..7
    comp_len = kv_len - swa_len
    nb = tl.cdiv(comp_len, BLOCK_KC)
    for j in tl.range(0, nb, num_stages=NS_C):
        k_pos = j * BLOCK_KC + kc_offs
        valid = k_pos < comp_len
        slot = tl.load(
            kv_indices_ptr + kv_start + swa_len + k_pos, mask=valid, other=swa_pages
        )
        loc = slot - swa_pages  # compressed row id

        # --- dequant DIRECT to q.dtype (bf16); no f32 intermediate ---
        # fp8_e4m3 -> bf16 is exact (3 mantissa bits fit in bf16's 7). scale is a
        # pure power-of-two (ue8m0, pre-exponentiated), exactly representable in
        # bf16, so bf16*bf16 here is a lossless exponent shift == f32-then-round.
        # Materializing bf16 (not f32) halves VGPR pressure -> higher occupancy
        # -> better latency hiding for the fp8 loads.
        nope_q = tl.load(
            nope_ptr + loc[:, None] * nope_sn + d_offs[None, :],
            mask=valid[:, None] & nope_mask[None, :], other=0.0,
        ).to(q.dtype)
        # COMPACT scale load [BLOCK_KC, 8], broadcast in REGISTERS.
        scale8 = tl.load(
            scale_ptr + loc[:, None] * scale_sn + g_cols[None, :],
            mask=valid[:, None], other=1.0,
        ).to(q.dtype)  # [BLOCK_KC, 8], pow2 -> exact in bf16
        nope3 = tl.reshape(nope_q, (BLOCK_KC, NUM_G_PAD, GROUP))
        nope3 = nope3 * scale8[:, :, None]
        nope_deq = tl.reshape(nope3, (BLOCK_KC, BLOCK_D))
        # rope bf16 [BLOCK_KC, 64] placed at d in [DIM_NOPE, DIM_NOPE+DIM_ROPE)
        rope_val = tl.load(
            rope_ptr + loc[:, None] * rope_sn + (d_offs[None, :] - DIM_NOPE),
            mask=valid[:, None] & rope_mask[None, :], other=0.0,
        ).to(q.dtype)
        kv = tl.where(nope_mask[None, :], nope_deq, rope_val)

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg)
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    sink_raw = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg).to(tl.float32)
    sink = sink_raw * log2e
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink
    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha_kv[:, None]) / denom[:, None], 0.0)
    tl.store(
        out_ptr + t * o_st + h_offs[:, None] * o_sh + d_offs[None, :] * o_sd,
        out.to(out_ptr.dtype.element_ty),
        mask=h_mask[:, None] & d_mask[None, :],
    )


@triton.jit
def _decode_soa_v3_split_kernel(
    q_ptr, swa_ptr, nope_ptr, rope_ptr, scale_ptr,
    kv_indices_ptr, kv_indptr_ptr, swa_len_ptr,
    m_partial_ptr, l_partial_ptr, acc_partial_ptr,
    q_st, q_sh, q_sd,
    swa_sn, swa_sd,
    nope_sn, rope_sn, scale_sn,
    mp_st, mp_sk, mp_sh,
    lp_st, lp_sk, lp_sh,
    ap_st, ap_sk, ap_sh, ap_sd,
    qk_scale, swa_pages,
    H: tl.constexpr, D: tl.constexpr, KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
    DIM_NOPE: tl.constexpr, DIM_ROPE: tl.constexpr,
    GROUP: tl.constexpr, NUM_G_PAD: tl.constexpr,
    NS_A: tl.constexpr, NS_C: tl.constexpr,
):
    """Split-K variant. Partitions the WHOLE [0,kv_len) stream identically to
    the baseline _paged_decode_split_kernel so the shared reduce kernel's
    tiles_per_segment / act_num_segments masking stays consistent. Within a
    segment, the swa (bf16) and compressed (SoA fp8) portions are processed in
    two single-source sub-loops sharing one online-softmax state, emitting
    pre-sink (m,l,acc) partials."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D
    q = tl.load(
        q_ptr + t * q_st + h_offs[:, None] * q_sh + d_offs[None, :] * q_sd,
        mask=h_mask[:, None] & d_mask[None, :], other=0.0,
    )

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    swa_len = tl.load(swa_len_ptr + t)
    swa_len = tl.minimum(tl.maximum(swa_len, 0), kv_len)

    tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
    seg_lo = pid_k * tiles_per_segment * BLOCK_K
    if seg_lo >= kv_len:
        return
    seg_hi = tl.minimum((pid_k + 1) * tiles_per_segment * BLOCK_K, kv_len)

    neg = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    k_offs = tl.arange(0, BLOCK_K)

    # ---- SWA sub-range within this segment (tl.range -> software-pipelined) ----
    swa_lo = seg_lo
    swa_hi = tl.minimum(seg_hi, swa_len)
    na = tl.cdiv(swa_hi - swa_lo, BLOCK_K)  # 0 when this segment has no swa
    for jj in tl.range(0, na, num_stages=NS_A):
        k_pos = swa_lo + jj * BLOCK_K + k_offs
        valid = k_pos < swa_hi
        slot = tl.load(kv_indices_ptr + kv_start + k_pos, mask=valid, other=0)
        kv = tl.load(
            swa_ptr + slot[:, None] * swa_sn + d_offs[None, :] * swa_sd,
            mask=valid[:, None] & d_mask[None, :], other=0.0,
        )
        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg)
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    # ---- compressed sub-range within this segment (tl.range -> pipelined) ----
    comp_lo = tl.maximum(seg_lo, swa_len)
    comp_hi = seg_hi
    g_cols = tl.arange(0, NUM_G_PAD)
    nope_mask = d_offs < DIM_NOPE
    rope_mask = (d_offs >= DIM_NOPE) & (d_offs < DIM_NOPE + DIM_ROPE)
    nb = tl.cdiv(comp_hi - comp_lo, BLOCK_K)  # 0 when this segment has no compressed
    for jj in tl.range(0, nb, num_stages=NS_C):
        k_pos = comp_lo + jj * BLOCK_K + k_offs
        valid = k_pos < comp_hi
        slot = tl.load(
            kv_indices_ptr + kv_start + k_pos, mask=valid, other=swa_pages
        )
        loc = slot - swa_pages
        # dequant DIRECT to q.dtype (bf16); no f32 intermediate (see fused kernel).
        nope_q = tl.load(
            nope_ptr + loc[:, None] * nope_sn + d_offs[None, :],
            mask=valid[:, None] & nope_mask[None, :], other=0.0,
        ).to(q.dtype)
        scale8 = tl.load(
            scale_ptr + loc[:, None] * scale_sn + g_cols[None, :],
            mask=valid[:, None], other=1.0,
        ).to(q.dtype)
        nope3 = tl.reshape(nope_q, (BLOCK_K, NUM_G_PAD, GROUP))
        nope3 = nope3 * scale8[:, :, None]
        nope_deq = tl.reshape(nope3, (BLOCK_K, BLOCK_D))
        rope_val = tl.load(
            rope_ptr + loc[:, None] * rope_sn + (d_offs[None, :] - DIM_NOPE),
            mask=valid[:, None] & rope_mask[None, :], other=0.0,
        ).to(q.dtype)
        kv = tl.where(nope_mask[None, :], nope_deq, rope_val)
        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg)
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    m_base = t * mp_st + pid_k * mp_sk
    tl.store(m_partial_ptr + m_base + h_offs * mp_sh, m_i, mask=h_mask)
    l_base = t * lp_st + pid_k * lp_sk
    tl.store(l_partial_ptr + l_base + h_offs * lp_sh, l_i, mask=h_mask)
    a_base = t * ap_st + pid_k * ap_sk
    tl.store(
        acc_partial_ptr + a_base + h_offs[:, None] * ap_sh + d_offs[None, :] * ap_sd,
        acc, mask=h_mask[:, None] & d_mask[None, :],
    )


def sparse_attn_v4_paged_decode_soa_v3(
    q, swa_kv, nope_fp8, rope_bf16, scale_f32,
    kv_indices, kv_indptr, swa_len, attn_sink, softmax_scale,
    swa_pages, block_h=None, block_k=None, block_kc=None, kv_splits=None,
    ns_a=3, ns_c=3,
):
    assert q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
    assert swa_kv.dtype == q.dtype
    assert nope_fp8.dtype == _FP8_DTYPE and nope_fp8.is_contiguous()
    assert rope_bf16.dtype == torch.bfloat16 and rope_bf16.is_contiguous()
    assert scale_f32.dtype == torch.float32 and scale_f32.is_contiguous()
    T, H, D = q.shape
    swa_len = swa_len.to(torch.int32)
    out = torch.empty_like(q)
    if block_h is None:
        block_h = triton.next_power_of_2(min(H, 64))
    block_h = max(triton.next_power_of_2(block_h), 16)
    block_d = triton.next_power_of_2(D)
    if block_k is None:
        block_k = 16  # measured optimum (job 27984): narrow tile keeps VGPR low
    if block_kc is None:
        block_kc = 16
    n_hb = (H + block_h - 1) // block_h
    h_padded = n_hb * block_h
    qk_scale = float(softmax_scale) * LOG2E
    nw = 4 if block_h <= 32 else 8

    if kv_splits is None:
        kv_splits = _kv_splits_heuristic(T, H, block_h)

    # -------- fused single-pass (grid already saturates GPU) --------
    if kv_splits == 1:
        grid = (T, n_hb)
        _decode_soa_v3_kernel[grid](
            q, swa_kv, nope_fp8, rope_bf16, scale_f32,
            kv_indices, kv_indptr, swa_len, attn_sink, out,
            q.stride(0), q.stride(1), q.stride(2),
            swa_kv.stride(0), swa_kv.stride(1),
            nope_fp8.stride(0), rope_bf16.stride(0), scale_f32.stride(0),
            out.stride(0), out.stride(1), out.stride(2),
            qk_scale, LOG2E, swa_pages,
            H=H, D=D, BLOCK_H=block_h, BLOCK_D=block_d,
            BLOCK_K=block_k, BLOCK_KC=block_kc,
            DIM_NOPE=_DIM_NOPE, DIM_ROPE=_DIM_ROPE,
            GROUP=_GROUP, NUM_G_PAD=_NUM_G_PAD,
            NS_A=ns_a, NS_C=ns_c,
            num_warps=nw, num_stages=1,
        )
        return out

    # -------- split-K: v3 split kernel writes partials, reuse baseline reduce --------
    # split path uses a single BLOCK_K for the whole stream partitioning so the
    # reduce's tiles_per_segment / act_num_segments math stays consistent.
    bk = block_k
    m_partial = torch.empty((T, kv_splits, h_padded), dtype=torch.float32, device=q.device)
    l_partial = torch.empty_like(m_partial)
    acc_partial = torch.empty((T, kv_splits, h_padded, D), dtype=torch.float32, device=q.device)
    grid_split = (T, n_hb, kv_splits)
    _decode_soa_v3_split_kernel[grid_split](
        q, swa_kv, nope_fp8, rope_bf16, scale_f32,
        kv_indices, kv_indptr, swa_len,
        m_partial, l_partial, acc_partial,
        q.stride(0), q.stride(1), q.stride(2),
        swa_kv.stride(0), swa_kv.stride(1),
        nope_fp8.stride(0), rope_bf16.stride(0), scale_f32.stride(0),
        m_partial.stride(0), m_partial.stride(1), m_partial.stride(2),
        l_partial.stride(0), l_partial.stride(1), l_partial.stride(2),
        acc_partial.stride(0), acc_partial.stride(1), acc_partial.stride(2), acc_partial.stride(3),
        qk_scale, swa_pages,
        H=H, D=D, KV_SPLITS=kv_splits,
        BLOCK_H=block_h, BLOCK_D=block_d, BLOCK_K=bk,
        DIM_NOPE=_DIM_NOPE, DIM_ROPE=_DIM_ROPE,
        GROUP=_GROUP, NUM_G_PAD=_NUM_G_PAD,
        NS_A=ns_a, NS_C=ns_c,
        num_warps=nw, num_stages=1,
    )

    base_grid_t_h = T * H
    target_reduce_wg = 2 * _cu_count()
    if base_grid_t_h >= target_reduce_wg:
        d_chunk = block_d
    else:
        d_chunks_needed = max(1, target_reduce_wg // base_grid_t_h)
        d_chunks_needed = min(d_chunks_needed, block_d // 32)
        d_chunk = max(32, triton.next_power_of_2(block_d // d_chunks_needed))
    grid_reduce = (T, H, (D + d_chunk - 1) // d_chunk)
    _paged_decode_reduce_kernel[grid_reduce](
        m_partial, l_partial, acc_partial, attn_sink, kv_indptr, out,
        m_partial.stride(0), m_partial.stride(1), m_partial.stride(2),
        l_partial.stride(0), l_partial.stride(1), l_partial.stride(2),
        acc_partial.stride(0), acc_partial.stride(1), acc_partial.stride(2), acc_partial.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        LOG2E, H, D, kv_splits,
        BLOCK_D=block_d, D_CHUNK=d_chunk, BLOCK_K=bk, num_warps=4,
    )
    return out
