# SPDX-License-Identifier: MIT
# Stage-3 PoC v2: TWO-SEGMENT single-source unified_kv decode.
#
# Motivation (from PoC v1 measurement on MI355X, job 27639):
#   v1 loaded BOTH the bf16 SWA tile and the packed compressed tile for EVERY
#   BLOCK_K tile and merged with tl.where. That triples/quadruples KV memory
#   traffic vs the bf16 baseline (1 coalesced load/tile) and adds dequant ALU,
#   giving a 3.9x-22.9x slowdown despite the 43% resident-byte saving.
#
# Key layout fact (verified in runtime.build_decode_streams / fill_compress_tail):
#   per request the index stream is CONTIGUOUS and MONOTONE:
#       [  SWA slots  (swa_len[t] of them)  |  compressed slots (tail)  ]
#   so a request has exactly ONE crossover point at swa_len[t]. Therefore a
#   BLOCK_K tile is (almost) always single-source; the only mixed tile is the
#   single boundary tile, and we avoid even that by iterating the two segments
#   in SEPARATE K-loops that share one online-softmax accumulator.
#
#   loop A: k in [0, swa_len)      -> load ONLY bf16 swa_kv         (baseline-fast)
#   loop B: k in [swa_len, kv_len) -> load ONLY packed compressed    (dequant)
#
# Each tile issues exactly one KV load of its own source. No tl.where merge, no
# double traffic. Online softmax is associative so processing the two segments
# sequentially in one (m_i, l_i, acc) state is exact.
#
# The packed dequant addressing is byte-identical to dsv4.dequant_k_cache and to
# PoC v1 (already validated bitwise on MI355X).

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

LOG2E = 1.4426950408889634
_FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# Packed layout constants (must match dsv4.dequant_k_cache).
_DIM_NOPE = 448
_DIM_ROPE = 64
_TILE_SIZE = 64
_NUM_SCALE_TILES = _DIM_NOPE // _TILE_SIZE  # 7
_NOPE_ROPE_BYTES = _DIM_NOPE + _DIM_ROPE * 2  # 576
_PADDED_SCALE_PER_TOKEN = _NUM_SCALE_TILES + 1  # 8

_NEG = -3.4028234663852886e38


@triton.jit
def _paged_decode_v2_kernel(
    q_ptr,  # [N, H, D] bf16/fp16
    swa_ptr,  # [swa_pages, D] bf16
    pk_fp8_ptr,  # packed buffer, fp8 view (flat)
    pk_bf16_ptr,  # packed buffer, bf16 view (flat)
    pk_u8_ptr,  # packed buffer, uint8 view (flat)
    kv_indices_ptr,  # [total_indices] int32 (logical slots, [swa|comp] per req)
    kv_indptr_ptr,  # [N+1] int32
    swa_len_ptr,  # [N] int32  per-request SWA prefix length (crossover point)
    attn_sink_ptr,  # [H] fp32
    out_ptr,  # [N, H, D]
    q_stride_t,
    q_stride_h,
    q_stride_d,
    swa_stride_n,
    swa_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    qk_scale,
    log2e,
    swa_pages,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_KC: tl.constexpr,  # separate K tile for the (heavier) compressed loop
    # packed-layout constexprs
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr + t * q_stride_t + h_offs[:, None] * q_stride_h + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    swa_len = tl.load(swa_len_ptr + t)
    # clamp defensively
    swa_len = tl.minimum(tl.maximum(swa_len, 0), kv_len)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    # =============================== LOOP A: SWA (bf16) ===============================
    k_offs = tl.arange(0, BLOCK_K)
    num_tiles_a = tl.cdiv(swa_len, BLOCK_K)
    for j in tl.range(0, num_tiles_a, num_stages=3):
        k_pos = j * BLOCK_K + k_offs
        valid = k_pos < swa_len
        slot = tl.load(kv_indices_ptr + kv_start + k_pos, mask=valid, other=0)
        kv = tl.load(
            swa_ptr + slot[:, None] * swa_stride_n + d_offs[None, :] * swa_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    # =========================== LOOP B: compressed (packed) ===========================
    kc_offs = tl.arange(0, BLOCK_KC)
    nope_mask = d_offs < DIM_NOPE
    rope_mask = (d_offs >= DIM_NOPE) & (d_offs < DIM_NOPE + DIM_ROPE)
    g_idx_per_d = d_offs // TILE_SIZE
    comp_len = kv_len - swa_len
    num_tiles_b = tl.cdiv(comp_len, BLOCK_KC)
    for j in tl.range(0, num_tiles_b, num_stages=3):
        k_pos = j * BLOCK_KC + kc_offs  # offset WITHIN the compressed segment
        valid = k_pos < comp_len
        slot = tl.load(
            kv_indices_ptr + kv_start + swa_len + k_pos, mask=valid, other=swa_pages
        )
        loc = slot - swa_pages  # compressed token id
        page_idx = loc // PAGE_SIZE
        in_page = loc % PAGE_SIZE
        page_byte_base = page_idx * BYTES_PER_PAGE
        token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
        token_scale_base = page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN

        cmask = valid[:, None] & d_mask[None, :]
        # nope fp8
        fp8_off = token_data_base[:, None] + d_offs[None, :]
        fp8_vals = tl.load(
            pk_fp8_ptr + fp8_off, mask=cmask & nope_mask[None, :], other=0.0
        ).to(tl.float32)
        scale_off = token_scale_base[:, None] + g_idx_per_d[None, :]
        scale_u8 = tl.load(
            pk_u8_ptr + scale_off, mask=cmask & nope_mask[None, :], other=127
        ).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))
        nope_val = fp8_vals * scale_pow2
        # rope bf16
        rope_base = (token_data_base + DIM_NOPE) // 2
        bf16_off = rope_base[:, None] + (d_offs[None, :] - DIM_NOPE)
        rope_val = tl.load(
            pk_bf16_ptr + bf16_off, mask=cmask & rope_mask[None, :], other=0.0
        ).to(tl.float32)
        kv = tl.where(nope_mask[None, :], nope_val, rope_val).to(q.dtype)

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    # ===================================== EPILOGUE =====================================
    sink_raw = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    sink = sink_raw * log2e
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink
    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha_kv[:, None]) / denom[:, None], 0.0)
    tl.store(
        out_ptr + t * out_stride_t + h_offs[:, None] * out_stride_h + d_offs[None, :] * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=h_mask[:, None] & d_mask[None, :],
    )


def sparse_attn_v4_paged_decode_split_src_v2(
    q: torch.Tensor,
    swa_kv: torch.Tensor,
    packed_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    swa_len: torch.Tensor,  # [N] int32 per-request SWA prefix length
    attn_sink: torch.Tensor,
    softmax_scale: float,
    swa_pages: int,
    packed_page_size: int,
    block_h: int | None = None,
    block_k: int | None = None,
    block_kc: int | None = None,
) -> torch.Tensor:
    """v2 two-segment single-source unified_kv decode (fused single-pass)."""
    assert q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
    assert swa_kv.dtype == q.dtype
    assert packed_kv.dtype == torch.uint8 and packed_kv.is_contiguous()
    T, H, D = q.shape
    assert swa_kv.shape[1] == D
    swa_len = swa_len.to(torch.int32)

    u8 = packed_kv.view(torch.uint8)
    bytes_per_page = u8.shape[-1]
    pk_fp8 = u8.view(_FP8_DTYPE).reshape(-1)
    pk_bf16 = u8.view(torch.bfloat16).reshape(-1)
    pk_u8 = u8.reshape(-1)

    out = torch.empty_like(q)
    if block_h is None:
        block_h = triton.next_power_of_2(min(H, 64))
    block_h = max(triton.next_power_of_2(block_h), 16)
    block_d = triton.next_power_of_2(D)
    if block_k is None:
        block_k = 32  # bf16 SWA loop; wide tile amortizes MFMA setup
    if block_kc is None:
        block_kc = 32  # compressed loop: wider tile amortizes dequant ALU
    n_head_blocks = (H + block_h - 1) // block_h
    qk_scale = float(softmax_scale) * LOG2E

    grid = (T, n_head_blocks)
    _paged_decode_v2_kernel[grid](
        q,
        swa_kv,
        pk_fp8,
        pk_bf16,
        pk_u8,
        kv_indices,
        kv_indptr,
        swa_len,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        swa_kv.stride(0),
        swa_kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        qk_scale,
        LOG2E,
        swa_pages,
        H=H,
        D=D,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        BLOCK_KC=block_kc,
        BYTES_PER_PAGE=bytes_per_page,
        PAGE_SIZE=packed_page_size,
        DIM_NOPE=_DIM_NOPE,
        DIM_ROPE=_DIM_ROPE,
        TILE_SIZE=_TILE_SIZE,
        NOPE_ROPE_BYTES=_NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=_PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=packed_page_size * _NOPE_ROPE_BYTES,
        num_warps=4 if block_h <= 32 else 8,
        num_stages=2,
    )
    return out
