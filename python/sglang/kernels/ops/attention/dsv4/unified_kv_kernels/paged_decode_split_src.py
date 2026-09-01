# SPDX-License-Identifier: MIT
# Stage-3 PoC: split-source unified_kv decode.
#
# "Logical unified, physical split": one logical unified_kv slot space, backed by
# TWO physical buffers of different dtype/layout:
#
#   swa_kv     : [swa_pages, D]                    bf16   (1024 B/row for D=512)
#   packed_kv  : [num_pages, bytes_per_page]       uint8  (mixed packed, 584 B/row)
#                per-token: 448 fp8 nope + 64 bf16 rope (=576 B) + 7 ue8m0 scale (+1 pad)
#                per-page : [P tokens nope+rope (P*576)] [P tokens scale (P*8)] padded
#
# Slot dispatch (matches runtime.py index contract):
#   slot <  swa_pages  -> SWA ring entry   -> read swa_kv[slot]        (bf16)
#   slot >= swa_pages  -> compressed entry -> read packed_kv[slot-swa_pages]
#                                             (fp8 nope * exp2(scale-127) ++ bf16 rope)
#
# The dequant addressing is copied verbatim from dsv4.dequant_k_cache so the
# in-kernel dequant is bit-identical to the production non-unified reader.
#
# This PoC implements ONLY the single-pass fused path (kv_splits == 1). The
# split-K path is a mechanical copy of the same KV-load block into
# paged_decode._paged_decode_split_kernel; deferred to the production patch.

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


@triton.jit
def _paged_decode_fused_split_src_kernel(
    q_ptr,  # [N, H, D] bf16/fp16
    swa_ptr,  # [swa_pages, D] bf16  (same dtype as q)
    pk_fp8_ptr,  # packed buffer viewed as fp8   (flat)
    pk_bf16_ptr,  # packed buffer viewed as bf16  (flat)
    pk_u8_ptr,  # packed buffer viewed as uint8 (flat)
    kv_indices_ptr,  # [total_indices] int32   (logical slots)
    kv_indptr_ptr,  # [N+1] int32
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
    """Single-pass online-softmax decode with per-slot source dispatch.

    Identical arithmetic to paged_decode._paged_decode_fused_kernel; the only
    change is the KV tile is assembled from two physical buffers based on
    ``slot >= swa_pages``. CUDAGraph-safe: dispatch is a runtime tl.where over
    slot VALUES, launch shape depends only on capture-time (T, H).
    """
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    num_tiles = tl.cdiv(kv_len, BLOCK_K)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)

    # Per-D compile-time masks for the packed split (nope vs rope tail).
    nope_mask = d_offs < DIM_NOPE
    rope_mask = (d_offs >= DIM_NOPE) & (d_offs < DIM_NOPE + DIM_ROPE)
    g_idx_per_d = d_offs // TILE_SIZE  # nope scale-tile index per column

    for j in tl.range(0, num_tiles, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < kv_len
        slot = tl.load(kv_indices_ptr + kv_start + k_pos, mask=valid, other=0)

        is_comp = slot >= swa_pages  # [BLOCK_K]

        # ---- SWA branch (bf16, full D) ----
        swa_v = tl.load(
            swa_ptr
            + slot[:, None] * swa_stride_n
            + d_offs[None, :] * swa_stride_d,
            mask=valid[:, None] & (~is_comp)[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # ---- compressed branch (packed: fp8 nope * ue8m0 scale ++ bf16 rope) ----
        loc = slot - swa_pages  # compressed token id
        page_idx = loc // PAGE_SIZE
        in_page = loc % PAGE_SIZE
        page_byte_base = page_idx * BYTES_PER_PAGE
        token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES  # [BLOCK_K]
        token_scale_base = (
            page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
        )

        cmask = valid[:, None] & is_comp[:, None] & d_mask[None, :]

        # nope: fp8 byte offset = token_data_base + d  (d < DIM_NOPE)
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

        # rope: bf16 elem offset = (token_data_base + DIM_NOPE)//2 + (d - DIM_NOPE)
        rope_base = (token_data_base + DIM_NOPE) // 2
        bf16_off = rope_base[:, None] + (d_offs[None, :] - DIM_NOPE)
        rope_val = tl.load(
            pk_bf16_ptr + bf16_off, mask=cmask & rope_mask[None, :], other=0.0
        ).to(tl.float32)

        comp_v = tl.where(nope_mask[None, :], nope_val, rope_val)

        # ---- merge sources ----
        kv = tl.where(is_comp[:, None], comp_v, swa_v).to(q.dtype)

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    sink_raw = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(
        tl.float32
    )
    sink = sink_raw * log2e
    m_final = tl.maximum(m_i, sink)
    alpha_kv = tl.exp2(m_i - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_i * alpha_kv + alpha_sink

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(
        l_final[:, None] > 0.0, (acc * alpha_kv[:, None]) / denom[:, None], 0.0
    )
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=h_mask[:, None] & d_mask[None, :],
    )


def sparse_attn_v4_paged_decode_split_src(
    q: torch.Tensor,  # [N, H, D]
    swa_kv: torch.Tensor,  # [swa_pages, D] bf16
    packed_kv: torch.Tensor,  # [num_pages, bytes_per_page] uint8
    kv_indices: torch.Tensor,  # [total_indices] int32 (logical slots)
    kv_indptr: torch.Tensor,  # [N+1] int32
    attn_sink: torch.Tensor,  # [H] fp32
    softmax_scale: float,
    swa_pages: int,
    packed_page_size: int,
    block_h: int | None = None,
    block_k: int = 16,
) -> torch.Tensor:
    """PoC split-source unified_kv decode (fused single-pass only)."""
    assert q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
    assert swa_kv.dtype == q.dtype
    assert packed_kv.dtype == torch.uint8 and packed_kv.is_contiguous()
    T, H, D = q.shape
    assert swa_kv.shape[1] == D

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
    n_head_blocks = (H + block_h - 1) // block_h
    qk_scale = float(softmax_scale) * LOG2E

    grid = (T, n_head_blocks)
    _paged_decode_fused_split_src_kernel[grid](
        q,
        swa_kv,
        pk_fp8,
        pk_bf16,
        pk_u8,
        kv_indices,
        kv_indptr,
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
