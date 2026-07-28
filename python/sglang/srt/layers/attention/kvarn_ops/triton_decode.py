# SPDX-License-Identifier: Apache-2.0
"""KVarN Triton decode kernels for SGLang.

Contains:
  - ``_kvarn_build_packed_kv_kernel``: block_table-driven dequant + pool gather
    → packed fp16 K/V for FlashAttention.
  - ``_kvarn_fused_decode_kernel``: fused dequant + online-softmax flash-decode
    (never materializes fp16 K/V to HBM).
  - ``_kvarn_fused_decode_stage1`` + ``_kvarn_fused_decode_stage2``: split-K
    flash-decoding for long-context / low-batch regimes.
  - ``_kvarn_scatter_store_kernel``: scatter fp16 K/V into the tail pool.

Uses ``import triton`` / ``import triton.language as tl``
directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Number of KV-sequence splits for the split-K flash-decoding kernel.
KVARN_NUM_KV_SPLITS = 16
KVARN_MAX_KV_SPLITS = 64

# Autotune configs for the fused decode kernel.  Applied lazily so the module
# can be imported on CPU-only machines without a Triton driver.
_KVARN_DECODE_CONFIGS = [
    triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
]


def _maybe_autotune(kernel):
    """Decorate with triton.autotune if a GPU driver is available.

    This allows the module to be imported on CPU-only machines (e.g. for
    unit tests of the non-Triton code) without raising.
    """
    try:
        return triton.autotune(
            configs=_KVARN_DECODE_CONFIGS,
            key=["D", "GROUP", "Q_PER_KV", "K_BITS", "V_BITS"],
        )(kernel)
    except RuntimeError:
        # No GPU driver — return the bare JIT kernel.
        return kernel


def _maybe_autotune_verify(kernel):
    """Decorate verify kernel with triton.autotune (includes QLEN in key)."""
    try:
        return triton.autotune(
            configs=_KVARN_DECODE_CONFIGS,
            key=["D", "GROUP", "Q_PER_KV", "QLEN", "K_BITS", "V_BITS"],
        )(kernel)
    except RuntimeError:
        return kernel


def adaptive_num_kv_splits(max_blocks_per_req: int) -> int:
    """Context-adaptive split-K count."""
    if max_blocks_per_req <= 80:
        return 16
    if max_blocks_per_req <= 256:
        return 32
    return KVARN_MAX_KV_SPLITS


# ──────────────────────────────────────────────────────────────────────────────
# Scatter store: writes (already-rotated) k, v into the tail pool.
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _kvarn_scatter_store_kernel(
    K_in_ptr,  # [N, Hk, D]                    fp16 (already rotated)
    V_in_ptr,  # [N, Hk, D]                    fp16
    Slot_mapping_ptr,  # [N]                           int32   (slot < 0 ⇒ pad)
    Block_to_slot_ptr,  # [num_blocks_lookup]           int32   (-1 = no slot)
    Pool_K_ptr,  # [POOL_SIZE, group, Hk, D]     fp16
    Pool_V_ptr,  # [POOL_SIZE, group, Hk, D]     fp16
    # strides
    stride_in_n,
    stride_in_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    # constexprs
    GROUP: tl.constexpr,
    D: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    """Scatter one (token, kv_head) row from k, v into pool[slot, pos, hk, :]."""
    i = tl.program_id(0)
    hk = tl.program_id(1)

    sm = tl.load(Slot_mapping_ptr + i)
    if sm < 0:
        return

    block_id = sm // GROUP
    pos = (sm % GROUP).to(tl.int64)
    in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
    if not in_range:
        return
    pool_slot = tl.load(Block_to_slot_ptr + block_id)
    if pool_slot < 0:
        return

    d = tl.arange(0, D)
    src_offs = i * stride_in_n + hk * stride_in_h + d
    k_row = tl.load(K_in_ptr + src_offs)
    v_row = tl.load(V_in_ptr + src_offs)

    dst_offs = (
        pool_slot.to(tl.int64) * stride_pool_b
        + pos * stride_pool_t
        + hk * stride_pool_h
        + d
    )
    tl.store(Pool_K_ptr + dst_offs, k_row)
    tl.store(Pool_V_ptr + dst_offs, v_row)


# ──────────────────────────────────────────────────────────────────────────────
# Block-table-driven build-packed-KV kernel (materialize path).
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _kvarn_build_packed_kv_kernel(
    Block_table_ptr,  # [B, max_blocks]                          int32
    Seq_lens_ptr,  # [B]                                      int32
    Cu_seqlens_ptr,  # [B+1]                                    int32
    Block_to_slot_ptr,  # [num_blocks_lookup]                      int32
    KV_cache_ptr,  # [num_blocks, num_kv_heads, TILE_BYTES]   uint8
    Tail_K_pool_ptr,  # [POOL_SIZE, group, Hk, D]                fp16
    Tail_V_pool_ptr,  # [POOL_SIZE, group, Hk, D]                fp16
    K_out_ptr,  # [max_total_tokens, Hk, D]                fp16
    V_out_ptr,  # [max_total_tokens, Hk, D]                fp16
    # strides
    stride_bt_b,
    stride_kv_b,
    stride_kv_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_out_t,
    stride_out_h,
    # constexprs
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
    K_PACKED_OFFSET: tl.constexpr,
    K_S_COL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr,
    K_S_ROW_OFFSET: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr,
    V_S_COL_OFFSET: tl.constexpr,
    V_S_ROW_OFFSET: tl.constexpr,
    V_ZP_OFFSET: tl.constexpr,
):
    """Grid: (B * MAX_BLOCKS_PER_REQ, Hk). One (request-block, head) per program."""
    bk = tl.program_id(0)
    hk = tl.program_id(1)
    b = bk // MAX_BLOCKS_PER_REQ
    k = bk % MAX_BLOCKS_PER_REQ

    seq_len = tl.load(Seq_lens_ptr + b)
    rem = seq_len - k * GROUP
    n_tok = tl.minimum(tl.maximum(rem, 0), GROUP)
    if n_tok <= 0:
        return

    block_id = tl.load(Block_table_ptr + b * stride_bt_b + k)
    dst_base = tl.load(Cu_seqlens_ptr + b).to(tl.int64) + k.to(tl.int64) * GROUP

    d_offs = tl.arange(0, D)
    g_offs = tl.arange(0, GROUP)
    g_mask = g_offs < n_tok

    in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
    safe_bid = tl.where(in_range, block_id, 0)
    pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)

    out_addrs = (
        (dst_base + g_offs)[:, None] * stride_out_t
        + hk * stride_out_h
        + d_offs[None, :]
    )

    if pool_slot >= 0:
        pool_base = pool_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h
        src_addrs = pool_base + g_offs[:, None] * stride_pool_t + d_offs[None, :]
        K_chunk = tl.load(Tail_K_pool_ptr + src_addrs, mask=g_mask[:, None], other=0.0)
        V_chunk = tl.load(Tail_V_pool_ptr + src_addrs, mask=g_mask[:, None], other=0.0)
        tl.store(K_out_ptr + out_addrs, K_chunk, mask=g_mask[:, None])
        tl.store(V_out_ptr + out_addrs, V_chunk, mask=g_mask[:, None])
    else:
        PACK_K: tl.constexpr = 8 // K_BITS
        PACK_V: tl.constexpr = 8 // V_BITS
        MASK_K: tl.constexpr = (1 << K_BITS) - 1
        MASK_V: tl.constexpr = (1 << V_BITS) - 1
        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        g_byte_k = g_offs // PACK_K
        g_shift_k = (g_offs % PACK_K) * K_BITS
        d_byte_v = d_offs // PACK_V
        d_shift_v = (d_offs % PACK_V) * V_BITS

        sk_lo = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        sk_hi = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        s_col_K = ((sk_lo | (sk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zk_lo = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        zk_hi = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        zp_K = ((zk_lo | (zk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        srk_lo = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + g_offs * 2).to(
            tl.uint16
        )
        srk_hi = tl.load(KV_cache_ptr + tile_base + K_S_ROW_OFFSET + g_offs * 2 + 1).to(
            tl.uint16
        )
        s_row_K = ((srk_lo | (srk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        k_addrs = (
            tile_base
            + K_PACKED_OFFSET
            + d_offs[:, None] * (GROUP // PACK_K)
            + g_byte_k[None, :]
        )
        k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
        q_K = ((k_bytes >> g_shift_k[None, :]) & MASK_K).to(tl.float32)
        K_rot = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
        K_rot_out = tl.trans(K_rot)

        scv_lo = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        scv_hi = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        s_col_V = ((scv_lo | (scv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        srv_lo = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + g_offs * 2).to(
            tl.uint16
        )
        srv_hi = tl.load(KV_cache_ptr + tile_base + V_S_ROW_OFFSET + g_offs * 2 + 1).to(
            tl.uint16
        )
        s_row_V = ((srv_lo | (srv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zpv_lo = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + g_offs * 2).to(
            tl.uint16
        )
        zpv_hi = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + g_offs * 2 + 1).to(
            tl.uint16
        )
        zp_V = ((zpv_lo | (zpv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        v_addrs = (
            tile_base
            + V_PACKED_OFFSET
            + g_offs[:, None] * (D // PACK_V)
            + d_byte_v[None, :]
        )
        v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)
        q_V = ((v_bytes >> d_shift_v[None, :]) & MASK_V).to(tl.float32)
        V_rot = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]

        tl.store(K_out_ptr + out_addrs, K_rot_out.to(tl.float16), mask=g_mask[:, None])
        tl.store(V_out_ptr + out_addrs, V_rot.to(tl.float16), mask=g_mask[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# Fused decode kernel: dequant int4 in registers + online-softmax flash-decode.
# ──────────────────────────────────────────────────────────────────────────────


@_maybe_autotune
@triton.jit
def _kvarn_fused_decode_kernel(
    Q_ptr,  # [B, Hq, D]                               fp16 (rotated)
    Req_row_ptr,  # [B] int32 — block-table row per program row
    Block_table_ptr,  # [B, max_blocks]                          int32
    Seq_lens_ptr,  # [B]                                      int32
    Block_to_slot_ptr,  # [num_blocks_lookup]                      int32 (-1 = int4)
    KV_cache_ptr,  # [num_blocks, num_kv_heads, TILE_BYTES]   uint8
    Tail_K_pool_ptr,  # [POOL_SIZE, group, Hk, D]                fp16 (rotated)
    Tail_V_pool_ptr,  # [POOL_SIZE, group, Hk, D]                fp16
    Out_ptr,  # [B, Hq, D]                               fp16 (rotated out)
    scale,
    # strides
    stride_q_b,
    stride_q_h,
    stride_bt_b,
    stride_kv_b,
    stride_kv_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_o_b,
    stride_o_h,
    # constexprs
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    Q_PER_KV_PAD: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
    K_PACKED_OFFSET: tl.constexpr,
    K_S_COL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr,
    K_S_ROW_OFFSET: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr,
    V_S_COL_OFFSET: tl.constexpr,
    V_S_ROW_OFFSET: tl.constexpr,
    V_ZP_OFFSET: tl.constexpr,
    VQ_INDIRECT: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    qh = tl.arange(0, Q_PER_KV_PAD)
    qmask = qh < Q_PER_KV
    hq0 = hk * Q_PER_KV

    bt_row = b
    if VQ_INDIRECT:
        bt_row = tl.load(Req_row_ptr + b)
    seq_len = tl.load(Seq_lens_ptr + b)
    if seq_len <= 0:
        return

    d_offs = tl.arange(0, D)
    PACK_K: tl.constexpr = 8 // K_BITS
    PACK_V: tl.constexpr = 8 // V_BITS
    MASK_K: tl.constexpr = (1 << K_BITS) - 1
    MASK_V: tl.constexpr = (1 << V_BITS) - 1
    d_byte_v = d_offs // PACK_V
    d_shift_v = (d_offs % PACK_V) * V_BITS

    q = tl.load(
        Q_ptr + b * stride_q_b + (hq0 + qh)[:, None] * stride_q_h + d_offs[None, :],
        mask=qmask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([Q_PER_KV_PAD], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_PER_KV_PAD], dtype=tl.float32)
    acc = tl.zeros([Q_PER_KV_PAD, D], dtype=tl.float32)

    n_blocks = (seq_len + GROUP - 1) // GROUP
    win_start = 0
    blk_lo = 0
    if SLIDING_WINDOW > 0:
        win_start = tl.maximum(seq_len - SLIDING_WINDOW, 0)
        blk_lo = win_start // GROUP
    for k in range(blk_lo, n_blocks):
        rem = seq_len - k * GROUP
        n_tok = tl.minimum(tl.maximum(rem, 0), GROUP)

        block_id = tl.load(Block_table_ptr + bt_row * stride_bt_b + k)
        in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
        safe_bid = tl.where(in_range, block_id, 0)
        pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)

        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        safe_slot = tl.where(pool_slot >= 0, pool_slot, 0)
        pool_base = safe_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h

        sk_lo = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        sk_hi = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        s_col_K = ((sk_lo | (sk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zk_lo = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        zk_hi = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        zp_K = ((zk_lo | (zk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        scv_lo = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        scv_hi = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        s_col_V = ((scv_lo | (scv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        for c0 in range(0, GROUP, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)
            cmask = cols < n_tok
            if SLIDING_WINDOW > 0:
                cmask = cmask & ((k * GROUP + cols) >= win_start)

            if pool_slot >= 0:
                src = pool_base + cols[:, None] * stride_pool_t + d_offs[None, :]
                Kc = tl.load(Tail_K_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )
                Vc = tl.load(Tail_V_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )
                K_dg = tl.trans(Kc)
            else:
                cb_k = cols // PACK_K
                cs_k = (cols % PACK_K) * K_BITS
                srk_lo = tl.load(
                    KV_cache_ptr + tile_base + K_S_ROW_OFFSET + cols * 2
                ).to(tl.uint16)
                srk_hi = tl.load(
                    KV_cache_ptr + tile_base + K_S_ROW_OFFSET + cols * 2 + 1
                ).to(tl.uint16)
                s_row_K = ((srk_lo | (srk_hi << 8)).to(tl.float16, bitcast=True)).to(
                    tl.float32
                )
                k_addrs = (
                    tile_base
                    + K_PACKED_OFFSET
                    + d_offs[:, None] * (GROUP // PACK_K)
                    + cb_k[None, :]
                )
                k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
                q_K = ((k_bytes >> cs_k[None, :]) & MASK_K).to(tl.float32)
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]

                srv_lo = tl.load(
                    KV_cache_ptr + tile_base + V_S_ROW_OFFSET + cols * 2
                ).to(tl.uint16)
                srv_hi = tl.load(
                    KV_cache_ptr + tile_base + V_S_ROW_OFFSET + cols * 2 + 1
                ).to(tl.uint16)
                s_row_V = ((srv_lo | (srv_hi << 8)).to(tl.float16, bitcast=True)).to(
                    tl.float32
                )
                zpv_lo = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + cols * 2).to(
                    tl.uint16
                )
                zpv_hi = tl.load(
                    KV_cache_ptr + tile_base + V_ZP_OFFSET + cols * 2 + 1
                ).to(tl.uint16)
                zp_V = ((zpv_lo | (zpv_hi << 8)).to(tl.float16, bitcast=True)).to(
                    tl.float32
                )
                v_addrs = (
                    tile_base
                    + V_PACKED_OFFSET
                    + cols[:, None] * (D // PACK_V)
                    + d_byte_v[None, :]
                )
                v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)
                q_V = ((v_bytes >> d_shift_v[None, :]) & MASK_V).to(tl.float32)
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]

            scores = tl.dot(q, K_dg)
            scores = tl.where(cmask[None, :], scores * scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, Vc)
            m_i = m_new

    out = (acc / l_i[:, None]).to(tl.float16)
    tl.store(
        Out_ptr + b * stride_o_b + (hq0 + qh)[:, None] * stride_o_h + d_offs[None, :],
        out,
        mask=qmask[:, None],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Split-K stage 1 + stage 2
# ──────────────────────────────────────────────────────────────────────────────


@triton.jit
def _kvarn_fused_decode_stage1(
    Q_ptr,
    Req_row_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Block_to_slot_ptr,
    KV_cache_ptr,
    Tail_K_pool_ptr,
    Tail_V_pool_ptr,
    MidO_ptr,  # [N, NUM_KV_SPLITS, D]            fp32
    MidLse_ptr,  # [N, NUM_KV_SPLITS]               fp32
    scale,
    stride_q_b,
    stride_q_h,
    stride_bt_b,
    stride_kv_b,
    stride_kv_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_mo_n,
    stride_mo_s,
    stride_ml_n,
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    HQ: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    Q_PER_KV_PAD: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
    K_PACKED_OFFSET: tl.constexpr,
    K_S_COL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr,
    K_S_ROW_OFFSET: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr,
    V_S_COL_OFFSET: tl.constexpr,
    V_S_ROW_OFFSET: tl.constexpr,
    V_ZP_OFFSET: tl.constexpr,
    VQ_INDIRECT: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    split = tl.program_id(2)
    qh = tl.arange(0, Q_PER_KV_PAD)
    qmask = qh < Q_PER_KV
    hq0 = hk * Q_PER_KV

    bt_row = b
    if VQ_INDIRECT:
        bt_row = tl.load(Req_row_ptr + b)
    seq_len = tl.load(Seq_lens_ptr + b)
    n_blocks = (seq_len + GROUP - 1) // GROUP
    blocks_per_split = (n_blocks + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
    blk_lo = split * blocks_per_split
    blk_hi = tl.minimum(blk_lo + blocks_per_split, n_blocks)

    d_offs = tl.arange(0, D)
    PACK_K: tl.constexpr = 8 // K_BITS
    PACK_V: tl.constexpr = 8 // V_BITS
    MASK_K: tl.constexpr = (1 << K_BITS) - 1
    MASK_V: tl.constexpr = (1 << V_BITS) - 1
    d_byte_v = d_offs // PACK_V
    d_shift_v = (d_offs % PACK_V) * V_BITS
    q = tl.load(
        Q_ptr + b * stride_q_b + (hq0 + qh)[:, None] * stride_q_h + d_offs[None, :],
        mask=qmask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([Q_PER_KV_PAD], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_PER_KV_PAD], dtype=tl.float32)
    acc = tl.zeros([Q_PER_KV_PAD, D], dtype=tl.float32)

    for k in range(blk_lo, blk_hi):
        rem = seq_len - k * GROUP
        n_tok = tl.minimum(tl.maximum(rem, 0), GROUP)
        block_id = tl.load(Block_table_ptr + bt_row * stride_bt_b + k)
        in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
        safe_bid = tl.where(in_range, block_id, 0)
        pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)
        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        safe_slot = tl.where(pool_slot >= 0, pool_slot, 0)
        pool_base = safe_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h

        sk_lo = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        sk_hi = tl.load(KV_cache_ptr + tile_base + K_S_COL_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        s_col_K = ((sk_lo | (sk_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)
        zk_lo = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        zk_hi = tl.load(KV_cache_ptr + tile_base + K_ZP_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        zp_K = (zk_lo | (zk_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        scv_lo = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2).to(
            tl.uint16
        )
        scv_hi = tl.load(KV_cache_ptr + tile_base + V_S_COL_OFFSET + d_offs * 2 + 1).to(
            tl.uint16
        )
        s_col_V = ((scv_lo | (scv_hi << 8)).to(tl.float16, bitcast=True)).to(tl.float32)

        for c0 in range(0, GROUP, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)
            cmask = cols < n_tok
            if SLIDING_WINDOW > 0:
                cmask = cmask & (
                    (k * GROUP + cols) >= tl.maximum(seq_len - SLIDING_WINDOW, 0)
                )
            if pool_slot >= 0:
                src = pool_base + cols[:, None] * stride_pool_t + d_offs[None, :]
                Kc = tl.load(Tail_K_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )
                Vc = tl.load(Tail_V_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )
                K_dg = tl.trans(Kc)
            else:
                cb_k = cols // PACK_K
                cs_k = (cols % PACK_K) * K_BITS
                srk_lo = tl.load(
                    KV_cache_ptr + tile_base + K_S_ROW_OFFSET + cols * 2
                ).to(tl.uint16)
                srk_hi = tl.load(
                    KV_cache_ptr + tile_base + K_S_ROW_OFFSET + cols * 2 + 1
                ).to(tl.uint16)
                s_row_K = ((srk_lo | (srk_hi << 8)).to(tl.float16, bitcast=True)).to(
                    tl.float32
                )
                k_addrs = (
                    tile_base
                    + K_PACKED_OFFSET
                    + d_offs[:, None] * (GROUP // PACK_K)
                    + cb_k[None, :]
                )
                k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
                q_K = ((k_bytes >> cs_k[None, :]) & MASK_K).to(tl.float32)
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                srv_lo = tl.load(
                    KV_cache_ptr + tile_base + V_S_ROW_OFFSET + cols * 2
                ).to(tl.uint16)
                srv_hi = tl.load(
                    KV_cache_ptr + tile_base + V_S_ROW_OFFSET + cols * 2 + 1
                ).to(tl.uint16)
                s_row_V = ((srv_lo | (srv_hi << 8)).to(tl.float16, bitcast=True)).to(
                    tl.float32
                )
                zpv_lo = tl.load(KV_cache_ptr + tile_base + V_ZP_OFFSET + cols * 2).to(
                    tl.uint16
                )
                zpv_hi = tl.load(
                    KV_cache_ptr + tile_base + V_ZP_OFFSET + cols * 2 + 1
                ).to(tl.uint16)
                zp_V = ((zpv_lo | (zpv_hi << 8)).to(tl.float16, bitcast=True)).to(
                    tl.float32
                )
                v_addrs = (
                    tile_base
                    + V_PACKED_OFFSET
                    + cols[:, None] * (D // PACK_V)
                    + d_byte_v[None, :]
                )
                v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)
                q_V = ((v_bytes >> d_shift_v[None, :]) & MASK_V).to(tl.float32)
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]

            scores = tl.dot(q, K_dg)
            scores = tl.where(cmask[None, :], scores * scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, Vc)
            m_i = m_new

    nonempty = l_i > 0
    O_s = acc / tl.where(nonempty, l_i, 1.0)[:, None]
    lse_s = tl.where(
        nonempty, m_i + tl.log(tl.where(nonempty, l_i, 1.0)), -float("inf")
    )
    rows = b * HQ + hq0 + qh
    tl.store(
        MidO_ptr + rows[:, None] * stride_mo_n + split * stride_mo_s + d_offs[None, :],
        O_s,
        mask=qmask[:, None],
    )
    tl.store(MidLse_ptr + rows * stride_ml_n + split, lse_s, mask=qmask)


@triton.jit
def _kvarn_fused_decode_stage2(
    MidO_ptr,  # [N, NUM_KV_SPLITS, D] fp32
    MidLse_ptr,  # [N, NUM_KV_SPLITS]    fp32
    Out_ptr,  # [N, D] fp16
    stride_mo_n,
    stride_mo_s,
    stride_ml_n,
    stride_o_n,
    D: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    n = tl.program_id(0)
    d_offs = tl.arange(0, D)
    s_offs = tl.arange(0, NUM_KV_SPLITS)
    lse = tl.load(MidLse_ptr + n * stride_ml_n + s_offs)
    g = tl.max(lse, axis=0)
    w = tl.exp(lse - g)
    denom = tl.sum(w, axis=0)
    O = tl.load(
        MidO_ptr + n * stride_mo_n + s_offs[:, None] * stride_mo_s + d_offs[None, :]
    )
    out = tl.sum(w[:, None] * O, axis=0) / denom
    tl.store(Out_ptr + n * stride_o_n + d_offs, out.to(tl.float16))


# ──────────────────────────────────────────────────────────────────────────────
# Python driver: kvarn_decode_attention
# ──────────────────────────────────────────────────────────────────────────────


def kvarn_decode_attention(
    query: torch.Tensor,  # [B, Hq, D]   fp16/bf16
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, TILE_BYTES] uint8
    tail_K: torch.Tensor,  # [POOL_SIZE, group, Hk, D] fp16 (layer-specific)
    tail_V: torch.Tensor,  # [POOL_SIZE, group, Hk, D] fp16 (layer-specific)
    hadamard: torch.Tensor,  # [D, D]       fp32
    scale: float,
    cfg,
    impl,  # KVarNAttnBackend (has block_to_slot_t + strides)
    block_table: torch.Tensor,  # [B, max_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    cu_seqlens: torch.Tensor,  # [B+1] int32 (prefix sum)
    sliding_window: int = 0,  # 0 = no sliding window
) -> torch.Tensor:
    """Fused decode driver — dequant int4 in registers + online-softmax.

    Never materializes a full fp16 K/V buffer to HBM. Moves ~int4 (0.25x FP16)
    KV traffic for the bulk history.

    Output: [B, Hq, D] in query's dtype, un-rotated frame.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[1]
    device = query.device
    out_dtype = query.dtype
    group = cfg.group
    N = B * Hq

    # 1. Q rotation — fp16 matmul
    # Use pre-allocated buffer when available (CUDA graph capture-safe)
    H16 = hadamard.to(torch.float16)
    if hasattr(impl, "_q_rot_fp16_buf") and impl._q_rot_fp16_buf is not None:
        q_rot_fp16 = impl._q_rot_fp16_buf[:N]
        torch.mm(query.reshape(N, D).to(torch.float16), H16, out=q_rot_fp16)
    else:
        q_rot_fp16 = torch.mm(query.reshape(N, D).to(torch.float16), H16)

    # 2. Fused decode kernel
    max_blocks_per_req = block_table.shape[1]
    _qpk = Hq // Hk
    _qpk_pad = 1 << (_qpk - 1).bit_length() if _qpk > 1 else 1

    fused_out = torch.empty(N, D, dtype=torch.float16, device=device)

    use_fused = True

    common = dict(
        MAX_BLOCKS_PER_REQ=max_blocks_per_req,
        D=D,
        GROUP=group,
        Q_PER_KV=_qpk,
        Q_PER_KV_PAD=_qpk_pad,
        SLIDING_WINDOW=sliding_window,
        K_BITS=cfg.key_bits,
        V_BITS=cfg.value_bits,
        NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
        K_PACKED_OFFSET=cfg.k_packed_offset,
        K_S_COL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset,
        K_S_ROW_OFFSET=cfg.k_s_row_offset,
        V_PACKED_OFFSET=cfg.v_packed_offset,
        V_S_COL_OFFSET=cfg.v_s_col_offset,
        V_S_ROW_OFFSET=cfg.v_s_row_offset,
        V_ZP_OFFSET=cfg.v_zp_offset,
        VQ_INDIRECT=False,
    )

    if use_fused:
        # Split-K (two-stage flash-decoding): only a win in the LOW-batch /
        # long-context regime (few (B,Hk) programs → the KV-split dim adds the
        # missing parallelism). At BURST (high batch) the single-stage (B,Hk)
        # grid already saturates the GPU.
        #
        # Auto-enable when context is long (>= 16 blocks) AND the single-stage
        # grid (B*Hk) doesn't already fill the SMs.
        #  Sliding-window layers read only
        # ~window/GROUP blocks so never split them.
        _sw = int(getattr(impl, "sliding_window", 0) or 0) or sliding_window
        sm_count = (
            getattr(impl, "_sm_count", 0)
            or torch.cuda.get_device_properties(device).multi_processor_count
        )
        use_split = (_sw <= 0) and (max_blocks_per_req >= 16) and (B * Hk <= sm_count)
        if use_split:
            kv_splits = adaptive_num_kv_splits(max_blocks_per_req)
            # Use pre-allocated buffers (CUDA graph capture-safe)
            mid_O = impl._mid_o_buf[:N, :kv_splits, :]
            mid_lse = impl._mid_lse_buf[:N, :kv_splits]
            fused_out = impl._fused_out_buf[:N]
            _bn = 16
            _nw = 4
            _ns = 2
            _kvarn_fused_decode_stage1[(B, Hk, kv_splits)](
                q_rot_fp16.view(B, Hq, D),
                seq_lens,
                block_table,
                seq_lens,
                impl._block_to_slot_t,
                kv_cache,
                tail_K,
                tail_V,
                mid_O,
                mid_lse,
                scale,
                Hq * D,
                D,
                block_table.stride(0),
                kv_cache.stride(0),
                kv_cache.stride(1),
                impl._tail_K_stride0,
                impl._tail_K_stride1,
                impl._tail_K_stride2,
                mid_O.stride(0),
                mid_O.stride(1),
                mid_lse.stride(0),
                MAX_BLOCKS_PER_REQ=max_blocks_per_req,
                D=D,
                GROUP=group,
                BLOCK_N=_bn,
                Q_PER_KV=_qpk,
                Q_PER_KV_PAD=_qpk_pad,
                NUM_KV_SPLITS=kv_splits,
                HQ=Hq,
                SLIDING_WINDOW=sliding_window,
                K_BITS=cfg.key_bits,
                V_BITS=cfg.value_bits,
                NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
                K_PACKED_OFFSET=cfg.k_packed_offset,
                K_S_COL_OFFSET=cfg.k_s_col_offset,
                K_ZP_OFFSET=cfg.k_zp_offset,
                K_S_ROW_OFFSET=cfg.k_s_row_offset,
                V_PACKED_OFFSET=cfg.v_packed_offset,
                V_S_COL_OFFSET=cfg.v_s_col_offset,
                V_S_ROW_OFFSET=cfg.v_s_row_offset,
                V_ZP_OFFSET=cfg.v_zp_offset,
                VQ_INDIRECT=False,
                num_warps=_nw,
                num_stages=_ns,
            )
            _kvarn_fused_decode_stage2[(N,)](
                mid_O,
                mid_lse,
                fused_out,
                mid_O.stride(0),
                mid_O.stride(1),
                mid_lse.stride(0),
                fused_out.stride(0),
                D=D,
                NUM_KV_SPLITS=kv_splits,
                num_warps=2,
            )
            output_rot = fused_out
        else:
            # Single-stage fused decode: (B, Hk) grid, online softmax
            _kvarn_fused_decode_kernel[(B, Hk)](
                q_rot_fp16.view(B, Hq, D),
                seq_lens,
                block_table,
                seq_lens,
                impl._block_to_slot_t,
                kv_cache,
                tail_K,
                tail_V,
                fused_out.view(B, Hq, D),
                scale,
                Hq * D,
                D,
                block_table.stride(0),
                kv_cache.stride(0),
                kv_cache.stride(1),
                impl._tail_K_stride0,
                impl._tail_K_stride1,
                impl._tail_K_stride2,
                Hq * D,
                D,
                **common,
            )
            output_rot = fused_out
    else:
        # Materialize path: build packed fp16 K/V then SDPA
        total_tokens = cu_seqlens[-1].item()
        K_packed = torch.empty(total_tokens, Hk, D, dtype=torch.float16, device=device)
        V_packed = torch.empty(total_tokens, Hk, D, dtype=torch.float16, device=device)

        _kvarn_build_packed_kv_kernel[(B * max_blocks_per_req, Hk)](
            block_table,
            seq_lens,
            cu_seqlens,
            impl._block_to_slot_t,
            kv_cache,
            tail_K,
            tail_V,
            K_packed,
            V_packed,
            block_table.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            impl._tail_K_stride0,
            impl._tail_K_stride1,
            impl._tail_K_stride2,
            K_packed.stride(0),
            K_packed.stride(1),
            MAX_BLOCKS_PER_REQ=max_blocks_per_req,
            D=D,
            GROUP=group,
            K_BITS=cfg.key_bits,
            V_BITS=cfg.value_bits,
            NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
            K_PACKED_OFFSET=cfg.k_packed_offset,
            K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset,
            K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset,
            V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset,
            V_ZP_OFFSET=cfg.v_zp_offset,
            num_warps=4,
            num_stages=2,
        )
        # Use SDPA for the attention
        output_rot = torch.empty(B, Hq, D, dtype=torch.float16, device=device)
        for i in range(B):
            seq_len = int(seq_lens[i].item())
            q_i = (
                q_rot_fp16.view(B, Hq, D)[i : i + 1]
                .transpose(0, 1)
                .unsqueeze(0)
                .float()
            )
            K_t = (
                K_packed[cu_seqlens[i] : cu_seqlens[i] + seq_len]
                .transpose(0, 1)
                .unsqueeze(0)
                .float()
            )
            V_t = (
                V_packed[cu_seqlens[i] : cu_seqlens[i] + seq_len]
                .transpose(0, 1)
                .unsqueeze(0)
                .float()
            )
            o = F.scaled_dot_product_attention(
                q_i,
                K_t,
                V_t,
                is_causal=False,
                scale=scale,
                enable_gqa=Hk < Hq,
            )
            output_rot[i] = o[0, :, 0, :].to(torch.float16)

    # 3. Un-rotate output
    out_unrot = torch.mm(output_rot.reshape(N, D), H16)
    return out_unrot.view(B, Hq, D).to(out_dtype)


def kvarn_scatter_store(
    k_rot: torch.Tensor,  # [N, Hk, D] fp16 (already rotated)
    v_rot: torch.Tensor,  # [N, Hk, D] fp16
    slot_mapping: torch.Tensor,  # [N] int32 (out_cache_loc)
    block_to_slot_t: torch.Tensor,  # [num_blocks_lookup] int32
    pool_K: torch.Tensor,  # [POOL_SIZE, group, Hk, D] fp16
    pool_V: torch.Tensor,  # [POOL_SIZE, group, Hk, D] fp16
    group: int,
    D: int,
    num_blocks_lookup: int,
):
    """Scatter already-rotated K/V into the tail pool via block_to_slot lookup."""
    N = k_rot.shape[0]
    Hk = k_rot.shape[1]
    _kvarn_scatter_store_kernel[(N, Hk)](
        k_rot,
        v_rot,
        slot_mapping,
        block_to_slot_t,
        pool_K,
        pool_V,
        k_rot.stride(0),
        k_rot.stride(1),
        pool_K.stride(0),
        pool_K.stride(1),
        pool_K.stride(2),
        GROUP=group,
        D=D,
        NUM_BLOCKS_LOOKUP=num_blocks_lookup,
        num_warps=2,
        num_stages=2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Speculative decode verify driver: kvarn_verify_attention
# ──────────────────────────────────────────────────────────────────────────────


def kvarn_verify_attention(
    query: torch.Tensor,  # [NQ, Hq, D]  fp16/bf16 (token-major)
    kv_cache: torch.Tensor,  # [num_blocks, Hk, TILE_BYTES] uint8
    tail_K: torch.Tensor,  # [POOL_SIZE, group, Hk, D] fp16
    tail_V: torch.Tensor,  # [POOL_SIZE, group, Hk, D] fp16
    hadamard: torch.Tensor,  # [D, D] fp32
    scale: float,
    cfg,
    impl,  # KVarNAttnBackend
    block_table: torch.Tensor,  # [B, max_blocks] int32
    vq_req: torch.Tensor,  # [NQ] int32 — block-table row per token
    vq_seqlen: torch.Tensor,  # [NQ] int32 — causal len: cached+i+1
    max_ctx_blocks: int,  # ceil(max context / group) upper bound
    sliding_window: int = 0,
) -> torch.Tensor:
    """Fused multi-query verify (speculative decode), reading int4 tiles +
    the fp16 tail pool directly — no fp16 materialization of the context.

    Each query token becomes a virtual kernel row with its own causal length
    and block-table row indirection (VQ_INDIRECT=True).

    Output: [NQ, Hq, D] in query's dtype, un-rotated frame.
    """
    NQ, Hq, D = query.shape
    Hk = kv_cache.shape[1]
    device = query.device
    out_dtype = query.dtype
    group = cfg.group
    Nrows = NQ * Hq

    H16 = hadamard.to(torch.float16)
    q_rot = torch.mm(query.reshape(Nrows, D).to(torch.float16), H16)

    _qpk = Hq // Hk
    _qpk_pad = 1 << (_qpk - 1).bit_length() if _qpk > 1 else 1

    common = dict(
        MAX_BLOCKS_PER_REQ=max_ctx_blocks,
        D=D,
        GROUP=group,
        Q_PER_KV=_qpk,
        Q_PER_KV_PAD=_qpk_pad,
        SLIDING_WINDOW=sliding_window,
        K_BITS=cfg.key_bits,
        V_BITS=cfg.value_bits,
        NUM_BLOCKS_LOOKUP=impl._block_lookup_size,
        K_PACKED_OFFSET=cfg.k_packed_offset,
        K_S_COL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset,
        K_S_ROW_OFFSET=cfg.k_s_row_offset,
        V_PACKED_OFFSET=cfg.v_packed_offset,
        V_S_COL_OFFSET=cfg.v_s_col_offset,
        V_S_ROW_OFFSET=cfg.v_s_row_offset,
        V_ZP_OFFSET=cfg.v_zp_offset,
        VQ_INDIRECT=True,
    )

    out_rot = torch.empty(NQ, Hq, D, dtype=torch.float16, device=device)

    _kvarn_fused_decode_kernel[(NQ, Hk)](
        q_rot,
        vq_req,
        block_table,
        vq_seqlen,
        impl._block_to_slot_t,
        kv_cache,
        tail_K,
        tail_V,
        out_rot,
        scale,
        Hq * D,
        D,
        block_table.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        impl._tail_K_stride0,
        impl._tail_K_stride1,
        impl._tail_K_stride2,
        Hq * D,
        D,
        **common,
    )

    out_unrot = torch.mm(out_rot.reshape(Nrows, D), H16)
    return out_unrot.view(NQ, Hq, D).to(out_dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Shared-dequant verify kernel: one program per (REQUEST, kv-head, split)
# All QLEN verify tokens of a request share each block's dequant.
# Q tile is [QLEN * Q_PER_KV_PAD, D] with per-row causal limit.
# ──────────────────────────────────────────────────────────────────────────────


@_maybe_autotune_verify
@triton.jit
def _kvarn_fused_verify_stage1(
    Q_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Block_to_slot_ptr,
    KV_cache_ptr,
    Tail_K_pool_ptr,
    Tail_V_pool_ptr,
    MidO_ptr,
    MidLse_ptr,
    scale,
    stride_q_t,
    stride_q_h,
    stride_bt_b,
    stride_kv_b,
    stride_kv_h,
    stride_pool_b,
    stride_pool_t,
    stride_pool_h,
    stride_mo_n,
    stride_mo_s,
    stride_ml_n,
    MAX_BLOCKS_PER_REQ: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QLEN: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    Q_PER_KV_PAD: tl.constexpr,
    HQ: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
    K_PACKED_OFFSET: tl.constexpr,
    K_S_COL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr,
    K_S_ROW_OFFSET: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr,
    V_S_COL_OFFSET: tl.constexpr,
    V_S_ROW_OFFSET: tl.constexpr,
    V_ZP_OFFSET: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    split = tl.program_id(2)

    seq_len = tl.load(Seq_lens_ptr + b * QLEN + (QLEN - 1))
    if seq_len <= 0:
        return

    M: tl.constexpr = QLEN * Q_PER_KV_PAD
    r = tl.arange(0, M)
    j = r // Q_PER_KV_PAD
    lane = r % Q_PER_KV_PAD
    rmask = lane < Q_PER_KV
    limit = seq_len - QLEN + j + 1
    hq0 = hk * Q_PER_KV
    d_offs = tl.arange(0, D)

    PACK_K: tl.constexpr = 8 // K_BITS
    PACK_V: tl.constexpr = 8 // V_BITS
    MASK_K: tl.constexpr = (1 << K_BITS) - 1
    MASK_V: tl.constexpr = (1 << V_BITS) - 1
    d_byte_v = d_offs // PACK_V
    d_shift_v = (d_offs % PACK_V) * V_BITS

    tok_row = b * QLEN + j
    q = tl.load(
        Q_ptr
        + tok_row[:, None] * stride_q_t
        + (hq0 + lane)[:, None] * stride_q_h
        + d_offs[None, :],
        mask=rmask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, D], dtype=tl.float32)

    n_blocks = (seq_len + GROUP - 1) // GROUP
    blocks_per_split = (n_blocks + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS
    blk_lo = split * blocks_per_split
    blk_hi = tl.minimum(blk_lo + blocks_per_split, n_blocks)

    for k in range(blk_lo, blk_hi):
        rem = seq_len - k * GROUP
        n_tok = tl.minimum(tl.maximum(rem, 0), GROUP)
        block_id = tl.load(Block_table_ptr + b * stride_bt_b + k)
        in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
        safe_bid = tl.where(in_range, block_id, 0)
        pool_slot = tl.load(Block_to_slot_ptr + safe_bid, mask=in_range, other=-1)
        tile_base = block_id.to(tl.int64) * stride_kv_b + hk * stride_kv_h
        safe_slot = tl.where(pool_slot >= 0, pool_slot, 0)
        pool_base = safe_slot.to(tl.int64) * stride_pool_b + hk * stride_pool_h

        s_col_K = tl.load(
            (KV_cache_ptr + tile_base + K_S_COL_OFFSET).to(tl.pointer_type(tl.float16))
            + d_offs
        ).to(tl.float32)
        zp_K = tl.load(
            (KV_cache_ptr + tile_base + K_ZP_OFFSET).to(tl.pointer_type(tl.float16))
            + d_offs
        ).to(tl.float32)
        s_col_V = tl.load(
            (KV_cache_ptr + tile_base + V_S_COL_OFFSET).to(tl.pointer_type(tl.float16))
            + d_offs
        ).to(tl.float32)

        for c0 in range(0, GROUP, BLOCK_N):
            cols = c0 + tl.arange(0, BLOCK_N)
            cmask = cols < n_tok
            kvpos = k * GROUP + cols

            if pool_slot >= 0:
                src = pool_base + cols[:, None] * stride_pool_t + d_offs[None, :]
                Kc = tl.load(Tail_K_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )
                Vc = tl.load(Tail_V_pool_ptr + src, mask=cmask[:, None], other=0.0).to(
                    tl.float32
                )
                K_dg = tl.trans(Kc)
            else:
                cb_k = cols // PACK_K
                cs_k = (cols % PACK_K) * K_BITS
                s_row_K = tl.load(
                    (KV_cache_ptr + tile_base + K_S_ROW_OFFSET).to(
                        tl.pointer_type(tl.float16)
                    )
                    + cols
                ).to(tl.float32)
                k_addrs = (
                    tile_base
                    + K_PACKED_OFFSET
                    + d_offs[:, None] * (GROUP // PACK_K)
                    + cb_k[None, :]
                )
                k_bytes = tl.load(KV_cache_ptr + k_addrs).to(tl.int32)
                q_K = ((k_bytes >> cs_k[None, :]) & MASK_K).to(tl.float32)
                K_dg = (q_K * s_col_K[:, None] + zp_K[:, None]) * s_row_K[None, :]
                s_row_V = tl.load(
                    (KV_cache_ptr + tile_base + V_S_ROW_OFFSET).to(
                        tl.pointer_type(tl.float16)
                    )
                    + cols
                ).to(tl.float32)
                zp_V = tl.load(
                    (KV_cache_ptr + tile_base + V_ZP_OFFSET).to(
                        tl.pointer_type(tl.float16)
                    )
                    + cols
                ).to(tl.float32)
                v_addrs = (
                    tile_base
                    + V_PACKED_OFFSET
                    + cols[:, None] * (D // PACK_V)
                    + d_byte_v[None, :]
                )
                v_bytes = tl.load(KV_cache_ptr + v_addrs).to(tl.int32)
                q_V = ((v_bytes >> d_shift_v[None, :]) & MASK_V).to(tl.float32)
                Vc = (q_V * s_row_V[:, None] + zp_V[:, None]) * s_col_V[None, :]

            scores = tl.dot(q, K_dg)
            smask = cmask[None, :] & (kvpos[None, :] < limit[:, None])
            if SLIDING_WINDOW > 0:
                smask = smask & (
                    kvpos[None, :] >= tl.maximum(limit[:, None] - SLIDING_WINDOW, 0)
                )
            scores = tl.where(smask, scores * scale, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, Vc)
            m_i = m_new

    nonempty = l_i > 0
    O_s = acc / tl.where(nonempty, l_i, 1.0)[:, None]
    lse_s = tl.where(
        nonempty, m_i + tl.log(tl.where(nonempty, l_i, 1.0)), -float("inf")
    )
    rows = tok_row * HQ + hq0 + lane
    tl.store(
        MidO_ptr + rows[:, None] * stride_mo_n + split * stride_mo_s + d_offs[None, :],
        O_s,
        mask=rmask[:, None],
    )
    tl.store(MidLse_ptr + rows * stride_ml_n + split, lse_s, mask=rmask)
