"""Triton ports of HISA kernels.

Each kernel here mirrors a tilelang kernel in ``hisa/custom_ops.py`` so we
can microbench the two implementations on identical inputs.

Coverage (so far):
  1. ``batch_pool_mqa_triton`` — triton port of
     ``batch_decode_pool_mqa_attn_return_logits_fp8``. Contiguous blocked_k
     input, fp8×fp8 MQA. Simpler pattern, no paged indirection.

Pending (add next):
  2. ``sparse_paged_mqa_triton`` — port of
     ``fp8_native_paged_block_sparse_mqa_attn_return_logits``. The 80% hotspot.

Notes on the fp8 path:
  - Requires ``triton >= 3.0`` for ``tl.dot`` on fp8 operands.
  - fp8 scales are per-K-block (one f32 per K-block), applied **after** the
    GEMM as a scalar multiply — same as tilelang.
  - ``tl.dot`` needs the accumulator and one operand in ``float32``/``tf32``
    depending on arch; fp8 × fp8 → fp32 accumulator is supported on H100+.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: contiguous block-MQA (fp8 Q × fp8 K + per-block-scale + weights)
#   Mirrors: batch_decode_pool_mqa_attn_return_logits_fp8 (tilelang)
#   Shapes:
#     Q:          [B, H, D]                fp8
#     BlockedK:   [B, nb, D]               fp8
#     BlockedKS:  [B, nb]                  f32    per-K-block scale
#     Weights:    [B, H]                   f32
#     ContextLens:[B]                      i32
#     Logits:     [B, nb]                  f32    OUT
#
#   Fused: post-GEMM: max(s * k_scale, 0) * weight, reduce over H,
#          mask by ContextLens, force +inf at pos 0 and last-valid pos
#          (matches tilelang kernel's force_maintain behaviour).
# ---------------------------------------------------------------------------


@triton.jit
def _batch_pool_mqa_kernel(
    Q_ptr,           # [B, H, D] fp8
    BK_ptr,          # [B, nb, D] fp8
    BKS_ptr,         # [B, nb]   f32
    Logits_ptr,      # [B, nb]   f32 OUT
    W_ptr,           # [B, H]    f32
    ContextLens_ptr, # [B]       i32
    # strides (in elements of the pointee dtype)
    stride_q_b, stride_q_h, stride_q_d,
    stride_bk_b, stride_bk_n, stride_bk_d,
    stride_bks_b, stride_bks_n,
    stride_logits_b, stride_logits_n,
    stride_w_b, stride_w_h,
    # shapes
    nb, heads, index_dim,
    # tile
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    k_blk = tl.program_id(1)   # which BLOCK_N-chunk of nb

    k_start = k_blk * BLOCK_N
    k_offs = k_start + tl.arange(0, BLOCK_N)    # [BLOCK_N]
    k_mask = k_offs < nb                         # [BLOCK_N]

    d_offs = tl.arange(0, BLOCK_D)               # [D]  BLOCK_D == index_dim
    h_offs = tl.arange(0, BLOCK_H)               # [H]  BLOCK_H == heads

    # --- Load Q[b, :, :]   [H, D] fp8 ---
    q_ptrs = Q_ptr + b * stride_q_b \
        + h_offs[:, None] * stride_q_h \
        + d_offs[None, :] * stride_q_d
    q = tl.load(q_ptrs)                          # fp8 [H, D]

    # --- Load W[b, :]  [H] f32 ---
    w = tl.load(W_ptr + b * stride_w_b + h_offs * stride_w_h)  # f32 [H]

    # --- Load BK[b, k_offs, :]  [BLOCK_N, D] fp8 ---
    # triton can't cast int→fp8 for ``other=``; masked-out rows get garbage
    # but their logits are overwritten to ±inf later via the k_e mask, so it
    # doesn't matter. Use a safe clamped address to avoid OOB.
    safe_k_offs = tl.where(k_mask, k_offs, 0)
    bk_ptrs = BK_ptr + b * stride_bk_b \
        + safe_k_offs[:, None] * stride_bk_n \
        + d_offs[None, :] * stride_bk_d
    bk = tl.load(bk_ptrs)                                  # fp8 [BLOCK_N, D]

    # --- Load BKS[b, k_offs]  [BLOCK_N] f32 ---
    bks_ptrs = BKS_ptr + b * stride_bks_b + k_offs * stride_bks_n
    bks = tl.load(bks_ptrs, mask=k_mask, other=0.0)       # f32 [BLOCK_N]

    # --- GEMM:  s[BLOCK_N, H] = bk @ q.T   (fp8 × fp8 -> f32) ---
    # tl.dot requires operands in fp8 / fp16 / bf16 / tf32; accumulator fp32.
    # bk: [BLOCK_N, D] fp8; q: [H, D] fp8  -> transpose q to get [D, H]
    s = tl.dot(bk, q.trans(1, 0), out_dtype=tl.float32)   # f32 [BLOCK_N, H]

    # --- Post: max(s * k_scale, 0) * weight, reduce over H ---
    s = s * bks[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]                                    # [BLOCK_N, H]
    logits = tl.sum(s, axis=1)                            # [BLOCK_N] f32

    # --- Mask + force_maintain ---
    context_len = tl.load(ContextLens_ptr + b)           # i32
    k_e = tl.minimum(context_len, nb)
    pos_mask_valid = (k_offs < k_e) & k_mask
    pos_mask_maintain = ((k_offs == 0) | (k_offs == (k_e - 1))) & k_mask
    # Start with -inf, overwrite with logits where valid, then set +inf at
    # position 0 and last-valid pos.
    out = tl.where(pos_mask_valid, logits, float("-inf"))
    out = tl.where(pos_mask_maintain, float("inf"), out)

    # --- Store ---
    logits_ptrs = Logits_ptr + b * stride_logits_b + k_offs * stride_logits_n
    tl.store(logits_ptrs, out, mask=k_mask)


def batch_pool_mqa_triton(
    q_fp8: torch.Tensor,          # [B, 1, H, D] fp8 (squeezed to [B, H, D] inside)
    blocked_k_fp8: torch.Tensor,  # [B, nb, D] fp8
    blocked_k_scale: torch.Tensor,  # [B, nb] f32
    weights_f32: torch.Tensor,    # [B, H] or [B*1, H] f32
    context_lens: torch.Tensor,   # [B] i32
    *,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """Triton equivalent of ``batch_pool_mqa_attn_return_logits_fp8_interface``.
    Returns logits of shape ``[B, 1, nb]`` (unsqueezed to match the tilelang
    wrapper's shape).
    """
    assert q_fp8.ndim == 4, f"expected q_fp8 [B, 1, H, D], got {q_fp8.shape}"
    B, seq_q, H, D = q_fp8.shape
    assert seq_q == 1, "decode expects q_len=1"
    assert blocked_k_fp8.ndim == 3
    B_, nb, D_ = blocked_k_fp8.shape
    assert B_ == B and D_ == D
    assert blocked_k_scale.shape == (B, nb)
    assert context_lens.shape == (B,) and context_lens.dtype == torch.int32

    q_2d = q_fp8.squeeze(1).contiguous()      # [B, H, D]
    w_2d = weights_f32.view(B, H).contiguous()

    logits = torch.empty((B, nb), device=q_fp8.device, dtype=torch.float32)

    # Tile shape picks: BLOCK_D = D (full dim tile, same as tilelang),
    # BLOCK_H = H (full head tile), BLOCK_N configurable (match tilelang 64).
    BLOCK_D = D
    BLOCK_H = H
    grid = (B, triton.cdiv(nb, BLOCK_N))
    _batch_pool_mqa_kernel[grid](
        q_2d, blocked_k_fp8, blocked_k_scale, logits, w_2d, context_lens,
        q_2d.stride(0), q_2d.stride(1), q_2d.stride(2),
        blocked_k_fp8.stride(0), blocked_k_fp8.stride(1), blocked_k_fp8.stride(2),
        blocked_k_scale.stride(0), blocked_k_scale.stride(1),
        logits.stride(0), logits.stride(1),
        w_2d.stride(0), w_2d.stride(1),
        nb, H, D,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
    return logits.unsqueeze(1)  # [B, 1, nb] to match tilelang wrapper


# ---------------------------------------------------------------------------
# Kernel 2: sparse paged block-MQA — THE 80% HOTSPOT
#   Mirrors: fp8_native_paged_block_sparse_mqa_attn_return_logits (tilelang)
#   Shapes:
#     IndexQ:       [B, seq_len, H, D]               fp8
#     KvCache_fp8:  [num_phys_blocks, paged_block_size * (D + 4)]  fp8 bytes
#     KvCache_fp32: [num_phys_blocks, paged_block_size * (D + 4) // 4] f32
#                   (same memory, f32 view for the per-token scale slots)
#     TopK:         [B, seq_len, topk]               i64  (torch.topk output)
#     Weights:      [B, seq_len, H]                  f32
#     ContextLens:  [B]                              i32
#     BlockTables:  [B, max_blocks]                  i32
#     Logits:       [B, seq_len, topk * kv_block_size] f32  OUT
#
#   Each output column c = n_i * kv_block_size + sub_i * paged_block_size +
#   local_i is a logit for K token position
#   topk_block_id[b, s, n_i] * kv_block_size + sub_i * paged_block_size + local_i.
#
#   Parallelism strategy: one CTA per (b, seq, paged_sub_block). Grid =
#   (B, seq, topk * subs_per_topk).  Each CTA does one paged_block_size-wide
#   GEMM and writes paged_block_size contiguous logits.  Q and weights are
#   reloaded per-CTA — redundant vs tilelang's per-(b, seq) load-once, but
#   the higher grid count lets many SMs work in parallel.
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_paged_mqa_kernel(
    Q_ptr,            # [B, seq, H, D] fp8
    KvCacheFp8_ptr,   # [num_phys, paged_block_size * (D + 4)] fp8 (view of uint8)
    KvCacheFp32_ptr,  # same memory, f32 view: [num_phys, paged_block_size * (D + 4) // 4]
    TopK_ptr,         # [B, seq, topk]               i64
    Logits_ptr,       # [B, seq, topk * kv_block_size] f32  OUT
    W_ptr,            # [B, seq, H]                  f32
    ContextLens_ptr,  # [B]                          i32
    BlockTables_ptr,  # [B, max_blocks]              i32
    # strides (in elements of the pointee dtype)
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,
    stride_kv8_p, stride_kv8_b,
    stride_kv32_p, stride_kv32_b,
    stride_topk_b, stride_topk_s, stride_topk_n,
    stride_logits_b, stride_logits_s, stride_logits_n,
    stride_w_b, stride_w_s, stride_w_h,
    stride_bt_b, stride_bt_mb,
    # shapes
    max_blocks,
    # constexpr
    PAGED_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SUBS_PER_TOPK: tl.constexpr,
):
    b = tl.program_id(0)
    seq_i = tl.program_id(1)
    subblock_idx = tl.program_id(2)

    n_i = subblock_idx // SUBS_PER_TOPK
    sub_i = subblock_idx % SUBS_PER_TOPK

    # --- topk_block_id and the absolute paged-block index it maps to ---
    topk_id_ptr = (
        TopK_ptr
        + b * stride_topk_b
        + seq_i * stride_topk_s
        + n_i * stride_topk_n
    )
    topk_block_id = tl.load(topk_id_ptr).to(tl.int32)
    block_s_i = topk_block_id * KV_BLOCK_SIZE + sub_i * PAGED_BLOCK_SIZE
    logical_page = block_s_i // PAGED_BLOCK_SIZE
    valid_page = (logical_page >= 0) & (logical_page < max_blocks)

    # --- Physical page lookup (clamped on invalid so the K load stays in-bounds). ---
    phys = tl.load(
        BlockTables_ptr + b * stride_bt_b + logical_page * stride_bt_mb,
        mask=valid_page, other=0,
    ).to(tl.int32)

    # --- Q [H, D] fp8 ---
    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q_ptrs = (
        Q_ptr
        + b * stride_q_b
        + seq_i * stride_q_s
        + h_offs[:, None] * stride_q_h
        + d_offs[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs)  # fp8 [H, D]

    # --- Weights [H] f32 ---
    w_ptrs = W_ptr + b * stride_w_b + seq_i * stride_w_s + h_offs * stride_w_h
    w = tl.load(w_ptrs)  # f32 [H]

    # --- K tile [PAGED_BLOCK_SIZE, D] fp8 from paged cache ---
    # Layout: bytes [0, PAGED_BLOCK_SIZE * D) = fp8 data, token i at bytes
    # [i * D, (i + 1) * D).
    bn_offs = tl.arange(0, PAGED_BLOCK_SIZE)
    k_byte_offs = bn_offs[:, None] * DIM + d_offs[None, :]
    k_ptrs = (
        KvCacheFp8_ptr
        + phys * stride_kv8_p
        + k_byte_offs * stride_kv8_b
    )
    k = tl.load(k_ptrs)  # fp8 [PAGED_BLOCK_SIZE, D]

    # --- K per-token scale [PAGED_BLOCK_SIZE] f32 ---
    # Scale is the last PAGED_BLOCK_SIZE f32 slots in the row (after the fp8 data).
    SCALE_OFFSET: tl.constexpr = PAGED_BLOCK_SIZE * DIM // 4
    ks_ptrs = (
        KvCacheFp32_ptr
        + phys * stride_kv32_p
        + (SCALE_OFFSET + bn_offs) * stride_kv32_b
    )
    k_scale = tl.load(ks_ptrs)  # f32 [PAGED_BLOCK_SIZE]

    # --- GEMM: s = k @ q.T  (fp8 × fp8 → f32) ---
    s = tl.dot(k, q.trans(1, 0), out_dtype=tl.float32)  # [PAGED_BLOCK_SIZE, H]

    # --- Post: max(s * k_scale, 0) * w, reduce over H ---
    s = s * k_scale[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]
    logits = tl.sum(s, axis=1)  # [PAGED_BLOCK_SIZE] f32

    # --- Mask out-of-range positions (-inf) ---
    context_len = tl.load(ContextLens_ptr + b)
    k_i = block_s_i + bn_offs
    pos_valid = (k_i >= 0) & (k_i < context_len) & valid_page
    logits = tl.where(pos_valid, logits, float("-inf"))

    # --- Store logits at [b, seq_i, n_i * KV_BLOCK_SIZE + sub_i * PAGED_BLOCK_SIZE + bn_offs] ---
    out_cols = n_i * KV_BLOCK_SIZE + sub_i * PAGED_BLOCK_SIZE + bn_offs
    out_ptrs = (
        Logits_ptr
        + b * stride_logits_b
        + seq_i * stride_logits_s
        + out_cols * stride_logits_n
    )
    tl.store(out_ptrs, logits)


def sparse_paged_mqa_triton(
    q_fp8: torch.Tensor,               # [B, seq, H, D] fp8
    kv_cache_fp8: torch.Tensor,        # [num_phys, paged_block_size, 1, D+4] uint8 (as returned by sglang's NSATokenToKVPool)
    topk_block_index: torch.Tensor,    # [B, seq, topk] i64
    kv_block_size: int,
    weights: torch.Tensor,             # [B, seq, H] or [B*seq, H] f32
    context_lens: torch.Tensor,        # [B] i32
    block_tables: torch.Tensor,        # [B, max_blocks] i32
) -> torch.Tensor:
    """Triton equivalent of ``fp8_native_paged_block_sparse_mqa_attn_return_logits_interface``.
    Returns logits of shape ``[B, seq, topk * kv_block_size]`` f32.
    """
    assert q_fp8.ndim == 4, f"q_fp8 should be [B, seq, H, D], got {q_fp8.shape}"
    B, seq_len, H, D = q_fp8.shape
    topk = int(topk_block_index.shape[-1])
    num_phys_blocks, paged_block_size, _, DPlus4 = kv_cache_fp8.shape
    assert _ == 1 and DPlus4 == D + 4
    assert kv_block_size % paged_block_size == 0
    subs_per_topk = kv_block_size // paged_block_size
    max_blocks = int(block_tables.shape[-1])
    total_subblocks = topk * subs_per_topk

    assert topk_block_index.dtype == torch.int64
    assert context_lens.dtype == torch.int32
    assert block_tables.dtype == torch.int32

    # Views (no data copy).
    kv_cache_flat = kv_cache_fp8.view(num_phys_blocks, -1)      # uint8 [num_phys, P*(D+4)]
    kv_fp8_view = kv_cache_flat.view(torch.float8_e4m3fn)       # fp8 same layout
    kv_f32_view = kv_cache_flat.view(torch.float32)             # f32 view

    if weights.ndim == 2:
        weights = weights.view(B, seq_len, H)

    logits = torch.empty(
        (B, seq_len, topk * kv_block_size),
        device=q_fp8.device, dtype=torch.float32,
    )

    grid = (B, seq_len, total_subblocks)
    _sparse_paged_mqa_kernel[grid](
        q_fp8, kv_fp8_view, kv_f32_view, topk_block_index,
        logits, weights, context_lens, block_tables,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2), q_fp8.stride(3),
        kv_fp8_view.stride(0), kv_fp8_view.stride(1),
        kv_f32_view.stride(0), kv_f32_view.stride(1),
        topk_block_index.stride(0), topk_block_index.stride(1), topk_block_index.stride(2),
        logits.stride(0), logits.stride(1), logits.stride(2),
        weights.stride(0), weights.stride(1), weights.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        max_blocks,
        PAGED_BLOCK_SIZE=paged_block_size,
        KV_BLOCK_SIZE=kv_block_size,
        HEADS=H,
        DIM=D,
        SUBS_PER_TOPK=subs_per_topk,
    )
    return logits


# ---------------------------------------------------------------------------
# Kernel 3: ragged block-sparse MQA (prefill path)
#   Mirrors: fp8_native_block_sparse_mqa_attn_return_logits (tilelang)
#   Shapes:
#     Q:    [seq, H, D]                 fp8
#     K:    [seq_kv, D]                 fp8
#     KS:   [seq_kv]                    f32
#     TopK: [seq, topk]                 i64
#     W:    [seq, H]                    f32
#     CuKS: [seq]  int32   (cu_seqlen_ks: K-start per query)
#     CuKE: [seq]  int32   (cu_seqlen_ke: K-end   per query)
#     Logits: [seq, topk * kv_block_size] f32 OUT
#
#   Parallelism: one CTA per (seq_i, sub_block) where
#   sub_block ∈ [0, topk * (kv_block_size / BLOCK_N)). Each CTA handles
#   one BLOCK_N-wide slice of one topk block.
# ---------------------------------------------------------------------------


@triton.jit
def _block_sparse_mqa_kernel(
    Q_ptr, K_ptr, KS_ptr,
    TopK_ptr, Logits_ptr, W_ptr,
    CuKS_ptr, CuKE_ptr,
    stride_q_s, stride_q_h, stride_q_d,
    stride_k_s, stride_k_d,
    stride_ks_s,
    stride_topk_s, stride_topk_n,
    stride_logits_s, stride_logits_n,
    stride_w_s, stride_w_h,
    seq_kv,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SUBS_PER_TOPK: tl.constexpr,
):
    seq_i = tl.program_id(0)
    subblock_idx = tl.program_id(1)
    n_i = subblock_idx // SUBS_PER_TOPK
    sub_i = subblock_idx % SUBS_PER_TOPK

    topk_id = tl.load(TopK_ptr + seq_i * stride_topk_s + n_i * stride_topk_n).to(tl.int32)
    block_s_i = topk_id * KV_BLOCK_SIZE + sub_i * BLOCK_N

    # Load Q [H, D].
    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q = tl.load(
        Q_ptr + seq_i * stride_q_s
        + h_offs[:, None] * stride_q_h + d_offs[None, :] * stride_q_d
    )  # fp8

    # Load W [H].
    w = tl.load(W_ptr + seq_i * stride_w_s + h_offs * stride_w_h)  # f32

    # Load K tile [BLOCK_N, D] + scales [BLOCK_N] from ragged K.
    bn_offs = tl.arange(0, BLOCK_N)
    k_rows = block_s_i + bn_offs
    k_mask = (k_rows >= 0) & (k_rows < seq_kv)
    safe_rows = tl.where(k_mask, k_rows, 0)
    k_ptrs = K_ptr + safe_rows[:, None] * stride_k_s + d_offs[None, :] * stride_k_d
    k = tl.load(k_ptrs)  # fp8 [BLOCK_N, D]
    ks = tl.load(KS_ptr + safe_rows * stride_ks_s, mask=k_mask, other=0.0)

    # GEMM.
    s = tl.dot(k, q.trans(1, 0), out_dtype=tl.float32)  # [BLOCK_N, H]
    s = s * ks[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]
    logits = tl.sum(s, axis=1)  # [BLOCK_N] f32

    # Mask by cu_seqlen_ks / cu_seqlen_ke.
    ks_min = tl.load(CuKS_ptr + seq_i)
    ke_max = tl.load(CuKE_ptr + seq_i)
    pos_valid = (k_rows >= ks_min) & (k_rows < ke_max) & k_mask
    logits = tl.where(pos_valid, logits, float("-inf"))

    out_cols = n_i * KV_BLOCK_SIZE + sub_i * BLOCK_N + bn_offs
    tl.store(
        Logits_ptr + seq_i * stride_logits_s + out_cols * stride_logits_n,
        logits,
    )


def block_sparse_mqa_triton(
    q_fp8: torch.Tensor,              # [seq, H, D] fp8
    k_fp8: torch.Tensor,              # [seq_kv, D] fp8
    k_scale: torch.Tensor,            # [seq_kv] f32
    topk_block_index: torch.Tensor,   # [seq, topk] i64
    kv_block_size: int,
    weights: torch.Tensor,            # [seq, H] f32
    cu_seqlen_ks: torch.Tensor,       # [seq] i32
    cu_seqlen_ke: torch.Tensor,       # [seq] i32
) -> torch.Tensor:
    assert q_fp8.ndim == 3
    seq_len, H, D = q_fp8.shape
    seq_kv = k_fp8.shape[0]
    topk = int(topk_block_index.shape[-1])
    BLOCK_N = 128 if kv_block_size >= 128 else 64
    if kv_block_size < BLOCK_N:
        BLOCK_N = kv_block_size
    assert kv_block_size % BLOCK_N == 0
    subs_per_topk = kv_block_size // BLOCK_N
    logits = torch.empty(
        (seq_len, topk * kv_block_size), device=q_fp8.device, dtype=torch.float32,
    )
    grid = (seq_len, topk * subs_per_topk)
    _block_sparse_mqa_kernel[grid](
        q_fp8, k_fp8, k_scale, topk_block_index, logits, weights,
        cu_seqlen_ks, cu_seqlen_ke,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        k_fp8.stride(0), k_fp8.stride(1),
        k_scale.stride(0),
        topk_block_index.stride(0), topk_block_index.stride(1),
        logits.stride(0), logits.stride(1),
        weights.stride(0), weights.stride(1),
        seq_kv,
        HEADS=H, DIM=D,
        KV_BLOCK_SIZE=kv_block_size,
        BLOCK_N=BLOCK_N,
        SUBS_PER_TOPK=subs_per_topk,
    )
    return logits


# ---------------------------------------------------------------------------
# Kernel 4: v3 decode block-MQA on paged pool_k_pages
#   Mirrors: batch_decode_pool_mqa_attn_return_logits_fp8_v3 (tilelang)
#   Shapes:
#     Q:               [B, H, D]                         fp8
#     PoolKPages_u8:   [N_pool_pages, PP * (D+4)]        uint8 (as returned by pool)
#                      viewed as fp8 for data, as f32 for scale
#     PoolPageTables:  [B, max_pool_pages]               i32
#     ContextLensPool: [B]                               i32  (num pool rows per req)
#     Weights:         [B, H]                            f32
#     Logits:          [B, max_pool_pages * PP]          f32 OUT
#
#   Grid: (B, max_pool_pages). Each CTA loads one pool page (PP=64 rows)
#   from pool_k_pages[phys] and does the block_N=64 GEMM against Q.
# ---------------------------------------------------------------------------


@triton.jit
def _batch_decode_pool_mqa_v3_kernel(
    Q_ptr,            # [B, H, D] fp8
    PKPagesFp8_ptr,   # [N_pp, PP * (D+4)] fp8
    PKPagesF32_ptr,   # [N_pp, PP * (D+4) // 4] f32
    PageTables_ptr,   # [B, max_pp] i32
    Logits_ptr,       # [B, max_pp * PP] f32 OUT
    W_ptr,            # [B, H] f32
    ContextLensPool_ptr,  # [B] i32
    stride_q_b, stride_q_h, stride_q_d,
    stride_pk8_p, stride_pk8_b,
    stride_pk32_p, stride_pk32_b,
    stride_pt_b, stride_pt_n,
    stride_logits_b, stride_logits_n,
    stride_w_b, stride_w_h,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PP: tl.constexpr,  # pool_page_size (= 64)
):
    b = tl.program_id(0)
    lp = tl.program_id(1)  # logical pool page

    phys = tl.load(PageTables_ptr + b * stride_pt_b + lp * stride_pt_n).to(tl.int32)

    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q = tl.load(
        Q_ptr + b * stride_q_b
        + h_offs[:, None] * stride_q_h + d_offs[None, :] * stride_q_d
    )  # fp8 [H, D]
    w = tl.load(W_ptr + b * stride_w_b + h_offs * stride_w_h)  # f32 [H]

    # Load pool page: PP rows × D fp8 data, then PP f32 scales.
    bn_offs = tl.arange(0, PP)
    k_byte_offs = bn_offs[:, None] * DIM + d_offs[None, :]
    k = tl.load(
        PKPagesFp8_ptr + phys * stride_pk8_p + k_byte_offs * stride_pk8_b
    )  # fp8 [PP, D]
    SCALE_OFFSET: tl.constexpr = PP * DIM // 4
    ks = tl.load(
        PKPagesF32_ptr + phys * stride_pk32_p
        + (SCALE_OFFSET + bn_offs) * stride_pk32_b
    )  # f32 [PP]

    # GEMM + post.
    s = tl.dot(k, q.trans(1, 0), out_dtype=tl.float32)
    s = s * ks[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]
    logits = tl.sum(s, axis=1)  # [PP] f32

    # Mask by ContextLensPool: positions >= k_e become -inf, and apply
    # force_maintain (+inf at pos 0 and at k_e - 1).
    k_e = tl.load(ContextLensPool_ptr + b)
    pool_idx = lp * PP + bn_offs
    pos_valid = pool_idx < k_e
    pos_maintain = (pool_idx == 0) | (pool_idx == (k_e - 1))
    logits = tl.where(pos_valid, logits, float("-inf"))
    logits = tl.where(pos_maintain, float("inf"), logits)

    tl.store(
        Logits_ptr + b * stride_logits_b + pool_idx * stride_logits_n,
        logits,
    )


def batch_decode_pool_mqa_v3_triton(
    q_fp8: torch.Tensor,              # [B, 1, H, D] fp8
    pool_k_pages: torch.Tensor,       # [N_pp, PP * (D+4)] uint8
    pool_page_tables: torch.Tensor,   # [B, max_pp] i32
    weights_f32: torch.Tensor,        # [B, H] or [B, 1, H] f32
    context_lens_pool: torch.Tensor,  # [B] i32
    *,
    pool_page_size: int = 64,
) -> torch.Tensor:
    assert q_fp8.ndim == 4
    B, seq_q, H, D = q_fp8.shape
    assert seq_q == 1
    max_pp = pool_page_tables.shape[-1]
    assert pool_page_tables.shape == (B, max_pp)
    assert pool_k_pages.shape[1] == pool_page_size * (D + 4)
    assert context_lens_pool.dtype == torch.int32
    assert pool_page_tables.dtype == torch.int32

    q_2d = q_fp8.squeeze(1).contiguous()  # [B, H, D]
    w_2d = weights_f32.view(B, H).contiguous()
    pk8 = pool_k_pages.view(torch.float8_e4m3fn)
    pk32 = pool_k_pages.view(torch.float32)

    logits = torch.empty(
        (B, max_pp * pool_page_size), device=q_fp8.device, dtype=torch.float32,
    )
    grid = (B, max_pp)
    _batch_decode_pool_mqa_v3_kernel[grid](
        q_2d, pk8, pk32, pool_page_tables, logits, w_2d, context_lens_pool,
        q_2d.stride(0), q_2d.stride(1), q_2d.stride(2),
        pk8.stride(0), pk8.stride(1),
        pk32.stride(0), pk32.stride(1),
        pool_page_tables.stride(0), pool_page_tables.stride(1),
        logits.stride(0), logits.stride(1),
        w_2d.stride(0), w_2d.stride(1),
        HEADS=H, DIM=D, PP=pool_page_size,
    )
    return logits.unsqueeze(1)  # [B, 1, max_pp * PP]


# ---------------------------------------------------------------------------
# Kernel 5: v1 paged mean pooling (full batch × all blocks)
#   Mirrors: fp8_native_paged_mean_pooling (tilelang)
#   Inputs:
#     kv_cache: [num_phys, paged_block_size, 1, D+4] uint8
#     block_tables: [B, max_blocks] i32
#     context_lens: [B] i32
#   Outputs:
#     blocked_k: [B, max_num_pool, D] fp8
#     blocked_k_scale: [B, max_num_pool] f32
#
#   Grid: (B, max_num_pool). Each CTA pools one (request, pool_block_idx)
#   pair by summing over its K=128 tokens from paged cache, mean, and
#   fp8-requantizing with a per-pool-block scale.
# ---------------------------------------------------------------------------


@triton.jit
def _paged_mean_pooling_kernel(
    KvCacheFp8_ptr,   # [num_phys, paged_block_size * (D+4)] fp8
    KvCacheF32_ptr,   # [num_phys, paged_block_size * (D+4) // 4] f32
    BlockTables_ptr,  # [B, max_blocks] i32
    ContextLens_ptr,  # [B] i32
    BlockedK_ptr,     # [B, max_num_pool, D] fp8 OUT
    BlockedKScale_ptr,  # [B, max_num_pool] f32 OUT
    stride_kv8_p, stride_kv8_b,
    stride_kv32_p, stride_kv32_b,
    stride_bt_b, stride_bt_mb,
    stride_bk_b, stride_bk_n, stride_bk_d,
    stride_bks_b, stride_bks_n,
    max_blocks,
    PAGED_BLOCK_SIZE: tl.constexpr,
    POOLING_BLOCK_SIZE: tl.constexpr,  # K=128
    DIM: tl.constexpr,
):
    b = tl.program_id(0)
    pblk = tl.program_id(1)

    seq_len = tl.load(ContextLens_ptr + b)
    num_pool = (seq_len + POOLING_BLOCK_SIZE - 1) // POOLING_BLOCK_SIZE

    # Skip pblk >= num_pool (leave outputs uninitialized, same as tilelang).
    if pblk >= num_pool:
        return

    k_start = pblk * POOLING_BLOCK_SIZE
    k_end = tl.minimum(k_start + POOLING_BLOCK_SIZE, seq_len)
    cur_size = k_end - k_start

    # Accumulator over D.
    d_offs = tl.arange(0, DIM)
    acc = tl.zeros([DIM], dtype=tl.float32)

    # Each pool block spans POOLING_BLOCK_SIZE / PAGED_BLOCK_SIZE paged chunks.
    CHUNKS: tl.constexpr = POOLING_BLOCK_SIZE // PAGED_BLOCK_SIZE
    SCALE_OFFSET: tl.constexpr = PAGED_BLOCK_SIZE * DIM // 4
    bn_offs = tl.arange(0, PAGED_BLOCK_SIZE)
    for c in tl.static_range(CHUNKS):
        paged_block_s = k_start + c * PAGED_BLOCK_SIZE
        logical_page = paged_block_s // PAGED_BLOCK_SIZE
        valid_page = (logical_page >= 0) & (logical_page < max_blocks)
        phys = tl.load(
            BlockTables_ptr + b * stride_bt_b + logical_page * stride_bt_mb,
            mask=valid_page, other=0,
        ).to(tl.int32)

        # Load K tile [PAGED_BLOCK_SIZE, D] fp8, convert to f32.
        k_byte = bn_offs[:, None] * DIM + d_offs[None, :]
        k_fp8 = tl.load(KvCacheFp8_ptr + phys * stride_kv8_p + k_byte * stride_kv8_b)
        k = k_fp8.to(tl.float32)

        # Load per-token scale [PAGED_BLOCK_SIZE] f32.
        scale = tl.load(
            KvCacheF32_ptr + phys * stride_kv32_p
            + (SCALE_OFFSET + bn_offs) * stride_kv32_b,
        )

        # Zero out tokens past k_end.
        tl_block_idx = paged_block_s + bn_offs
        in_block = tl_block_idx < k_end
        k = tl.where(in_block[:, None], k, 0.0)
        # Multiply scale applied to dequantized fp8.
        k = k * scale[:, None]
        # Accumulate over block_N.
        acc += tl.sum(k, axis=0)

    # Mean and fp8 re-quantize.
    inv_count = 1.0 / tl.cast(cur_size, tl.float32)
    acc = acc * inv_count
    max_abs = tl.max(tl.abs(acc))
    block_scale = tl.maximum(max_abs * (1.0 / 448.0), 1e-10)
    inv_block_scale = 1.0 / block_scale

    out_fp8 = (acc * inv_block_scale).to(tl.float8e4nv)

    # Store.
    tl.store(
        BlockedK_ptr + b * stride_bk_b + pblk * stride_bk_n
        + d_offs * stride_bk_d,
        out_fp8,
    )
    tl.store(BlockedKScale_ptr + b * stride_bks_b + pblk * stride_bks_n, block_scale)


def paged_mean_pooling_triton(
    max_num_pooling_blocks: int,
    kv_cache: torch.Tensor,           # [num_phys, paged_block_size, 1, D+4] uint8
    context_lens: torch.Tensor,       # [B] i32
    block_tables: torch.Tensor,       # [B, max_blocks] i32
    k_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_phys, paged_block_size, head, DPlus4 = kv_cache.shape
    assert head == 1
    D = DPlus4 - 4
    B, max_blocks = block_tables.shape

    kv_flat = kv_cache.view(num_phys, paged_block_size * DPlus4)

    blocked_k = torch.empty(
        (B, max_num_pooling_blocks, D),
        device=kv_cache.device, dtype=torch.float8_e4m3fn,
    )
    blocked_k_scale = torch.empty(
        (B, max_num_pooling_blocks),
        device=kv_cache.device, dtype=torch.float32,
    )

    grid = (B, max_num_pooling_blocks)
    _paged_mean_pooling_kernel[grid](
        kv_flat.view(torch.float8_e4m3fn),
        kv_flat.view(torch.float32),
        block_tables, context_lens,
        blocked_k, blocked_k_scale,
        kv_flat.stride(0), kv_flat.stride(1),
        (kv_flat.view(torch.float32)).stride(0), (kv_flat.view(torch.float32)).stride(1),
        block_tables.stride(0), block_tables.stride(1),
        blocked_k.stride(0), blocked_k.stride(1), blocked_k.stride(2),
        blocked_k_scale.stride(0), blocked_k_scale.stride(1),
        max_blocks,
        PAGED_BLOCK_SIZE=paged_block_size,
        POOLING_BLOCK_SIZE=k_block_size,
        DIM=D,
    )
    num_pooling_blocks = (context_lens + k_block_size - 1) // k_block_size
    return blocked_k, blocked_k_scale, num_pooling_blocks


# ---------------------------------------------------------------------------
# Kernel 6: ragged block mean-pool
#   Mirrors: fp8_native_block_mean_pooling (tilelang)
#   Inputs:
#     K:       [seq_kv, D]       fp8
#     KScale:  [seq_kv]          f32
#   Outputs:
#     BlockedK:  [num_pool, D]   fp8
#     BlockedKS: [num_pool]      f32
#
#   Grid: (ceildiv(seq_kv, K),). One CTA per pool block; sums K tokens,
#   mean, fp8-requantizes.
# ---------------------------------------------------------------------------


@triton.jit
def _block_mean_pooling_kernel(
    K_ptr,            # [seq_kv, D] fp8
    KScale_ptr,       # [seq_kv] f32
    BlockedK_ptr,     # [num_pool, D] fp8 OUT
    BlockedKS_ptr,    # [num_pool] f32 OUT
    stride_k_s, stride_k_d,
    stride_ks_s,
    stride_bk_n, stride_bk_d,
    stride_bks_n,
    seq_kv,
    POOLING_BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pblk = tl.program_id(0)
    k_start = pblk * POOLING_BLOCK_SIZE
    k_end = tl.minimum(k_start + POOLING_BLOCK_SIZE, seq_kv)
    cur_size = k_end - k_start

    d_offs = tl.arange(0, DIM)
    acc = tl.zeros([DIM], dtype=tl.float32)

    CHUNKS: tl.constexpr = POOLING_BLOCK_SIZE // BLOCK_N
    bn_offs = tl.arange(0, BLOCK_N)
    for c in tl.static_range(CHUNKS):
        tl_block_s = k_start + c * BLOCK_N
        rows = tl_block_s + bn_offs
        in_block = (rows >= k_start) & (rows < k_end)
        safe_rows = tl.where(in_block, rows, 0)

        # Load [BLOCK_N, D] fp8, cast to f32.
        k_ptrs = K_ptr + safe_rows[:, None] * stride_k_s + d_offs[None, :] * stride_k_d
        k = tl.load(k_ptrs).to(tl.float32)

        # Load scales [BLOCK_N] f32.
        scale = tl.load(KScale_ptr + safe_rows * stride_ks_s, mask=in_block, other=0.0)

        k = tl.where(in_block[:, None], k, 0.0)
        k = k * scale[:, None]
        acc += tl.sum(k, axis=0)

    inv_count = 1.0 / tl.cast(cur_size, tl.float32)
    acc = acc * inv_count
    max_abs = tl.max(tl.abs(acc))
    block_scale = tl.maximum(max_abs * (1.0 / 448.0), 1e-10)
    inv_block_scale = 1.0 / block_scale
    out_fp8 = (acc * inv_block_scale).to(tl.float8e4nv)
    tl.store(BlockedK_ptr + pblk * stride_bk_n + d_offs * stride_bk_d, out_fp8)
    tl.store(BlockedKS_ptr + pblk * stride_bks_n, block_scale)


def block_mean_pooling_triton(
    k_fp8: torch.Tensor,       # [seq_kv, D] fp8
    k_scale: torch.Tensor,     # [seq_kv] f32
    k_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_kv, D = k_fp8.shape
    max_num_pool = (seq_kv + k_block_size - 1) // k_block_size
    BLOCK_N = min(64, k_block_size)
    assert k_block_size % BLOCK_N == 0
    blocked_k = torch.empty(
        (max_num_pool, D), device=k_fp8.device, dtype=torch.float8_e4m3fn,
    )
    blocked_k_scale = torch.empty(
        (max_num_pool,), device=k_fp8.device, dtype=torch.float32,
    )
    grid = (max_num_pool,)
    _block_mean_pooling_kernel[grid](
        k_fp8, k_scale, blocked_k, blocked_k_scale,
        k_fp8.stride(0), k_fp8.stride(1),
        k_scale.stride(0),
        blocked_k.stride(0), blocked_k.stride(1),
        blocked_k_scale.stride(0),
        seq_kv,
        POOLING_BLOCK_SIZE=k_block_size,
        DIM=D,
        BLOCK_N=BLOCK_N,
    )
    return blocked_k, blocked_k_scale


# ---------------------------------------------------------------------------
# Kernels 7 + 8: v3 pool-K write helpers — left as TODO (tail_only_v3 and
# completed_blocks_v3). They mean-pool into pool_k_pages[phys, slot] where
# `phys` is a page ID + `slot` is the row within the page. The math is the
# same as paged_mean_pooling_triton above; only the output indirection and
# the gating condition (pblk_rel < n_new) differ.
#
# Skipping the port for now because both kernels take at most ~10 μs/call
# and are not on the decode hotspot (they run during K writes). The same
# per-kernel speedup that shows up for sparse_paged (6-15×) is not in the
# critical decode path for these.
# ---------------------------------------------------------------------------
