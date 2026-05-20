"""Triton ports of HISA kernels.

Notes on the fp8 path:
  - Requires ``triton >= 3.0`` for ``tl.dot`` on fp8 operands.
  - fp8 scales are per-K-block (one f32 per K-block), applied **after** the
    GEMM as a scalar multiply.
  - ``tl.dot`` needs the accumulator and one operand in ``float32``/``tf32``
    depending on arch; fp8 × fp8 → fp32 accumulator is supported on H100+.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Kernel 1: contiguous block-MQA (fp8 Q × fp8 K + per-block-scale + weights)
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
    Q_ptr,  # [B, H, D] fp8
    BK_ptr,  # [B, nb, D] fp8
    BKS_ptr,  # [B, nb]   f32
    Logits_ptr,  # [B, nb]   f32 OUT
    W_ptr,  # [B, H]    f32
    ContextLens_ptr,  # [B]       i32
    # strides (in elements of the pointee dtype)
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_bk_b,
    stride_bk_n,
    stride_bk_d,
    stride_bks_b,
    stride_bks_n,
    stride_logits_b,
    stride_logits_n,
    stride_w_b,
    stride_w_h,
    # shapes
    nb,
    heads,
    index_dim,
    # tile
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    k_blk = tl.program_id(1)  # which BLOCK_N-chunk of nb

    k_start = k_blk * BLOCK_N
    k_offs = k_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k_mask = k_offs < nb  # [BLOCK_N]

    d_offs = tl.arange(0, BLOCK_D)  # [D]  BLOCK_D == index_dim
    h_offs = tl.arange(0, BLOCK_H)  # [H]  BLOCK_H == heads

    # --- Load Q[b, :, :]   [H, D] fp8 ---
    q_ptrs = (
        Q_ptr
        + b * stride_q_b
        + h_offs[:, None] * stride_q_h
        + d_offs[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs)  # fp8 [H, D]

    # --- Load W[b, :]  [H] f32 ---
    w = tl.load(W_ptr + b * stride_w_b + h_offs * stride_w_h)  # f32 [H]

    # --- Load BK[b, k_offs, :]  [BLOCK_N, D] fp8 ---
    # triton can't cast int→fp8 for ``other=``; masked-out rows get garbage
    # but their logits are overwritten to ±inf later via the k_e mask, so it
    # doesn't matter. Use a safe clamped address to avoid OOB.
    safe_k_offs = tl.where(k_mask, k_offs, 0)
    bk_ptrs = (
        BK_ptr
        + b * stride_bk_b
        + safe_k_offs[:, None] * stride_bk_n
        + d_offs[None, :] * stride_bk_d
    )
    bk = tl.load(bk_ptrs)  # fp8 [BLOCK_N, D]

    # --- Load BKS[b, k_offs]  [BLOCK_N] f32 ---
    bks_ptrs = BKS_ptr + b * stride_bks_b + k_offs * stride_bks_n
    bks = tl.load(bks_ptrs, mask=k_mask, other=0.0)  # f32 [BLOCK_N]

    # --- GEMM:  s[BLOCK_N, H] = bk @ q.T   (fp8 × fp8 -> f32) ---
    # tl.dot requires operands in fp8 / fp16 / bf16 / tf32; accumulator fp32.
    # bk: [BLOCK_N, D] fp8; q: [H, D] fp8  -> transpose q to get [D, H]
    s = tl.dot(bk, q.trans(1, 0), out_dtype=tl.float32)  # f32 [BLOCK_N, H]

    # --- Post: max(s * k_scale, 0) * weight, reduce over H ---
    s = s * bks[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]  # [BLOCK_N, H]
    logits = tl.sum(s, axis=1)  # [BLOCK_N] f32

    # --- Mask + force_maintain ---
    context_len = tl.load(ContextLens_ptr + b)  # i32
    k_e = tl.minimum(context_len, nb)
    pos_mask_valid = (k_offs < k_e) & k_mask
    pos_mask_maintain = ((k_offs == 0) | (k_offs == (k_e - 1))) & k_mask
    # Start with -inf, overwrite with logits where valid, then set +inf at
    # position 0 and last-valid pos. Single nested tl.where (D4 pattern —
    # let triton compiler emit one fused select chain).
    out = tl.where(
        pos_mask_maintain,
        float("inf"),
        tl.where(pos_mask_valid, logits, float("-inf")),
    )

    # --- Store ---
    logits_ptrs = Logits_ptr + b * stride_logits_b + k_offs * stride_logits_n
    tl.store(logits_ptrs, out, mask=k_mask)


def batch_pool_mqa_triton(
    q_fp8: torch.Tensor,  # [B, 1, H, D] fp8 (squeezed to [B, H, D] inside)
    blocked_k_fp8: torch.Tensor,  # [B, nb, D] fp8
    blocked_k_scale: torch.Tensor,  # [B, nb] f32
    weights_f32: torch.Tensor,  # [B, H] or [B*1, H] f32
    context_lens: torch.Tensor,  # [B] i32
    *,
    BLOCK_N: int | None = None,
) -> torch.Tensor:
    """Contiguous block-MQA (decode): fp8 Q × fp8 BlockedK with per-block scales.

    Returns logits of shape ``[B, 1, nb]``.
    """
    assert q_fp8.ndim == 4, f"expected q_fp8 [B, 1, H, D], got {q_fp8.shape}"
    B, seq_q, H, D = q_fp8.shape
    assert seq_q == 1, "decode expects q_len=1"
    assert blocked_k_fp8.ndim == 3
    B_, nb, D_ = blocked_k_fp8.shape
    assert B_ == B and D_ == D
    assert blocked_k_scale.shape == (B, nb)
    assert context_lens.shape == (B,) and context_lens.dtype == torch.int32

    q_2d = q_fp8.squeeze(1).contiguous()  # [B, H, D]
    w_2d = weights_f32.view(B, H).contiguous()

    logits = torch.empty((B, nb), device=q_fp8.device, dtype=torch.float32)

    # Tile shape picks: BLOCK_D = D (full dim tile, same as tilelang),
    # BLOCK_H = H (full head tile). BLOCK_N: auto-select 128 for nb >= 256
    # (lifts WGMMA m=64→128 for ~10% throughput) else 64 (small-grid case
    # where reducing CTAs would hurt more than the bigger tile helps).
    if BLOCK_N is None:
        BLOCK_N = 128 if nb >= 256 else 64
    BLOCK_D = D
    BLOCK_H = H
    grid = (B, triton.cdiv(nb, BLOCK_N))
    _batch_pool_mqa_kernel[grid](
        q_2d,
        blocked_k_fp8,
        blocked_k_scale,
        logits,
        w_2d,
        context_lens,
        q_2d.stride(0),
        q_2d.stride(1),
        q_2d.stride(2),
        blocked_k_fp8.stride(0),
        blocked_k_fp8.stride(1),
        blocked_k_fp8.stride(2),
        blocked_k_scale.stride(0),
        blocked_k_scale.stride(1),
        logits.stride(0),
        logits.stride(1),
        w_2d.stride(0),
        w_2d.stride(1),
        nb,
        H,
        D,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
    return logits.unsqueeze(1)  # [B, 1, nb] to match tilelang wrapper


# ---------------------------------------------------------------------------
# Kernel 1b: ragged prefill block-MQA on already-pooled blocked_k
#   Mirrors: pool_mqa_attn_return_logits_fp8 (tilelang) — but ragged across
#   the prefill sequence dimension (per-row [ks, ke) range in blocked space)
#   instead of decode's (B, single-token) layout.
#
#   Shapes:
#     Q:          [seq, H, D]               fp8
#     BlockedK:   [num_pool, D]             fp8     (concatenated across batch)
#     BlockedKS:  [num_pool]                f32     per-block fp8 scale
#     Logits:     [seq, num_pool]           f32     OUT (full grid; outside
#                                                       [ks, ke) → -inf)
#     Weights:    [seq, H]                  f32
#     CuKS/CuKE:  [seq]                     i32     per-query [ks, ke) range
#                                                   in blocked space
#
#   Grid: (seq, ceildiv(num_pool, BLOCK_N)). Each CTA processes one query
#   token + BLOCK_N pool blocks. clean_logits + force_maintain fused into
#   the GEMM post-processing (D4 nested tl.where pattern).
#
#   This kernel exists because tilelang's fp8_native_block_mean_pooling has
#   a boundary OOB read at K < block_N (T.copy reads block_N=64 rows from a
#   K-row pool block). Routing the entire prefill stack through triton at
#   K<64 sidesteps that bug. K>=64 still uses tilelang for both cache miss
#   handling and bench parity.
# ---------------------------------------------------------------------------


@triton.jit
def _ragged_pool_mqa_kernel(
    Q_ptr,  # [seq, H, D] fp8
    BK_ptr,  # [num_pool, D] fp8
    BKS_ptr,  # [num_pool] f32
    Logits_ptr,  # [seq, num_pool] f32 OUT
    W_ptr,  # [seq, H] f32
    CuKS_ptr,  # [seq] i32 — ks per query (raw token space when K_BLOCK_SIZE>1, else blocked)
    CuKE_ptr,  # [seq] i32 — ke per query (raw token space, exclusive)
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_bk_n,
    stride_bk_d,
    stride_bks_n,
    stride_logits_s,
    stride_logits_n,
    stride_w_s,
    stride_w_h,
    stride_ks_s,
    stride_ke_s,
    num_pool,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_BLOCK_SIZE: tl.constexpr,  # divides ks/ke to blocked space inside kernel
):
    seq_i = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    n_start = chunk_idx * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    n_mask = n_offs < num_pool  # in-tensor bound
    safe_n = tl.where(n_mask, n_offs, 0)

    # Load Q[seq_i, :, :] fp8 and W[seq_i, :] f32.
    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q = tl.load(
        Q_ptr
        + seq_i * stride_q_s
        + h_offs[:, None] * stride_q_h
        + d_offs[None, :] * stride_q_d
    )  # fp8 [H, D]
    w = tl.load(W_ptr + seq_i * stride_w_s + h_offs * stride_w_h)  # f32 [H]

    # Load BlockedK[n_offs, :] fp8 and BlockedKS[n_offs] f32. Safe pointer
    # for the fp8 load (triton can't cast int→fp8 default), then use
    # n_mask to mask the scale and the final logit.
    bk = tl.load(
        BK_ptr + safe_n[:, None] * stride_bk_n + d_offs[None, :] * stride_bk_d
    )  # fp8 [BLOCK_N, D]
    bks = tl.load(
        BKS_ptr + safe_n * stride_bks_n,
        mask=n_mask,
        other=0.0,
    )  # f32 [BLOCK_N]

    # GEMM: [BLOCK_N, D] @ [D, H] = [BLOCK_N, H], then post-GEMM reduce.
    s = tl.dot(bk, q.trans(1, 0), out_dtype=tl.float32)
    s = s * bks[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]
    logits = tl.sum(s, axis=1)  # [BLOCK_N] f32

    # Apply per-query [ks, ke) mask + force_maintain (matches tilelang's
    # clean_and_maintain_logits_kernel semantics). When K_BLOCK_SIZE>1 the
    # caller passes raw token-space ks/ke; we map to blocked space here so the
    # orchestrator can skip 3-4 host-side PyTorch elementwise launches.
    ks_raw = tl.load(CuKS_ptr + seq_i * stride_ks_s)
    ke_raw = tl.load(CuKE_ptr + seq_i * stride_ke_s)
    ks = ks_raw // K_BLOCK_SIZE
    ke = (ke_raw + (K_BLOCK_SIZE - 1)) // K_BLOCK_SIZE
    pos_valid = (n_offs >= ks) & (n_offs < ke) & n_mask
    pos_maintain = ((n_offs == ks) | (n_offs == ke - 1)) & n_mask
    out = tl.where(
        pos_maintain,
        float("inf"),
        tl.where(pos_valid, logits, float("-inf")),
    )

    tl.store(
        Logits_ptr + seq_i * stride_logits_s + n_offs * stride_logits_n,
        out,
        mask=n_mask,
    )


def ragged_pool_mqa_triton(
    q_fp8: torch.Tensor,  # [seq, H, D] fp8
    blocked_k_fp8: torch.Tensor,  # [num_pool, D] fp8
    blocked_k_scale: torch.Tensor,  # [num_pool] f32
    weights: torch.Tensor,  # [seq, H] f32
    cu_seqlen_ks: torch.Tensor,  # [seq] i32/i64 — raw token-space when k_block_size>1
    cu_seqlen_ke: torch.Tensor,  # [seq] i32/i64 — raw token-space (exclusive)
    k_block_size: int = 1,
    *,
    BLOCK_N: int | None = None,
) -> torch.Tensor:
    """Triton equivalent of ``pool_mqa_attn_return_logits_fp8_interface``.

    Returns logits ``[seq, num_pool]`` f32, with positions outside per-row
    pool-blocked [ks, ke) set to -inf and ks / (ke-1) set to +inf (matches
    tilelang's clean_and_maintain post-process). Output is suitable for direct
    ``torch.topk`` consumption.

    ``k_block_size`` lets the kernel compute ``ks_blocked = ks // K`` and
    ``ke_blocked = ceil(ke / K)`` internally so the caller can pass raw
    token-space cu_seqlen, saving 3-4 host-side PyTorch elementwise launches
    on the orchestrator hot path. Default ``k_block_size=1`` preserves the
    pre-existing API where callers pass already-blocked cu_seqlen.
    """
    assert q_fp8.ndim == 3, f"q_fp8 should be [seq, H, D], got {q_fp8.shape}"
    seq_len, H, D = q_fp8.shape
    num_pool, D_ = blocked_k_fp8.shape
    assert D_ == D
    assert blocked_k_scale.shape == (num_pool,)
    assert weights.shape == (seq_len, H)
    assert cu_seqlen_ks.shape == (seq_len,)
    assert cu_seqlen_ke.shape == (seq_len,)
    # Triton's tl.load handles i32/i64 transparently; accept either so callers
    # can skip a redundant .to(torch.int32) cast (= 1 kernel launch).
    assert cu_seqlen_ks.dtype in (torch.int32, torch.int64)
    assert cu_seqlen_ke.dtype in (torch.int32, torch.int64)
    assert k_block_size >= 1

    logits = torch.empty(
        (seq_len, num_pool),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    if seq_len == 0 or num_pool == 0:
        return logits

    # BLOCK_N picks: same heuristic as batch_pool_mqa — 128 when nb is large
    # (lifts WGMMA m=64→128 throughput) else 64 (small grid, prefer more
    # CTAs across SMs). Conservative compared to tilelang's 256 (which has
    # spill issues at small num_pool); can tune later.
    if BLOCK_N is None:
        BLOCK_N = 128 if num_pool >= 256 else 64

    grid = (seq_len, triton.cdiv(num_pool, BLOCK_N))
    _ragged_pool_mqa_kernel[grid](
        q_fp8,
        blocked_k_fp8,
        blocked_k_scale,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        blocked_k_fp8.stride(0),
        blocked_k_fp8.stride(1),
        blocked_k_scale.stride(0),
        logits.stride(0),
        logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        cu_seqlen_ks.stride(0),
        cu_seqlen_ke.stride(0),
        num_pool,
        HEADS=H,
        DIM=D,
        BLOCK_N=BLOCK_N,
        K_BLOCK_SIZE=k_block_size,
    )
    return logits


# ---------------------------------------------------------------------------
# Kernel 2: sparse paged block-MQA — THE 80% HOTSPOT
#   Mirrors: fp8_native_paged_block_sparse_mqa_attn_return_logits (tilelang)
#   Shapes:
#     IndexQ:       [B, seq_len, H, D]               fp8
#     KvCache_fp8:  [num_phys_blocks, paged_block_size * (D + 4)]  fp8 bytes
#     KvCache_fp32: [num_phys_blocks, paged_block_size * (D + 4) // 4] f32
#                   (same memory, f32 view for the per-token scale slots)
#     TopK:         [B, seq_len, topk]               i32 / i64
#     Weights:      [B, seq_len, H]                  f32
#     ContextLens:  [B]                              i32
#     BlockTables:  [B, max_blocks]                  i32
#     Logits:       [B, seq_len, topk * kv_block_size] f32  OUT
#
#   Single grouped kernel for K∈{8,16,32,64,128}: per-row paged lookup
#   (logical_page = token_abs // PAGED) handles K<paged, K==paged, K>paged
#   uniformly. GEMM_TILE = max(paged, K) per CTA.
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_paged_mqa_grouped_kernel(
    Q_ptr,
    KvCacheFp8_ptr,
    KvCacheFp32_ptr,
    TopK_ptr,
    Logits_ptr,
    W_ptr,
    ContextLens_ptr,
    BlockTables_ptr,
    stride_q_b,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_kv8_p,
    stride_kv8_b,
    stride_kv32_p,
    stride_kv32_b,
    stride_topk_b,
    stride_topk_s,
    stride_topk_n,
    stride_logits_b,
    stride_logits_s,
    stride_logits_n,
    stride_w_b,
    stride_w_s,
    stride_w_h,
    stride_bt_b,
    stride_bt_mb,
    max_blocks,
    num_phys,  # kv_cache.shape[0] — for phys bound (defensive)
    topk,  # runtime int, may not divide GROUP_SIZE
    PAGED_BLOCK_SIZE: tl.constexpr,  # 64
    KV_BLOCK_SIZE: tl.constexpr,  # k_block_size, must divide PAGED_BLOCK_SIZE
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # PAGED_BLOCK_SIZE // KV_BLOCK_SIZE  >= 2
):
    """Grouped variant for k_block_size < paged_block_size=64.

    GROUP_SIZE consecutive topk indices fuse into one [64,D] GEMM tile.
    Each group occupies one paged page (since k_block divides 64 and each
    group is k_block tokens). Per-row gather picks the right phys page
    for each row.

    Tail-chunk handling for non-divisible topk: invalid g lanes load topk_id
    masked with other=0 (safe phys index), then ANDed into ``pos_valid`` so
    their logits become -inf; output store uses ``out_cols < topk*K`` mask.
    """
    GEMM_TILE: tl.constexpr = KV_BLOCK_SIZE * GROUP_SIZE  # = PAGED_BLOCK_SIZE = 64
    SCALE_OFFSET: tl.constexpr = PAGED_BLOCK_SIZE * DIM // 4

    b = tl.program_id(0)
    seq_i = tl.program_id(1)
    chunk_idx = tl.program_id(2)
    n_i_start = chunk_idx * GROUP_SIZE

    # --- Q [H, D] fp8 (shared across the group), W [H] f32 ---
    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q = tl.load(
        Q_ptr
        + b * stride_q_b
        + seq_i * stride_q_s
        + h_offs[:, None] * stride_q_h
        + d_offs[None, :] * stride_q_d
    )
    w = tl.load(W_ptr + b * stride_w_b + seq_i * stride_w_s + h_offs * stride_w_h)

    # --- Per-group topk_block_ids ---
    g_offs = tl.arange(0, GROUP_SIZE)  # [G]
    g_idx = n_i_start + g_offs  # [G] absolute topk pos
    g_mask = g_idx < topk  # [G] False for tail-pad lanes
    topk_block_ids = tl.load(
        TopK_ptr + b * stride_topk_b + seq_i * stride_topk_s + g_idx * stride_topk_n,
        mask=g_mask,
        other=0,  # safe (page 0); pos_valid AND'd with g_valid below
    ).to(
        tl.int32
    )  # [G]
    block_s_per_g = topk_block_ids * KV_BLOCK_SIZE  # [G] first token of each group

    # --- Per-row token positions ---
    b_offs = tl.arange(0, KV_BLOCK_SIZE)  # [k]
    token_abs_2d = block_s_per_g[:, None] + b_offs[None, :]  # [G, k]
    # Per-row logical page lookup: works for both KV_BLOCK_SIZE <= PAGED
    # (group within 1 page) AND KV_BLOCK_SIZE > PAGED (group across pages,
    # e.g. k=128 G=1 spans 2 paged pages). Triton's compiler hoists the
    # constant-within-group portion when KV_BLOCK_SIZE <= PAGED.
    logical_page_2d = token_abs_2d // PAGED_BLOCK_SIZE  # [G, k]
    row_within_page_2d = token_abs_2d % PAGED_BLOCK_SIZE  # [G, k]
    valid_page_2d = (logical_page_2d >= 0) & (logical_page_2d < max_blocks)

    # Flatten to [GEMM_TILE].
    token_abs = tl.reshape(token_abs_2d, (GEMM_TILE,))
    logical_page = tl.reshape(logical_page_2d, (GEMM_TILE,))
    row_within_page = tl.reshape(row_within_page_2d, (GEMM_TILE,))
    valid_page = tl.reshape(valid_page_2d, (GEMM_TILE,))

    # Per-row phys lookup (same phys for rows in the same logical page).
    phys_per_row = tl.load(
        BlockTables_ptr + b * stride_bt_b + logical_page * stride_bt_mb,
        mask=valid_page,
        other=0,
    ).to(
        tl.int32
    )  # [GEMM_TILE]

    # Defensive: clamp phys to [0, num_phys) — if BlockTables has any
    # spurious value (stale, allocator race) this prevents OOB on the K
    # load below. SK15 / SK16 have the same defense.
    valid_per_row = valid_page & (phys_per_row >= 0) & (phys_per_row < num_phys)
    safe_phys = tl.where(valid_per_row, phys_per_row, 0)
    safe_row = tl.where(valid_per_row, row_within_page, 0)

    # --- K tile [GEMM_TILE, D] fp8 (per-row gather) ---
    k = tl.load(
        KvCacheFp8_ptr
        + safe_phys[:, None] * stride_kv8_p
        + (safe_row[:, None] * DIM + d_offs[None, :]) * stride_kv8_b
    )  # fp8 [GEMM_TILE, D]
    # --- K per-token scale [GEMM_TILE] f32 ---
    k_scale = tl.load(
        KvCacheFp32_ptr
        + safe_phys * stride_kv32_p
        + (SCALE_OFFSET + safe_row) * stride_kv32_b
    )  # f32 [GEMM_TILE]

    # --- GEMM: [GEMM_TILE, D] @ [D, H] = [GEMM_TILE, H] ---
    s = tl.dot(k, q.trans(1, 0), out_dtype=tl.float32)
    s = s * k_scale[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]
    logits = tl.sum(s, axis=1)  # [GEMM_TILE] f32

    # --- Mask: token_abs >= 0, < context_len, valid page+phys, g_valid ---
    # g_valid: per-lane copy of g_mask (broadcast [G] → [GEMM_TILE]). Needed
    # because for tail-pad lanes we used other=0, which routes them to a real
    # page 0 — without g_valid they'd get a real GEMM result instead of -inf.
    g_valid_2d = tl.broadcast_to(g_mask[:, None], (GROUP_SIZE, KV_BLOCK_SIZE))
    g_valid = tl.reshape(g_valid_2d, (GEMM_TILE,))
    context_len = tl.load(ContextLens_ptr + b)
    pos_valid = (token_abs >= 0) & (token_abs < context_len) & valid_per_row & g_valid
    logits = tl.where(pos_valid, logits, float("-inf"))

    # --- Store: rows are contiguous in logits[..., n_i_start*k : n_i_start*k + GEMM_TILE] ---
    bn_offs = tl.arange(0, GEMM_TILE)
    out_cols = n_i_start * KV_BLOCK_SIZE + bn_offs
    out_mask = out_cols < topk * KV_BLOCK_SIZE  # skip tail-pad OOB writes
    tl.store(
        Logits_ptr
        + b * stride_logits_b
        + seq_i * stride_logits_s
        + out_cols * stride_logits_n,
        logits,
        mask=out_mask,
    )


def sparse_paged_mqa_triton(
    q_fp8: torch.Tensor,  # [B, seq, H, D] fp8
    kv_cache_fp8: torch.Tensor,  # [num_phys, paged_block_size, 1, D+4] uint8 (as returned by sglang's NSATokenToKVPool)
    topk_block_index: torch.Tensor,  # [B, seq, topk] i64
    kv_block_size: int,
    weights: torch.Tensor,  # [B, seq, H] or [B*seq, H] f32
    context_lens: torch.Tensor,  # [B] i32
    block_tables: torch.Tensor,  # [B, max_blocks] i32
) -> torch.Tensor:
    """Triton equivalent of ``fp8_native_paged_block_sparse_mqa_attn_return_logits_interface``.
    Returns logits of shape ``[B, seq, topk * kv_block_size]`` f32.

    Unified grouped dispatch for K∈{8,16,32,64,128}. The grouped kernel's
    per-row paged lookup (``logical_page = token_abs // PAGED``) handles all
    K layouts uniformly:
      K<paged (8/16/32):   GROUP_SIZE=64/K, GEMM_TILE=64 — G consecutive
                           topk indices share one paged page.
      K==paged (64):       GROUP_SIZE=1,    GEMM_TILE=64 — one topk = one page.
      K>paged (128):       GROUP_SIZE=1,    GEMM_TILE=128 — one topk spans
                           2 pages, per-row gather looks up each.
    """
    assert q_fp8.ndim == 4, f"q_fp8 should be [B, seq, H, D], got {q_fp8.shape}"
    B, seq_len, H, D = q_fp8.shape
    topk = int(topk_block_index.shape[-1])
    num_phys_blocks, paged_block_size, _, DPlus4 = kv_cache_fp8.shape
    assert _ == 1 and DPlus4 == D + 4
    max_blocks = int(block_tables.shape[-1])

    # i32 or i64 — Triton load does .to(tl.int32) internally either way.
    # Accepting i32 lets us route fast_topk_runtime output through without
    # a redundant 64→32 cast.
    assert topk_block_index.dtype in (torch.int32, torch.int64)
    assert context_lens.dtype == torch.int32
    assert block_tables.dtype == torch.int32
    assert kv_block_size in (8, 16, 32, 64, 128), (
        f"unsupported kv_block_size={kv_block_size}; expected one of "
        "{8,16,32,64,128}"
    )

    # Views (no data copy).
    kv_cache_flat = kv_cache_fp8.view(num_phys_blocks, -1)  # uint8 [num_phys, P*(D+4)]
    kv_fp8_view = kv_cache_flat.view(torch.float8_e4m3fn)  # fp8 same layout
    kv_f32_view = kv_cache_flat.view(torch.float32)  # f32 view

    if weights.ndim == 2:
        weights = weights.view(B, seq_len, H)

    logits = torch.empty(
        (B, seq_len, topk * kv_block_size),
        device=q_fp8.device,
        dtype=torch.float32,
    )

    # GEMM_TILE = max(paged, K). GROUP_SIZE picked so GEMM_TILE = G*K is the
    # natural tile (one paged page for K<=paged, one topk index for K>paged):
    #   K<paged (8/16/32):  G = paged/K → TILE = paged = 64
    #   K==paged (64):      G = 1       → TILE = paged = 64
    #   K>paged (128):      G = 1       → TILE = K = 128
    if kv_block_size <= paged_block_size:
        GROUP_SIZE = paged_block_size // kv_block_size
    else:
        GROUP_SIZE = 1
    GEMM_TILE = GROUP_SIZE * kv_block_size
    assert GEMM_TILE % kv_block_size == 0
    # Non-divisible topk: ceil to num_chunks; tail-pad lanes are handled
    # in-kernel via masked load (other=0) + g_valid AND in pos_valid +
    # out_cols < topk*K store mask.
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE
    grid = (B, seq_len, num_chunks)
    _sparse_paged_mqa_grouped_kernel[grid](
        q_fp8,
        kv_fp8_view,
        kv_f32_view,
        topk_block_index,
        logits,
        weights,
        context_lens,
        block_tables,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        q_fp8.stride(3),
        kv_fp8_view.stride(0),
        kv_fp8_view.stride(1),
        kv_f32_view.stride(0),
        kv_f32_view.stride(1),
        topk_block_index.stride(0),
        topk_block_index.stride(1),
        topk_block_index.stride(2),
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        block_tables.stride(0),
        block_tables.stride(1),
        max_blocks,
        num_phys_blocks,
        topk,
        PAGED_BLOCK_SIZE=paged_block_size,
        KV_BLOCK_SIZE=kv_block_size,
        HEADS=H,
        DIM=D,
        GROUP_SIZE=GROUP_SIZE,
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
    Q_ptr,
    K_ptr,
    KS_ptr,
    TopK_ptr,
    Logits_ptr,
    W_ptr,
    CuKS_ptr,
    CuKE_ptr,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_d,
    stride_ks_s,
    stride_topk_s,
    stride_topk_n,
    stride_logits_s,
    stride_logits_n,
    stride_w_s,
    stride_w_h,
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

    topk_id = tl.load(TopK_ptr + seq_i * stride_topk_s + n_i * stride_topk_n).to(
        tl.int32
    )
    block_s_i = topk_id * KV_BLOCK_SIZE + sub_i * BLOCK_N

    # Load Q [H, D].
    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q = tl.load(
        Q_ptr
        + seq_i * stride_q_s
        + h_offs[:, None] * stride_q_h
        + d_offs[None, :] * stride_q_d
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


@triton.jit
def _block_sparse_mqa_grouped_kernel(
    Q_ptr,
    K_ptr,
    KS_ptr,
    TopK_ptr,
    Logits_ptr,
    W_ptr,
    CuKS_ptr,
    CuKE_ptr,
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_d,
    stride_ks_s,
    stride_topk_s,
    stride_topk_n,
    stride_logits_s,
    stride_logits_n,
    stride_w_s,
    stride_w_h,
    seq_kv,
    topk,  # runtime int, may not divide GROUP_SIZE
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,  # k_block_size, < 64
    GROUP_SIZE: tl.constexpr,  # 64 // KV_BLOCK_SIZE
):
    """Grouped variant for prefill block-sparse MQA when k<64.

    G consecutive topk indices fuse into one [GEMM_TILE=64, D] GEMM tile.
    Per-row gather from ragged K (no page lookup — K is a flat
    [seq_kv, D] tensor).

    Tail-chunk handling for non-divisible topk: invalid g lanes load topk_id
    via masked tl.load with -1 sentinel → k_rows ∈ [-K, 0) → existing
    `k_mask = (k_rows >= 0) ...` masks them; `pos_valid` (k_rows < ks_min)
    forces -inf; output store uses an `out_cols < topk*K` mask to skip OOB.
    """
    GEMM_TILE: tl.constexpr = KV_BLOCK_SIZE * GROUP_SIZE  # = 64
    seq_i = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    n_i_start = chunk_idx * GROUP_SIZE

    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q = tl.load(
        Q_ptr
        + seq_i * stride_q_s
        + h_offs[:, None] * stride_q_h
        + d_offs[None, :] * stride_q_d
    )  # fp8 [H, D]
    w = tl.load(W_ptr + seq_i * stride_w_s + h_offs * stride_w_h)  # f32 [H]

    # G topk_block_ids, broadcast to G*k absolute K rows.
    g_offs = tl.arange(0, GROUP_SIZE)
    g_idx = n_i_start + g_offs  # [G] absolute topk pos
    g_mask = g_idx < topk  # [G] False for tail-pad lanes
    topk_block_ids = tl.load(
        TopK_ptr + seq_i * stride_topk_s + g_idx * stride_topk_n,
        mask=g_mask,
        other=-1,
    ).to(
        tl.int32
    )  # [G]
    b_offs = tl.arange(0, KV_BLOCK_SIZE)
    k_rows_2d = topk_block_ids[:, None] * KV_BLOCK_SIZE + b_offs[None, :]  # [G, k]
    k_rows = tl.reshape(k_rows_2d, (GEMM_TILE,))  # [GEMM_TILE]

    k_mask = (k_rows >= 0) & (k_rows < seq_kv)
    safe_rows = tl.where(k_mask, k_rows, 0)

    k = tl.load(
        K_ptr + safe_rows[:, None] * stride_k_s + d_offs[None, :] * stride_k_d
    )  # fp8 [GEMM_TILE, D]
    ks = tl.load(
        KS_ptr + safe_rows * stride_ks_s,
        mask=k_mask,
        other=0.0,
    )  # f32 [GEMM_TILE]

    s = tl.dot(k, q.trans(1, 0), out_dtype=tl.float32)  # [GEMM_TILE, H]
    s = s * ks[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]
    logits = tl.sum(s, axis=1)  # [GEMM_TILE]

    ks_min = tl.load(CuKS_ptr + seq_i)
    ke_max = tl.load(CuKE_ptr + seq_i)
    pos_valid = (k_rows >= ks_min) & (k_rows < ke_max) & k_mask
    logits = tl.where(pos_valid, logits, float("-inf"))

    bn_offs = tl.arange(0, GEMM_TILE)
    out_cols = n_i_start * KV_BLOCK_SIZE + bn_offs
    out_mask = out_cols < topk * KV_BLOCK_SIZE  # skip tail-pad OOB writes
    tl.store(
        Logits_ptr + seq_i * stride_logits_s + out_cols * stride_logits_n,
        logits,
        mask=out_mask,
    )


@triton.jit
def _block_sparse_mqa_persistent_kernel(
    Q_ptr,
    K_ptr,
    KS_ptr,
    TopK_ptr,
    Logits_ptr,
    W_ptr,
    CuKS_ptr,
    CuKE_ptr,
    stride_q_s,
    stride_q_h,
    stride_k_s,
    stride_topk_s,
    stride_logits_s,
    stride_w_s,
    seq_kv,
    topk,  # runtime int, may not divide GROUP_SIZE
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,  # k_block_size, < 128
    GROUP_SIZE: tl.constexpr,  # GEMM_TILE // KV_BLOCK_SIZE
    K_CHUNKS: tl.constexpr,  # chunks per CTA (persistent inner-loop trip count)
):
    """Persistent variant of ``_block_sparse_mqa_grouped_kernel``.

    Each CTA iterates ``K_CHUNKS`` consecutive grouped chunks of the same
    seq instead of just one — Q[H,D] and W[H] (8.25KB / row) are loaded
    *once* and reused across the inner loop, dropping ~K_CHUNKS× redundant
    HBM traffic. ``tl.range(num_stages=N)`` lets the K-tile gather of
    iter i+1 overlap the GEMM of iter i, hiding gather latency.

    Grid: ``(seq_q, ceil(num_chunks_per_seq / K_CHUNKS))``. Tail chunks
    (where ``chunk_idx >= num_chunks_per_seq``) are handled by the same
    masks as the non-persistent kernel:
      - ``g_mask = g_idx < topk`` → load -1 sentinel for invalid lanes
      - ``k_mask = (k_rows >= 0) & ...`` and ``pos_valid`` force -inf
      - output ``out_mask = out_cols < topk*K`` skips OOB writes
    so the iteration is wasted but produces no incorrect side effects.
    """
    GEMM_TILE: tl.constexpr = KV_BLOCK_SIZE * GROUP_SIZE
    seq_i = tl.program_id(0)
    outer = tl.program_id(1)

    # Loaded ONCE — reused across K_CHUNKS inner iters.
    # Innermost strides (q_d, k_d, ks_s, topk_n, logits_n, w_h) are always 1
    # by torch contiguous-tensor convention — folded as compile-time 1 here
    # to drop the per-element runtime multiply (saves ~GEMM_TILE*D ALU ops).
    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q = tl.load(
        Q_ptr + seq_i * stride_q_s + h_offs[:, None] * stride_q_h + d_offs[None, :]
    )  # fp8 [H, D]
    q_T = q.trans(1, 0)  # [D, H] hoisted out of inner loop
    w = tl.load(W_ptr + seq_i * stride_w_s + h_offs)  # f32 [H]
    ks_min = tl.load(CuKS_ptr + seq_i)
    ke_max = tl.load(CuKE_ptr + seq_i)
    topk_K = topk * KV_BLOCK_SIZE  # hoist multiply out of inner loop

    g_offs = tl.arange(0, GROUP_SIZE)
    b_offs = tl.arange(0, KV_BLOCK_SIZE)
    bn_offs = tl.arange(0, GEMM_TILE)

    for k_iter in tl.range(K_CHUNKS, num_stages=2):
        chunk_idx = outer * K_CHUNKS + k_iter
        n_i_start = chunk_idx * GROUP_SIZE

        g_idx = n_i_start + g_offs  # [G]
        g_mask = g_idx < topk  # [G]
        topk_block_ids = tl.load(
            TopK_ptr + seq_i * stride_topk_s + g_idx,
            mask=g_mask,
            other=-1,
        ).to(tl.int32)
        k_rows_2d = topk_block_ids[:, None] * KV_BLOCK_SIZE + b_offs[None, :]
        k_rows = tl.reshape(k_rows_2d, (GEMM_TILE,))

        # Clamp safe_rows to [0, seq_kv-1] for in-bounds load. Lower clamp
        # handles masked-off slots (topk_block_ids = -1 → k_rows < 0). Upper
        # clamp handles seq_kv % KV_BLOCK_SIZE != 0: the last topk block's
        # tail rows would otherwise exceed seq_kv (OOB on K/KS load — silent
        # within an allocator block, illegal access at page boundary).
        # Garbage from out-of-range loads is masked to -inf via pos_valid below.
        safe_rows = tl.minimum(tl.maximum(k_rows, 0), seq_kv - 1)

        k = tl.load(
            K_ptr + safe_rows[:, None] * stride_k_s + d_offs[None, :]
        )  # fp8 [GEMM_TILE, D]
        ks = tl.load(KS_ptr + safe_rows)  # f32 [GEMM_TILE]

        s = tl.dot(k, q_T, out_dtype=tl.float32)
        s = tl.maximum(s * ks[:, None], 0.0) * w[None, :]
        logits = tl.sum(s, axis=1)

        pos_valid = (k_rows >= ks_min) & (k_rows < ke_max)
        logits = tl.where(pos_valid, logits, float("-inf"))

        out_cols = n_i_start * KV_BLOCK_SIZE + bn_offs
        out_mask = out_cols < topk_K
        tl.store(
            Logits_ptr + seq_i * stride_logits_s + out_cols,
            logits,
            mask=out_mask,
        )


def block_sparse_mqa_triton(
    q_fp8: torch.Tensor,  # [seq, H, D] fp8
    k_fp8: torch.Tensor,  # [seq_kv, D] fp8
    k_scale: torch.Tensor,  # [seq_kv] f32
    topk_block_index: torch.Tensor,  # [seq, topk] i64
    kv_block_size: int,
    weights: torch.Tensor,  # [seq, H] f32
    cu_seqlen_ks: torch.Tensor,  # [seq] i32
    cu_seqlen_ke: torch.Tensor,  # [seq] i32
) -> torch.Tensor:
    """Block-sparse MQA for ragged prefill. Supports kv_block_size in
    {8, 16, 32, 64, 128}.

    Dispatch:
      kv_block_size >= 64 → original kernel; CTA per (seq, topk*sub) with
        BLOCK_N=64 inner tile.
      kv_block_size <  64 → grouped kernel; GROUP_SIZE=64/k consecutive
        topk indices fuse into one [64,D] GEMM tile via row-gather.
    """
    assert q_fp8.ndim == 3
    assert kv_block_size in (8, 16, 32, 64, 128), (
        f"unsupported kv_block_size={kv_block_size}; expected one of "
        "{8,16,32,64,128}"
    )
    seq_len, H, D = q_fp8.shape
    seq_kv = k_fp8.shape[0]
    topk = int(topk_block_index.shape[-1])
    logits = torch.empty(
        (seq_len, topk * kv_block_size),
        device=q_fp8.device,
        dtype=torch.float32,
    )

    K_CONFIG = {
        # (GROUP_SIZE, K_CHUNKS, num_warps, num_stages)
        # K=8 K_CHUNKS=32 (not 64): KC=64 catastrophically regressed at
        # block_topk≤256 (constexpr loop runs full 64 iters, half masked).
        # KC=32 wins universally across block_topk ∈ {256,512,1024,2048}.
        8: (8, 32, 4, 2),
        16: (16, 32, 8, 3),
        32: (8, 32, 8, 3),
        64: (4, 32, 8, 3),
        128: (1, 64, 4, 3),
    }
    GROUP_SIZE, K_CHUNKS, num_warps, num_stages = K_CONFIG[kv_block_size]
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE
    outer = (num_chunks + K_CHUNKS - 1) // K_CHUNKS
    grid = (seq_len, outer)
    # Innermost strides (q.stride(2), k.stride(1), k_scale.stride(0),
    # topk.stride(1), logits.stride(1), weights.stride(1)) are baked as 1
    # in the kernel — assert here.
    assert q_fp8.stride(2) == 1 and k_fp8.stride(1) == 1 and k_scale.stride(0) == 1
    assert (
        topk_block_index.stride(1) == 1
        and logits.stride(1) == 1
        and weights.stride(1) == 1
    )
    _block_sparse_mqa_persistent_kernel[grid](
        q_fp8,
        k_fp8,
        k_scale,
        topk_block_index,
        logits,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        q_fp8.stride(0),
        q_fp8.stride(1),
        k_fp8.stride(0),
        topk_block_index.stride(0),
        logits.stride(0),
        weights.stride(0),
        seq_kv,
        topk,
        HEADS=H,
        DIM=D,
        KV_BLOCK_SIZE=kv_block_size,
        GROUP_SIZE=GROUP_SIZE,
        K_CHUNKS=K_CHUNKS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return logits


# ---------------------------------------------------------------------------
# Kernel 4: v3 decode block-MQA on paged pool_k_pages
#   Mirrors: batch_decode_pool_mqa_attn_return_logits_fp8 (tilelang)
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
def _batch_decode_pool_mqa_kernel(
    Q_ptr,  # [B, H, D] fp8
    PKPagesFp8_ptr,  # [N_pp, PP * (D+4)] fp8
    PKPagesF32_ptr,  # [N_pp, PP * (D+4) // 4] f32
    PageTables_ptr,  # [B, max_pp] i32
    Logits_ptr,  # [B, max_pp * PP] f32 OUT
    W_ptr,  # [B, H] f32
    ContextLensPool_ptr,  # [B] i32
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_pk8_p,
    stride_pk8_b,
    stride_pk32_p,
    stride_pk32_b,
    stride_pt_b,
    stride_pt_n,
    stride_logits_b,
    stride_logits_n,
    stride_w_b,
    stride_w_h,
    num_pool_phys,  # pool_k_pages.shape[0] — for phys bound (defensive)
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    PP: tl.constexpr,  # pool_page_size (= 64)
):
    b = tl.program_id(0)
    lp = tl.program_id(1)  # logical pool page

    phys = tl.load(PageTables_ptr + b * stride_pt_b + lp * stride_pt_n).to(tl.int32)
    # Defensive: clamp pool phys to [0, num_pool_phys). Prevents OOB on the
    # K/scale loads if PageTables holds stale or sentinel values for logical
    # pool pages past the current ContextLensPool watermark.
    phys_valid = (phys >= 0) & (phys < num_pool_phys)
    phys = tl.where(phys_valid, phys, 0)

    h_offs = tl.arange(0, HEADS)
    d_offs = tl.arange(0, DIM)
    q = tl.load(
        Q_ptr
        + b * stride_q_b
        + h_offs[:, None] * stride_q_h
        + d_offs[None, :] * stride_q_d
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
        PKPagesF32_ptr + phys * stride_pk32_p + (SCALE_OFFSET + bn_offs) * stride_pk32_b
    )  # f32 [PP]

    # GEMM + post.
    s = tl.dot(k, q.trans(1, 0), out_dtype=tl.float32)
    s = s * ks[:, None]
    s = tl.maximum(s, 0.0)
    s = s * w[None, :]
    logits = tl.sum(s, axis=1)  # [PP] f32

    # Mask by ContextLensPool: positions >= k_e become -inf, and apply
    # force_maintain (+inf at pos 0 and at k_e - 1). Single nested
    # tl.where → one select chain instead of two.
    k_e = tl.load(ContextLensPool_ptr + b)
    pool_idx = lp * PP + bn_offs
    pos_valid = (pool_idx < k_e) & phys_valid
    pos_maintain = ((pool_idx == 0) | (pool_idx == (k_e - 1))) & phys_valid
    logits = tl.where(
        pos_maintain,
        float("inf"),
        tl.where(pos_valid, logits, float("-inf")),
    )

    tl.store(
        Logits_ptr + b * stride_logits_b + pool_idx * stride_logits_n,
        logits,
    )


def batch_decode_pool_mqa_triton(
    q_fp8: torch.Tensor,  # [B, 1, H, D] fp8
    pool_k_pages: torch.Tensor,  # [N_pp, PP * (D+4)] uint8
    pool_page_tables: torch.Tensor,  # [B, max_pp] i32
    weights_f32: torch.Tensor,  # [B, H] or [B, 1, H] f32
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
    num_pool_phys = pool_k_pages.shape[0]

    logits = torch.empty(
        (B, max_pp * pool_page_size),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    grid = (B, max_pp)
    _batch_decode_pool_mqa_kernel[grid](
        q_2d,
        pk8,
        pk32,
        pool_page_tables,
        logits,
        w_2d,
        context_lens_pool,
        q_2d.stride(0),
        q_2d.stride(1),
        q_2d.stride(2),
        pk8.stride(0),
        pk8.stride(1),
        pk32.stride(0),
        pk32.stride(1),
        pool_page_tables.stride(0),
        pool_page_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        w_2d.stride(0),
        w_2d.stride(1),
        num_pool_phys,
        HEADS=H,
        DIM=D,
        PP=pool_page_size,
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
    KvCacheFp8_ptr,  # [num_phys, paged_block_size * (D+4)] fp8
    KvCacheF32_ptr,  # [num_phys, paged_block_size * (D+4) // 4] f32
    BlockTables_ptr,  # [B, max_blocks] i32
    ContextLens_ptr,  # [B] i32
    BlockedK_ptr,  # [B, max_num_pool, D] fp8 OUT
    BlockedKScale_ptr,  # [B, max_num_pool] f32 OUT
    stride_kv8_p,
    stride_kv8_b,
    stride_kv32_p,
    stride_kv32_b,
    stride_bt_b,
    stride_bt_mb,
    stride_bk_b,
    stride_bk_n,
    stride_bk_d,
    stride_bks_b,
    stride_bks_n,
    max_blocks,
    num_phys,  # kv_cache.shape[0] — for phys bound (defensive)
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
            mask=valid_page,
            other=0,
        ).to(tl.int32)
        # Defensive: clamp phys to [0, num_phys) to prevent OOB on K loads.
        phys_valid = valid_page & (phys >= 0) & (phys < num_phys)
        phys = tl.where(phys_valid, phys, 0)

        # Load K tile [PAGED_BLOCK_SIZE, D] fp8, convert to f32.
        k_byte = bn_offs[:, None] * DIM + d_offs[None, :]
        k_fp8 = tl.load(KvCacheFp8_ptr + phys * stride_kv8_p + k_byte * stride_kv8_b)
        k = k_fp8.to(tl.float32)

        # Load per-token scale [PAGED_BLOCK_SIZE] f32.
        scale = tl.load(
            KvCacheF32_ptr
            + phys * stride_kv32_p
            + (SCALE_OFFSET + bn_offs) * stride_kv32_b,
        )

        # Zero out tokens past k_end (and rows from invalid pages).
        tl_block_idx = paged_block_s + bn_offs
        in_block = (tl_block_idx < k_end) & phys_valid
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
        BlockedK_ptr + b * stride_bk_b + pblk * stride_bk_n + d_offs * stride_bk_d,
        out_fp8,
    )
    tl.store(BlockedKScale_ptr + b * stride_bks_b + pblk * stride_bks_n, block_scale)


@triton.jit
def _paged_mean_pooling_grouped_kernel(
    KvCacheFp8_ptr,
    KvCacheF32_ptr,
    BlockTables_ptr,
    ContextLens_ptr,
    BlockedK_ptr,
    BlockedKScale_ptr,
    stride_kv8_p,
    stride_kv8_b,
    stride_kv32_p,
    stride_kv32_b,
    stride_bt_b,
    stride_bt_mb,
    stride_bk_b,
    stride_bk_n,
    stride_bk_d,
    stride_bks_b,
    stride_bks_n,
    max_blocks,
    num_pool_total,  # for masked-store of last group
    num_phys,  # kv_cache.shape[0] — for phys bound (defensive)
    PAGED_BLOCK_SIZE: tl.constexpr,  # 64 (paged page size)
    POOLING_BLOCK_SIZE: tl.constexpr,  # k_block_size, < 64
    DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # 64 // k_block_size
):
    """Grouped variant for k_block_size < paged_block_size=64.

    Each CTA handles GROUP_SIZE consecutive pool blocks within ONE paged
    page (since GROUP_SIZE * k_block_size = PAGED_BLOCK_SIZE = 64). Loads
    the whole page once, reshapes to [G, k, D], and emits G outputs.
    """
    GEMM_TILE: tl.constexpr = POOLING_BLOCK_SIZE * GROUP_SIZE  # = PAGED_BLOCK_SIZE = 64
    b = tl.program_id(0)
    cta = tl.program_id(1)
    pblk_start = cta * GROUP_SIZE
    k_start = pblk_start * POOLING_BLOCK_SIZE  # = cta * 64, page-aligned

    seq_len = tl.load(ContextLens_ptr + b)
    if k_start >= seq_len:
        return

    # Logical page == cta because k_start is a multiple of 64.
    logical_page = k_start // PAGED_BLOCK_SIZE
    valid_page = (logical_page >= 0) & (logical_page < max_blocks)
    phys = tl.load(
        BlockTables_ptr + b * stride_bt_b + logical_page * stride_bt_mb,
        mask=valid_page,
        other=0,
    ).to(tl.int32)
    # Defensive: clamp phys to [0, num_phys) before use as offset.
    phys_valid = valid_page & (phys >= 0) & (phys < num_phys)
    phys = tl.where(phys_valid, phys, 0)

    d_offs = tl.arange(0, DIM)
    bn_offs = tl.arange(0, GEMM_TILE)
    rows = k_start + bn_offs
    in_seq = (rows < seq_len) & phys_valid

    # Single page-load of K [GEMM_TILE=64, D] fp8.
    k = tl.load(
        KvCacheFp8_ptr
        + phys * stride_kv8_p
        + (bn_offs[:, None] * DIM + d_offs[None, :]) * stride_kv8_b
    ).to(tl.float32)

    SCALE_OFFSET: tl.constexpr = PAGED_BLOCK_SIZE * DIM // 4
    scale = tl.load(
        KvCacheF32_ptr
        + phys * stride_kv32_p
        + (SCALE_OFFSET + bn_offs) * stride_kv32_b,
    )

    k = tl.where(in_seq[:, None], k * scale[:, None], 0.0)
    in_seq_i32 = in_seq.to(tl.int32)

    # Reshape [GEMM_TILE, D] → [G, k, D] and reduce per pool block.
    k_g = tl.reshape(k, (GROUP_SIZE, POOLING_BLOCK_SIZE, DIM))
    cnt_g = tl.reshape(in_seq_i32, (GROUP_SIZE, POOLING_BLOCK_SIZE))
    sum_per_block = tl.sum(k_g, axis=1)  # [G, D]
    cnt_per_block = tl.sum(cnt_g, axis=1).to(tl.float32)
    inv_cnt = tl.where(cnt_per_block > 0, 1.0 / cnt_per_block, 0.0)
    mean = sum_per_block * inv_cnt[:, None]

    max_abs = tl.max(tl.abs(mean), axis=1)
    block_scale = tl.maximum(max_abs * (1.0 / 448.0), 1e-10)
    out_fp8 = (mean * (1.0 / block_scale)[:, None]).to(tl.float8e4nv)

    pblk_idxs = pblk_start + tl.arange(0, GROUP_SIZE)
    valid = pblk_idxs < num_pool_total
    tl.store(
        BlockedK_ptr
        + b * stride_bk_b
        + pblk_idxs[:, None] * stride_bk_n
        + d_offs[None, :] * stride_bk_d,
        out_fp8,
        mask=valid[:, None],
    )
    tl.store(
        BlockedKScale_ptr + b * stride_bks_b + pblk_idxs * stride_bks_n,
        block_scale,
        mask=valid,
    )


def paged_mean_pooling_triton(
    max_num_pooling_blocks: int,
    kv_cache: torch.Tensor,  # [num_phys, paged_block_size, 1, D+4] uint8
    context_lens: torch.Tensor,  # [B] i32
    block_tables: torch.Tensor,  # [B, max_blocks] i32
    k_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean-pool from paged KV cache. Supports k_block_size in
    {8,16,32,64,128}.

    Dispatch:
      k_block_size >= paged_block_size (= 64) → original kernel; one CTA
        per pool block, inner loop over CHUNKS = k/64 paged pages.
      k_block_size <  paged_block_size       → grouped kernel; one CTA
        per (b, page) handling GROUP_SIZE = 64/k pool blocks from a
        single paged page (single page load + per-block reshape).
    """
    assert k_block_size in (8, 16, 32, 64, 128), (
        f"unsupported k_block_size={k_block_size}; expected one of " "{8,16,32,64,128}"
    )
    num_phys, paged_block_size, head, DPlus4 = kv_cache.shape
    assert head == 1
    D = DPlus4 - 4
    B, max_blocks = block_tables.shape

    kv_flat = kv_cache.view(num_phys, paged_block_size * DPlus4)

    blocked_k = torch.empty(
        (B, max_num_pooling_blocks, D),
        device=kv_cache.device,
        dtype=torch.float8_e4m3fn,
    )
    blocked_k_scale = torch.empty(
        (B, max_num_pooling_blocks),
        device=kv_cache.device,
        dtype=torch.float32,
    )

    if k_block_size < paged_block_size:
        # Grouped path — one CTA per (b, page) processes G pool blocks.
        GROUP_SIZE = paged_block_size // k_block_size  # 8/4/2 for k=8/16/32
        assert paged_block_size % k_block_size == 0
        num_groups = (max_num_pooling_blocks + GROUP_SIZE - 1) // GROUP_SIZE
        grid = (B, num_groups)
        _paged_mean_pooling_grouped_kernel[grid](
            kv_flat.view(torch.float8_e4m3fn),
            kv_flat.view(torch.float32),
            block_tables,
            context_lens,
            blocked_k,
            blocked_k_scale,
            kv_flat.stride(0),
            kv_flat.stride(1),
            (kv_flat.view(torch.float32)).stride(0),
            (kv_flat.view(torch.float32)).stride(1),
            block_tables.stride(0),
            block_tables.stride(1),
            blocked_k.stride(0),
            blocked_k.stride(1),
            blocked_k.stride(2),
            blocked_k_scale.stride(0),
            blocked_k_scale.stride(1),
            max_blocks,
            max_num_pooling_blocks,
            num_phys,
            PAGED_BLOCK_SIZE=paged_block_size,
            POOLING_BLOCK_SIZE=k_block_size,
            DIM=D,
            GROUP_SIZE=GROUP_SIZE,
        )
    else:
        grid = (B, max_num_pooling_blocks)
        _paged_mean_pooling_kernel[grid](
            kv_flat.view(torch.float8_e4m3fn),
            kv_flat.view(torch.float32),
            block_tables,
            context_lens,
            blocked_k,
            blocked_k_scale,
            kv_flat.stride(0),
            kv_flat.stride(1),
            (kv_flat.view(torch.float32)).stride(0),
            (kv_flat.view(torch.float32)).stride(1),
            block_tables.stride(0),
            block_tables.stride(1),
            blocked_k.stride(0),
            blocked_k.stride(1),
            blocked_k.stride(2),
            blocked_k_scale.stride(0),
            blocked_k_scale.stride(1),
            max_blocks,
            num_phys,
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
    K_ptr,  # [seq_kv, D] fp8
    KScale_ptr,  # [seq_kv] f32
    BlockedK_ptr,  # [num_pool, D] fp8 OUT
    BlockedKS_ptr,  # [num_pool] f32 OUT
    stride_k_s,
    stride_k_d,
    stride_ks_s,
    stride_bk_n,
    stride_bk_d,
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


@triton.jit
def _block_mean_pooling_grouped_kernel(
    K_ptr,  # [seq_kv, D]   fp8
    KScale_ptr,  # [seq_kv]      f32
    BlockedK_ptr,  # [num_pool, D] fp8 OUT
    BlockedKS_ptr,  # [num_pool]    f32 OUT
    stride_k_s,
    stride_k_d,
    stride_ks_s,
    stride_bk_n,
    stride_bk_d,
    stride_bks_n,
    seq_kv,
    num_pool_total,  # for masked store of last group
    POOLING_BLOCK_SIZE: tl.constexpr,  # k_block_size, < 64
    DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # 64 // k_block_size; CTA handles G blocks
):
    """Grouped variant: GROUP_SIZE consecutive pool blocks per CTA.

    GEMM_TILE = POOLING_BLOCK_SIZE * GROUP_SIZE = 64 always, so the K load
    and reductions stay at full 64-row WGMMA-friendly tile size even when
    k_block_size is 8/16/32. Each pool block keeps its own per-block mean
    + per-block fp8 max-abs scale (independent quantization).
    """
    GEMM_TILE: tl.constexpr = POOLING_BLOCK_SIZE * GROUP_SIZE  # = 64
    cta = tl.program_id(0)
    pblk_start = cta * GROUP_SIZE
    k_start = pblk_start * POOLING_BLOCK_SIZE  # token offset of first row

    d_offs = tl.arange(0, DIM)
    bn_offs = tl.arange(0, GEMM_TILE)
    rows = k_start + bn_offs  # [GEMM_TILE]
    in_seq = rows < seq_kv
    safe_rows = tl.where(in_seq, rows, 0)

    # Single coalesced [64, D] load shared across all GROUP_SIZE blocks.
    k = tl.load(
        K_ptr + safe_rows[:, None] * stride_k_s + d_offs[None, :] * stride_k_d
    ).to(
        tl.float32
    )  # [GEMM_TILE, D]
    scale = tl.load(
        KScale_ptr + safe_rows * stride_ks_s,
        mask=in_seq,
        other=0.0,
    )  # [GEMM_TILE]

    # Apply per-token scale; zero out OOB rows so they contribute nothing
    # to either sum or count.
    k = tl.where(in_seq[:, None], k * scale[:, None], 0.0)
    in_seq_i32 = in_seq.to(tl.int32)

    # Reshape to [GROUP_SIZE, POOLING_BLOCK_SIZE, D] and reduce per block.
    k_g = tl.reshape(k, (GROUP_SIZE, POOLING_BLOCK_SIZE, DIM))
    cnt_g = tl.reshape(in_seq_i32, (GROUP_SIZE, POOLING_BLOCK_SIZE))

    sum_per_block = tl.sum(k_g, axis=1)  # [GROUP_SIZE, D]
    cnt_per_block = tl.sum(cnt_g, axis=1).to(tl.float32)  # [GROUP_SIZE]
    inv_cnt = tl.where(cnt_per_block > 0, 1.0 / cnt_per_block, 0.0)
    mean = sum_per_block * inv_cnt[:, None]  # [GROUP_SIZE, D]

    # Per-block fp8 max-abs quantization.
    max_abs = tl.max(tl.abs(mean), axis=1)  # [GROUP_SIZE]
    block_scale = tl.maximum(max_abs * (1.0 / 448.0), 1e-10)
    out_fp8 = (mean * (1.0 / block_scale)[:, None]).to(tl.float8e4nv)

    # Masked store of GROUP_SIZE outputs (last group may straddle num_pool_total).
    pblk_idxs = pblk_start + tl.arange(0, GROUP_SIZE)
    valid = pblk_idxs < num_pool_total
    tl.store(
        BlockedK_ptr + pblk_idxs[:, None] * stride_bk_n + d_offs[None, :] * stride_bk_d,
        out_fp8,
        mask=valid[:, None],
    )
    tl.store(
        BlockedKS_ptr + pblk_idxs * stride_bks_n,
        block_scale,
        mask=valid,
    )


def block_mean_pooling_triton(
    k_fp8: torch.Tensor,  # [seq_kv, D] fp8
    k_scale: torch.Tensor,  # [seq_kv] f32
    k_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool fp8 K into k_block_size-token blocks. Supports
    k_block_size in {8, 16, 32, 64, 128}.

    Dispatch:
      k_block_size >= 64 → original kernel (chunked load BLOCK_N=64, the
        loop fully covers POOLING_BLOCK_SIZE in CHUNKS = k/64 iterations).
      k_block_size <  64 → grouped kernel, GROUP_SIZE = 64/k pool blocks
        per CTA; one [64, D] coalesced load + per-block reshape/reduce.
        Avoids loading sub-64-row tiles.
    """
    assert k_block_size in (8, 16, 32, 64, 128), (
        f"unsupported k_block_size={k_block_size}; expected one of " "{8,16,32,64,128}"
    )
    seq_kv, D = k_fp8.shape
    max_num_pool = (seq_kv + k_block_size - 1) // k_block_size
    blocked_k = torch.empty(
        (max_num_pool, D),
        device=k_fp8.device,
        dtype=torch.float8_e4m3fn,
    )
    blocked_k_scale = torch.empty(
        (max_num_pool,),
        device=k_fp8.device,
        dtype=torch.float32,
    )

    if k_block_size < 64:
        # Grouped path.
        GROUP_SIZE = 64 // k_block_size  # 8/4/2 for k=8/16/32
        num_groups = (max_num_pool + GROUP_SIZE - 1) // GROUP_SIZE
        grid = (num_groups,)
        _block_mean_pooling_grouped_kernel[grid](
            k_fp8,
            k_scale,
            blocked_k,
            blocked_k_scale,
            k_fp8.stride(0),
            k_fp8.stride(1),
            k_scale.stride(0),
            blocked_k.stride(0),
            blocked_k.stride(1),
            blocked_k_scale.stride(0),
            seq_kv,
            max_num_pool,
            POOLING_BLOCK_SIZE=k_block_size,
            DIM=D,
            GROUP_SIZE=GROUP_SIZE,
        )
    else:
        # Original path (k=64 or 128).  SK9: inner chunk BLOCK_N = min(128, k)
        # so k=128 fits in one [128,D] load (CHUNKS=1) vs the prior
        # BLOCK_N=64 / CHUNKS=2 layout — saves 11-17% across seq_kv.
        BLOCK_N = min(128, k_block_size)
        assert k_block_size % BLOCK_N == 0
        grid = (max_num_pool,)
        _block_mean_pooling_kernel[grid](
            k_fp8,
            k_scale,
            blocked_k,
            blocked_k_scale,
            k_fp8.stride(0),
            k_fp8.stride(1),
            k_scale.stride(0),
            blocked_k.stride(0),
            blocked_k.stride(1),
            blocked_k_scale.stride(0),
            seq_kv,
            POOLING_BLOCK_SIZE=k_block_size,
            DIM=D,
            BLOCK_N=BLOCK_N,
        )
    return blocked_k, blocked_k_scale


# ---------------------------------------------------------------------------
# Kernel 7: completed-blocks pool update (v3 paged write target)
#   Mirrors: fp8_native_paged_mean_pooling_completed_blocks (tilelang)
#
#   Each step writes a small slice of "newly completed" mean-pool blocks
#   into ``pool_k_pages``:
#     prev_complete = PrevSeqLens // K
#     new_complete  = NewSeqLens  // K
#     for pblk_rel in [0, new_complete - prev_complete):
#         pblk_abs   = prev_complete + pblk_rel
#         logical_pp = pblk_abs // pool_page_size
#         slot       = pblk_abs %  pool_page_size
#         phys       = pool_page_tables[req_idx, logical_pp]
#         mean K=k_block_size tokens of K, fp8-quantize, write to
#         pool_k_pages[phys, slot * D : (slot+1) * D]  (fp8)
#         pool_k_pages[phys, scale_offset + slot]      (f32)
#
#   Tile / grouping (SK15):
#     For K >= paged_block_size (= 64): GEMM_TILE = K, GROUP_SIZE = 1.
#       1 CTA per (b, pblk_rel). Loads K tokens (= K/64 paged pages) via
#       per-row gather of (phys_page, row_within_page) from ReqToToken.
#     For K <  paged_block_size: GEMM_TILE = 64, GROUP_SIZE = 64/K.
#       1 CTA per (b, group) processes G consecutive pblk_rel sharing one
#       64-token K load (per-row gather across at most 2 paged pages),
#       then reshape [GROUP_SIZE, K, D] and reduce per pool block.
#
#   Grid: (batch, ceildiv(max_pool_per_req, GROUP_SIZE)). Cells beyond
#   ``new_complete - prev_complete`` are masked out at write time. The
#   kernel reads PrevSeqLens/NewSeqLens at runtime so it's safe to capture
#   under CUDA graph (grid is fixed; no Python branches on tensor values).
# ---------------------------------------------------------------------------


@triton.jit
def _update_pool_for_completed_blocks_kernel(
    KvCacheFp8_ptr,  # [num_phys, paged*(D+4)] fp8 view
    KvCacheFp32_ptr,  # f32 view of same
    ReqToToken_ptr,  # [max_req, max_ctx] i32 — req → main KV pos
    PoolPageTables_ptr,  # [max_req, max_pool_pages] i32 — req → pool phys page
    ReqPoolIndices_ptr,  # [B] i64
    PrevSeqLens_ptr,  # [B] i32
    NewSeqLens_ptr,  # [B] i32
    PoolKPagesFp8_ptr,  # [num_pool_pages, page_bytes] fp8 view  IN-OUT
    PoolKPagesFp32_ptr,  # f32 view of same                       IN-OUT
    stride_kv8_p,
    stride_kv8_b,
    stride_kv32_p,
    stride_kv32_b,
    stride_r2t_b,
    stride_r2t_t,
    stride_ppt_b,
    stride_ppt_n,
    stride_pkp8_p,
    stride_pkp8_b,
    stride_pkp32_p,
    stride_pkp32_b,
    max_kv_blocks,
    max_pool_pages,
    max_ctx,
    num_pool_phys,  # pool_k_pages.shape[0] — for output phys bound
    PAGED_BLOCK_SIZE: tl.constexpr,  # 64
    POOLING_BLOCK_SIZE: tl.constexpr,  # K = k_block_size
    POOL_PAGE_SIZE: tl.constexpr,  # 64
    DIM: tl.constexpr,  # 128
    GROUP_SIZE: tl.constexpr,  # 64/K (K<64) or 1 (K>=64)
    GEMM_TILE: tl.constexpr,  # GROUP_SIZE * POOLING_BLOCK_SIZE
):
    K = POOLING_BLOCK_SIZE
    SCALE_OFFSET_IN: tl.constexpr = PAGED_BLOCK_SIZE * DIM // 4
    SCALE_OFFSET_OUT: tl.constexpr = POOL_PAGE_SIZE * DIM // 4

    b = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    prev_len = tl.load(PrevSeqLens_ptr + b)
    new_len = tl.load(NewSeqLens_ptr + b)
    prev_complete = prev_len // K
    new_complete = new_len // K
    n_new = new_complete - prev_complete

    pblk_rel_start = chunk_idx * GROUP_SIZE
    if pblk_rel_start >= n_new:
        return  # whole CTA does nothing

    req_idx = tl.load(ReqPoolIndices_ptr + b).to(tl.int32)

    # Per-group output destination
    g_offs = tl.arange(0, GROUP_SIZE)
    pblk_abs_per_g = prev_complete + pblk_rel_start + g_offs  # [G]
    valid_per_g = (pblk_rel_start + g_offs) < n_new  # [G]
    logical_pp_per_g = pblk_abs_per_g // POOL_PAGE_SIZE  # [G]
    slot_per_g = pblk_abs_per_g - logical_pp_per_g * POOL_PAGE_SIZE  # [G]
    pp_valid = (
        valid_per_g & (logical_pp_per_g >= 0) & (logical_pp_per_g < max_pool_pages)
    )
    phys_out_per_g = tl.load(
        PoolPageTables_ptr + req_idx * stride_ppt_b + logical_pp_per_g * stride_ppt_n,
        mask=pp_valid,
        other=0,
    ).to(
        tl.int32
    )  # [G]
    # Defensive: clamp output phys to [0, num_pool_phys) — if PoolPageTables
    # holds a stale/sentinel value the masked-store below could otherwise OOB.
    out_valid_per_g = (
        pp_valid & (phys_out_per_g >= 0) & (phys_out_per_g < num_pool_phys)
    )
    phys_out_per_g = tl.where(out_valid_per_g, phys_out_per_g, 0)

    # Source: GEMM_TILE consecutive logical tokens starting at base_token
    base_token = (prev_complete + pblk_rel_start) * K
    bn_offs = tl.arange(0, GEMM_TILE)
    token_logical = base_token + bn_offs  # [GEMM_TILE]
    in_ctx = (
        (token_logical >= 0) & (token_logical < new_len) & (token_logical < max_ctx)
    )
    safe_token = tl.where(in_ctx, token_logical, 0)

    # Per-row phys page lookup via ReqToToken (safe pointer, mask later).
    # Gate on `buf_pos >= 0` BEFORE the divmod: with Triton's C-style trunc
    # integer division, a negative buf_pos (e.g. sentinel -1) would compute
    # phys_per_row = 0 (truncated toward zero) and row_within_page = -1,
    # passing the (phys_per_row >= 0) check yet producing a negative byte
    # offset on the K-cache load below — a real OOB. Caught by
    # test_oob_sanitizer.py poison-phase.
    buf_pos = tl.load(
        ReqToToken_ptr + req_idx * stride_r2t_b + safe_token * stride_r2t_t,
    )
    pos_nonneg = buf_pos >= 0
    safe_buf_pos = tl.where(pos_nonneg, buf_pos, 0)
    phys_per_row = (safe_buf_pos // PAGED_BLOCK_SIZE).to(tl.int32)
    row_within_page = (safe_buf_pos - phys_per_row * PAGED_BLOCK_SIZE).to(tl.int32)
    src_valid = in_ctx & pos_nonneg & (phys_per_row < max_kv_blocks)
    safe_phys = tl.where(src_valid, phys_per_row, 0)
    safe_row = tl.where(src_valid, row_within_page, 0)

    # Load K tile [GEMM_TILE, D] fp8 + per-token scale [GEMM_TILE] f32.
    # No mask on the fp8 load (triton can't cast int→fp8 default); instead
    # we rely on safe_phys/safe_row pointing to a valid (clamped) location
    # and zero out the invalid lanes after fp32 cast.
    d_offs = tl.arange(0, DIM)
    k_byte_offs = (
        safe_phys[:, None] * stride_kv8_p
        + (safe_row[:, None] * DIM + d_offs[None, :]) * stride_kv8_b
    )
    k = tl.load(KvCacheFp8_ptr + k_byte_offs)
    ks_offs = safe_phys * stride_kv32_p + (SCALE_OFFSET_IN + safe_row) * stride_kv32_b
    k_scale = tl.load(KvCacheFp32_ptr + ks_offs)

    k_f32 = k.to(tl.float32) * k_scale[:, None]
    k_f32 = tl.where(src_valid[:, None], k_f32, 0.0)

    # Reshape [GEMM_TILE, D] → [G, K, D]; reduce per pool block
    k_reshaped = tl.reshape(k_f32, (GROUP_SIZE, POOLING_BLOCK_SIZE, DIM))
    sum_per_block = tl.sum(k_reshaped, axis=1)  # [G, D]
    inv_K: tl.constexpr = 1.0 / POOLING_BLOCK_SIZE
    mean_per_block = sum_per_block * inv_K  # [G, D]

    # Per-block fp8 max-abs quantization
    max_abs = tl.max(tl.abs(mean_per_block), axis=1)  # [G]
    block_scale = tl.maximum(max_abs * (1.0 / 448.0), 1e-10)
    out_fp8 = (mean_per_block * (1.0 / block_scale)[:, None]).to(tl.float8e4nv)

    # Paged store of G fp8 rows
    out_addr = (
        phys_out_per_g[:, None] * stride_pkp8_p
        + (slot_per_g[:, None] * DIM + d_offs[None, :]) * stride_pkp8_b
    )
    tl.store(PoolKPagesFp8_ptr + out_addr, out_fp8, mask=out_valid_per_g[:, None])

    # Paged store of G f32 scales
    scale_addr = (
        phys_out_per_g * stride_pkp32_p
        + (SCALE_OFFSET_OUT + slot_per_g) * stride_pkp32_b
    )
    tl.store(PoolKPagesFp32_ptr + scale_addr, block_scale, mask=out_valid_per_g)


# ---------------------------------------------------------------------------
# Kernel 8: tail-only mean-pool refresh into pool_k_pages (v3 paged target)
#   Mirrors: fp8_native_paged_mean_pooling_tail_only (tilelang)
#
#   Per request, refresh ONLY the last (tail) pool block:
#     tail_pblk    = ceildiv(seq_len, K) - 1
#     k_start      = tail_pblk * K
#     cur_size     = min(k_start + K, seq_len) - k_start  (may be < K)
#     mean over cur_size tokens (NOT K) of K, fp8-quantize, write to
#     pool_k_pages[phys_tail, slot_tail].
#
#   Grid: (batch,). Tile = POOLING_BLOCK_SIZE rows; per-row gather of
#   phys_page from BlockTables[b] handles cross-page tail (e.g. K=128
#   spans 2 paged pages). Work per CTA is small — tail kernel is ~10 μs
#   total in production, not on the perf hot path.
# ---------------------------------------------------------------------------


@triton.jit
def _tail_only_kernel(
    KvCacheFp8_ptr,  # [num_phys, paged*(D+4)] fp8 view
    KvCacheFp32_ptr,  # f32 view of same
    BlockTables_ptr,  # [B, max_blocks] i32 — per-batch logical→phys (main KV)
    ContextLens_ptr,  # [B] i32
    PoolPageTables_ptr,  # [B, max_pool_pages] i32 — per-batch pool phys page
    PoolKPagesFp8_ptr,  # [num_pool_pages, page_bytes] fp8 view  IN-OUT
    PoolKPagesFp32_ptr,  # f32 view of same                       IN-OUT
    stride_kv8_p,
    stride_kv8_b,
    stride_kv32_p,
    stride_kv32_b,
    stride_bt_b,
    stride_bt_n,
    stride_ppt_b,
    stride_ppt_n,
    stride_pkp8_p,
    stride_pkp8_b,
    stride_pkp32_p,
    stride_pkp32_b,
    max_blocks_per_req,  # block_tables.shape[1] — for logical_pp bounds
    num_phys,  # kv_cache.shape[0] — for phys bounds (defensive)
    max_pool_pages,
    num_pool_phys,  # pool_k_pages.shape[0] — for output phys bound
    PAGED_BLOCK_SIZE: tl.constexpr,  # 64
    POOLING_BLOCK_SIZE: tl.constexpr,  # K
    POOL_PAGE_SIZE: tl.constexpr,  # 64
    DIM: tl.constexpr,  # 128
):
    b = tl.program_id(0)
    K = POOLING_BLOCK_SIZE
    SCALE_OFFSET_IN: tl.constexpr = PAGED_BLOCK_SIZE * DIM // 4
    SCALE_OFFSET_OUT: tl.constexpr = POOL_PAGE_SIZE * DIM // 4

    seq_len = tl.load(ContextLens_ptr + b)
    if seq_len <= 0:
        return
    num_pool = (seq_len + K - 1) // K
    tail_pblk = num_pool - 1
    k_start = tail_pblk * K
    k_end = tl.minimum(k_start + K, seq_len)
    cur_size = k_end - k_start
    if cur_size <= 0:
        return

    # Per-row token positions in the tail block.
    bn_offs = tl.arange(0, POOLING_BLOCK_SIZE)
    token_logical = k_start + bn_offs  # [K]
    in_block = (token_logical < k_end) & (token_logical >= k_start)

    # Per-row phys page lookup via BlockTables[b, ...].
    logical_pp_per_row = token_logical // PAGED_BLOCK_SIZE
    row_within_page = token_logical - logical_pp_per_row * PAGED_BLOCK_SIZE
    src_valid = (
        in_block & (logical_pp_per_row >= 0) & (logical_pp_per_row < max_blocks_per_req)
    )
    safe_logical = tl.where(src_valid, logical_pp_per_row, 0)

    phys_per_row = tl.load(
        BlockTables_ptr + b * stride_bt_b + safe_logical * stride_bt_n,
    ).to(tl.int32)
    # Defensive: clamp phys to [0, num_phys) — if BlockTables has any
    # spurious value (stale data, allocator race) this prevents OOB on the
    # K-cache load below. SK15 does this; SK16 was missing it.
    src_valid = src_valid & (phys_per_row >= 0) & (phys_per_row < num_phys)
    phys_safe = tl.where(src_valid, phys_per_row, 0)
    row_safe = tl.where(src_valid, row_within_page, 0)

    # K tile + per-token scale (safe pointers; mask later via fp32 cast).
    d_offs = tl.arange(0, DIM)
    k_byte_offs = (
        phys_safe[:, None] * stride_kv8_p
        + (row_safe[:, None] * DIM + d_offs[None, :]) * stride_kv8_b
    )
    k = tl.load(KvCacheFp8_ptr + k_byte_offs)
    ks_offs = phys_safe * stride_kv32_p + (SCALE_OFFSET_IN + row_safe) * stride_kv32_b
    k_scale = tl.load(KvCacheFp32_ptr + ks_offs)

    k_f32 = k.to(tl.float32) * k_scale[:, None]
    k_f32 = tl.where(src_valid[:, None], k_f32, 0.0)

    # Sum and divide by ACTUAL cur_size (not K — tail can be partial).
    sum_v = tl.sum(k_f32, axis=0)  # [D]
    inv_cur = 1.0 / cur_size.to(tl.float32)
    mean = sum_v * inv_cur  # [D]

    max_abs = tl.max(tl.abs(mean))
    block_scale = tl.maximum(max_abs * (1.0 / 448.0), 1e-10)
    out_fp8 = (mean * (1.0 / block_scale)).to(tl.float8e4nv)

    # Output destination = pool_k_pages[phys_tail, slot_tail, :]
    logical_pp_out = tail_pblk // POOL_PAGE_SIZE
    slot_out = tail_pblk - logical_pp_out * POOL_PAGE_SIZE
    pp_valid = (logical_pp_out >= 0) & (logical_pp_out < max_pool_pages)
    if not pp_valid:
        return
    phys_out = tl.load(
        PoolPageTables_ptr + b * stride_ppt_b + logical_pp_out * stride_ppt_n,
    ).to(tl.int32)
    # Defensive: skip the store entirely if phys_out is OOB. Avoids any
    # corrupt write into pool_k_pages when PoolPageTables holds a stale
    # sentinel (e.g. -1 / large positive).
    if (phys_out < 0) | (phys_out >= num_pool_phys):
        return

    out_fp8_offs = phys_out * stride_pkp8_p + (slot_out * DIM + d_offs) * stride_pkp8_b
    tl.store(PoolKPagesFp8_ptr + out_fp8_offs, out_fp8)

    out_scale_off = (
        phys_out * stride_pkp32_p + (SCALE_OFFSET_OUT + slot_out) * stride_pkp32_b
    )
    tl.store(PoolKPagesFp32_ptr + out_scale_off, block_scale)


def tail_only_triton(
    kv_cache_flat: torch.Tensor,  # [num_phys, paged*(D+4)] uint8
    context_lens: torch.Tensor,  # [B] int32
    block_tables: torch.Tensor,  # [B, max_blocks] int32
    pool_page_tables: torch.Tensor,  # [B, max_pool_pages] int32
    pool_k_pages: torch.Tensor,  # [num_pool_pages, page_bytes] uint8 IN-OUT
    k_block_size: int,
    paged_block_size: int,
    pool_page_size: int,
) -> None:
    """Triton equivalent of ``fp8_native_paged_mean_pooling_tail_only_interface``."""
    assert k_block_size in (8, 16, 32, 64, 128)
    num_phys, page_bytes = kv_cache_flat.shape
    DPlus4 = page_bytes // paged_block_size
    D = DPlus4 - 4
    assert pool_k_pages.shape[1] == pool_page_size * (D + 4)
    B, max_blocks = block_tables.shape
    max_pool_pages = pool_page_tables.shape[1]

    kv_fp8 = kv_cache_flat.view(torch.float8_e4m3fn)
    kv_f32 = kv_cache_flat.view(torch.float32)
    pkp_fp8 = pool_k_pages.view(torch.float8_e4m3fn)
    pkp_f32 = pool_k_pages.view(torch.float32)
    num_pool_phys = pool_k_pages.shape[0]

    grid = (B,)
    _tail_only_kernel[grid](
        kv_fp8,
        kv_f32,
        block_tables,
        context_lens,
        pool_page_tables,
        pkp_fp8,
        pkp_f32,
        kv_fp8.stride(0),
        kv_fp8.stride(1),
        kv_f32.stride(0),
        kv_f32.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        pool_page_tables.stride(0),
        pool_page_tables.stride(1),
        pkp_fp8.stride(0),
        pkp_fp8.stride(1),
        pkp_f32.stride(0),
        pkp_f32.stride(1),
        max_blocks,  # max_blocks_per_req — for logical_pp bound
        num_phys,  # num_phys           — for phys bound (defensive)
        max_pool_pages,
        num_pool_phys,  # for output phys bound (defensive)
        PAGED_BLOCK_SIZE=paged_block_size,
        POOLING_BLOCK_SIZE=k_block_size,
        POOL_PAGE_SIZE=pool_page_size,
        DIM=D,
    )


def update_pool_for_completed_blocks_triton(
    kv_cache_flat: torch.Tensor,  # [num_phys, paged*(D+4)] uint8
    req_to_token: torch.Tensor,  # [max_req, max_ctx] int32
    pool_page_tables: torch.Tensor,  # [max_req, max_pool_pages] int32
    req_pool_indices: torch.Tensor,  # [B] int64
    prev_seq_lens: torch.Tensor,  # [B] int32
    new_seq_lens: torch.Tensor,  # [B] int32
    pool_k_pages: torch.Tensor,  # [num_pool_pages, page_bytes] uint8 IN-OUT
    k_block_size: int,
    paged_block_size: int,
    pool_page_size: int,
    max_pool_per_req_grid: int,
) -> None:
    """Triton equivalent of ``fp8_native_paged_mean_pooling_completed_blocks_interface``.

    Supports k_block_size in {8, 16, 32, 64, 128}. Layout assumptions:
    same SoA byte layout as production (fp8 then scales per page).
    """
    assert k_block_size in (8, 16, 32, 64, 128)
    num_phys, page_bytes = kv_cache_flat.shape
    DPlus4 = page_bytes // paged_block_size
    D = DPlus4 - 4
    assert pool_k_pages.shape[1] == pool_page_size * (D + 4)
    max_req, max_ctx = req_to_token.shape
    max_pool_pages = pool_page_tables.shape[1]
    B = req_pool_indices.shape[0]

    GROUP_SIZE = (
        paged_block_size // k_block_size if k_block_size < paged_block_size else 1
    )
    GEMM_TILE = GROUP_SIZE * k_block_size
    assert (
        GROUP_SIZE * k_block_size <= max(paged_block_size, k_block_size) + 1
    ), "internal: GEMM_TILE invariant violated"
    num_chunks = (max_pool_per_req_grid + GROUP_SIZE - 1) // GROUP_SIZE

    kv_fp8 = kv_cache_flat.view(torch.float8_e4m3fn)
    kv_f32 = kv_cache_flat.view(torch.float32)
    pkp_fp8 = pool_k_pages.view(torch.float8_e4m3fn)
    pkp_f32 = pool_k_pages.view(torch.float32)
    num_pool_phys = pool_k_pages.shape[0]

    grid = (B, num_chunks)
    _update_pool_for_completed_blocks_kernel[grid](
        kv_fp8,
        kv_f32,
        req_to_token,
        pool_page_tables,
        req_pool_indices,
        prev_seq_lens,
        new_seq_lens,
        pkp_fp8,
        pkp_f32,
        kv_fp8.stride(0),
        kv_fp8.stride(1),
        kv_f32.stride(0),
        kv_f32.stride(1),
        req_to_token.stride(0),
        req_to_token.stride(1),
        pool_page_tables.stride(0),
        pool_page_tables.stride(1),
        pkp_fp8.stride(0),
        pkp_fp8.stride(1),
        pkp_f32.stride(0),
        pkp_f32.stride(1),
        num_phys,
        max_pool_pages,
        max_ctx,
        num_pool_phys,
        PAGED_BLOCK_SIZE=paged_block_size,
        POOLING_BLOCK_SIZE=k_block_size,
        POOL_PAGE_SIZE=pool_page_size,
        DIM=D,
        GROUP_SIZE=GROUP_SIZE,
        GEMM_TILE=GEMM_TILE,
    )


@triton.jit
def _force_maintain_logits_kernel(
    LOGITS_PTR,
    CU_KS_PTR,
    CU_KE_PTR,
    stride_row,
):
    row = tl.program_id(0)
    ks = tl.load(CU_KS_PTR + row)
    ke = tl.load(CU_KE_PTR + row)
    if ks < ke:
        base = LOGITS_PTR + row * stride_row
        pos_inf = float("inf")
        tl.store(base + ks, pos_inf)
        tl.store(base + (ke - 1), pos_inf)


def force_maintain_logits_triton(
    logits: torch.Tensor,  # [seq, n_blocks] f32, stride_row may exceed n_blocks
    cu_seqlen_blocked_ks: torch.Tensor,  # [seq] i32
    cu_seqlen_blocked_ke: torch.Tensor,  # [seq] i32
) -> torch.Tensor:
    """In-place +inf write at ks and ke-1 per row. Stride-aware so it can be
    applied directly to a sliced view of DeepGEMM's SM-aligned output (where
    ``stride(0) > shape(1)``), avoiding a ``.contiguous()`` copy.
    """
    assert logits.dtype == torch.float32
    assert logits.dim() == 2 and logits.stride(1) == 1
    seq = logits.shape[0]
    if seq == 0:
        return logits
    _force_maintain_logits_kernel[(seq,)](
        logits,
        cu_seqlen_blocked_ks,
        cu_seqlen_blocked_ke,
        logits.stride(0),
    )
    return logits


@triton.jit
def _clean_and_force_maintain_logits_decode_kernel(
    LOGITS_PTR,
    CU_KE_PTR,
    stride_row,
    L,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    ke = tl.load(CU_KE_PTR + row)
    base = LOGITS_PTR + row * stride_row

    block_start = blk * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    in_range = offsets < L
    # Clean: positions >= ke (and within row) become -inf. Replaces what
    # deep_gemm.fp8_paged_mqa_logits used to do via ``clean_logits=True``,
    # which on SM90 routes to smxx_clean_logits and asserts 1D context_lens
    # (incompatible with the 2D requirement at attention.hpp:352).
    invalid = in_range & (offsets >= ke)
    tl.store(base + offsets, float("-inf"), mask=invalid)

    # Force-maintain +inf sentinels at pos 0 and pos ke-1. Writes happen
    # after the clean above (program-local ordering), and only one block
    # touches each position so no inter-block race.
    pos_inf = float("inf")
    if blk == 0 and ke > 0:
        tl.store(base, pos_inf)
    if ke > 0:
        ke_last = ke - 1
        if (ke_last >= block_start) and (ke_last < block_start + BLOCK):
            tl.store(base + ke_last, pos_inf)


def clean_and_force_maintain_logits_decode_triton(
    logits: torch.Tensor,  # [B, max_seq_len] f32 (or [B, 1, max_seq_len])
    num_pool_blocks_per_req: torch.Tensor,  # [B] i32
) -> torch.Tensor:
    """Decode-specific clean + force_maintain in a single kernel:
    * positions ``>= num_pool_blocks_per_req[b]`` per row -> ``-inf``
      (mask out logits past the per-row valid range; replaces DG's
      ``clean_logits=True`` which is incompatible with 2D context_lens
      on SM90)
    * pos 0 and pos ``num_pool_blocks_per_req[b] - 1`` per row -> ``+inf``
      (sentinel for downstream radix select). ``ks`` is implicitly 0 —
      for paged decode, every request's pool blocks start at logical
      index 0 in its own pool_page_table.
    """
    assert logits.dtype == torch.float32
    if logits.dim() == 3:
        B, S, L = logits.shape
        logits_2d = logits.view(B * S, L)
    else:
        logits_2d = logits
    assert logits_2d.dim() == 2 and logits_2d.stride(1) == 1
    seq, L = logits_2d.shape
    if seq == 0:
        return logits
    BLOCK = 256
    grid = (seq, triton.cdiv(L, BLOCK))
    _clean_and_force_maintain_logits_decode_kernel[grid](
        logits_2d,
        num_pool_blocks_per_req,
        logits_2d.stride(0),
        L,
        BLOCK=BLOCK,
    )
    return logits


# =============================================================================
# Coord-transform: post-fast_topk_v2 fused index conversion
# =============================================================================


@triton.jit
def hisa_coord_transform_kernel(
    relevant_ptr,  # [M, INDEX_TOPK] int32 — fast_topk_v2 output
    topk_block_ptr,  # [M, BLOCK_TOPK] int32 — abs block IDs
    ks_ptr,  # [M] int32  — per-query K start (RAGGED); unused on PAGED
    lens_ptr,  # [M] int32  — per-query ke (RAGGED) or seq_len (PAGED)
    out_ptr,  # [M, INDEX_TOPK] int32
    K_BLOCK_SIZE: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    RAGGED: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.arange(0, INDEX_TOPK)

    # Load this row's fast_topk_v2 output.
    r = tl.load(relevant_ptr + m * INDEX_TOPK + offs)
    r_is_valid = r != -1
    r_safe = tl.maximum(r, 0)

    # Gather: abs_block[i] = topk_block_indices[m, r_safe[i] // k_block_size].
    # slot is clamped to [0, BLOCK_TOPK-1] to keep the address computation
    # in-bounds even on disabled lanes — predicated `tl.load` masks the
    # *result* but the address is still computed for every lane, and an
    # OOB address can fault. r ∈ [0, BLOCK_TOPK*K_BLOCK_SIZE) by contract,
    # so a valid lane already lands in range; the upper clamp only matters
    # for r==-1 lanes (slot would be 0 after r_safe → 0) and any stale-r
    # corruption from upstream.
    slot = tl.minimum(r_safe // K_BLOCK_SIZE, BLOCK_TOPK - 1)
    abs_block = tl.load(
        topk_block_ptr + m * BLOCK_TOPK + slot, mask=r_is_valid, other=0
    )

    # raw = abs_block * k_block_size + (r_safe % k_block_size).
    # Use r_safe - slot * K_BLOCK_SIZE to avoid a second modulo.
    raw = abs_block * K_BLOCK_SIZE + (r_safe - slot * K_BLOCK_SIZE)

    if RAGGED:
        ks = tl.load(ks_ptr + m)
        ke = tl.load(lens_ptr + m)
        raw_rel = raw - ks
        valid = (raw_rel >= 0) & (raw_rel < (ke - ks))
    else:
        seq_len = tl.load(lens_ptr + m)
        raw_rel = raw
        valid = raw < seq_len

    out = tl.where(valid & r_is_valid, raw_rel, -1)
    tl.store(out_ptr + m * INDEX_TOPK + offs, out)


def hisa_coord_transform(
    relevant: torch.Tensor,
    topk_block_indices: torch.Tensor,
    lens: torch.Tensor,
    k_block_size: int,
    ks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused coord transform. ``ks is None`` selects PAGED semantics.

    Args:
        relevant: [M, index_topk] int32 — fast_topk_v2 output (positions in
                  the sparse score array, -1 for invalid).
        topk_block_indices: [M, block_topk] int — absolute block IDs.
        lens: [M] int32 — ``ke`` for RAGGED, ``seq_lens`` for PAGED.
        k_block_size: int — hisa block size (e.g. 128).
        ks: [M] int32 or None — RAGGED per-query K start. None → PAGED decode
            (no ks-subtract; mask ``raw < seq_len``).
        out: optional pre-allocated output buffer.

    Returns:
        [M, index_topk] int32. RAGGED: ks-relative positions; PAGED: absolute
        per-request K positions. ``-1`` for invalid / out-of-range entries.
    """
    assert (
        relevant.ndim == 2 and relevant.dtype == torch.int32
    ), f"relevant must be [M, index_topk] int32, got {relevant.shape} {relevant.dtype}"
    M, index_topk = relevant.shape
    block_topk = topk_block_indices.shape[-1]

    # Kernel assumes row_stride == row_width (no stride args passed); force
    # contiguity to be safe. Production inputs (fast_topk_v2 output, hisa
    # kernel output, 1-D slices of ks/ke) are already contiguous, so these
    # calls are typically no-ops.
    relevant = relevant.contiguous()
    topk_block_indices = topk_block_indices.contiguous()

    # Normalize dtypes — the triton kernel assumes int32 throughout.
    if topk_block_indices.dtype != torch.int32:
        topk_block_indices = topk_block_indices.to(torch.int32)
    if lens.dtype != torch.int32:
        lens = lens.to(torch.int32)
    lens = lens.contiguous()

    ragged = ks is not None
    if ragged:
        if ks.dtype != torch.int32:
            ks = ks.to(torch.int32)
        ks = ks.contiguous()
    # In PAGED mode the kernel ignores ks_ptr; pass any valid pointer.
    ks_arg = ks if ragged else lens

    if out is None:
        out = torch.empty((M, index_topk), dtype=torch.int32, device=relevant.device)
    else:
        assert out.shape == (M, index_topk) and out.dtype == torch.int32

    hisa_coord_transform_kernel[(M,)](
        relevant,
        topk_block_indices,
        ks_arg,
        lens,
        out,
        K_BLOCK_SIZE=k_block_size,
        INDEX_TOPK=index_topk,
        BLOCK_TOPK=block_topk,
        RAGGED=ragged,
    )
    return out
