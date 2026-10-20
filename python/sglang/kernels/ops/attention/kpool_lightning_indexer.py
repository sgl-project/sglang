"""Fused Triton kernel for DSA KPool topk selection on NPU.

Replaces ``npu_lightning_indexer`` with a pure-Triton implementation that:

1. Reads pooled-K cache in **paged** layout (page_size=128), iterating over
   pages to gather keys.
2. Computes MQA logits: ``logit = sum_h w[h] * relu(q[h,:] @ k[j,:])``
3. Applies **per-token causal masking**: query row *i* may only attend to
   pool entries ``[0, pool_seqlens[i])``.
4. Selects **top-k pools** per query row via ``torch.topk``.

Both prefill (extend) and decode use the same paged read path; the only
difference is the shape of ``pool_seqlens`` (per-token for extend,
per-request for decode).

Optimization highlights vs the original implementation:
- **Page-at-a-time access**: each loop iteration loads ONE ``page_id``
  (scalar) and reads the entire page of keys as a contiguous block,
  eliminating the per-entry gather pattern (``page_ids[:, None]``) that
  caused scattered global-memory reads.
- **``tl.range`` pipelining**: the main loop uses ``tl.range`` to enable
  the compiler's software-pipelining pass.
- **Two-phase selection**: the Triton logits kernel is followed by
  ``torch.topk`` over the causally masked logits.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ── Optimized logits kernel: page-at-a-time access ──────────────────────


@triton.jit
def _paged_mqa_logits_kernel(
    q_ptr,
    k_cache_ptr,
    weights_ptr,
    pool_seqlens_ptr,
    block_tables_ptr,
    logits_ptr,
    q_stride_q,
    q_stride_h,
    q_stride_d,
    k_stride_page,
    k_stride_slot,
    k_stride_h,
    k_stride_d,
    w_stride_q,
    w_stride_h,
    bt_stride_q,
    bt_stride_page,
    log_stride_q,
    log_stride_pool,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    max_pool_len: tl.constexpr,
    max_pages: tl.constexpr,
):
    """One program per query row.  Iterates over **pages** (not individual
    pool entries), loading one page_id as a scalar per iteration and reading
    the entire page of keys as a contiguous block."""

    row = tl.program_id(0)

    pool_seq_len = tl.load(pool_seqlens_ptr + row).to(tl.int32)

    # ── Load query: (n_heads, head_dim) ──
    offs_h = tl.arange(0, n_heads)
    offs_d = tl.arange(0, head_dim)
    q = tl.load(
        q_ptr
        + row * q_stride_q
        + offs_h[:, None] * q_stride_h
        + offs_d[None, :] * q_stride_d
    )
    q_f32 = q.to(tl.float32)

    # ── Load weights: (n_heads,) ──
    w = tl.load(weights_ptr + row * w_stride_q + offs_h * w_stride_h)

    # ── Slot offsets within a page (shared across all iterations) ──
    slot_offs = tl.arange(0, page_size)

    # ── Iterate over pages, loading keys contiguously ──
    for page_iter in tl.range(0, max_pages):
        pool_start = page_iter * page_size

        # Load ONE page_id (scalar, not gather)
        page_id = tl.load(
            block_tables_ptr + row * bt_stride_q + page_iter * bt_stride_page
        ).to(tl.int64)

        # Load entire page of keys contiguously: (page_size, head_dim)
        k_ptrs = (
            k_cache_ptr
            + page_id * k_stride_page
            + slot_offs[:, None].to(tl.int64) * k_stride_slot
            + offs_d[None, :] * k_stride_d
        )
        k = tl.load(k_ptrs)
        k_f32 = k.to(tl.float32)

        # MQA logits: (n_heads, page_size)
        scores = tl.dot(q_f32, tl.trans(k_f32))
        scores = tl.maximum(scores, 0.0)

        weighted = scores * w[:, None]
        logits_page = tl.sum(weighted, axis=0)

        # Causal mask
        pool_ids = pool_start + slot_offs
        valid = pool_ids < pool_seq_len
        logits_page = tl.where(valid, logits_page, float("-inf"))

        # Store
        log_ptrs = logits_ptr + row * log_stride_q + pool_ids * log_stride_pool
        tl.store(log_ptrs, logits_page, mask=pool_ids < max_pool_len)


# ── Public API ──────────────────────────────────────────────────────────


def fused_paged_mqa_logits(
    q_bf16: torch.Tensor,
    k_cache_bf16: torch.Tensor,
    weights: torch.Tensor,
    pool_seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    max_pool_len: int,
    page_size: int = 128,
) -> torch.Tensor:
    """Compute paged MQA logits with causal masking via Triton kernel.

    Uses page-at-a-time access: each iteration loads one page_id (scalar)
    and reads the entire page of keys contiguously, eliminating the
    per-entry gather pattern.

    Args:
        q_bf16: (num_q, n_heads, head_dim) BF16
        k_cache_bf16: (num_pages, page_size, 1, head_dim) BF16
        weights: (num_q, n_heads) FP32
        pool_seqlens: (num_q,) int32
        block_tables: (num_q, max_pages) int32
        max_pool_len: max number of pool entries across all rows
        page_size: page size of the index cache

    Returns:
        logits: (num_q, max_pool_len) FP32 with -inf for masked entries
    """
    num_q, n_heads, head_dim = q_bf16.shape
    max_pages = block_tables.shape[1]

    logits = torch.empty(
        (num_q, max_pool_len), dtype=torch.float32, device=q_bf16.device
    )
    logits.fill_(float("-inf"))

    grid = (num_q,)
    _paged_mqa_logits_kernel[grid](
        q_bf16,
        k_cache_bf16,
        weights,
        pool_seqlens,
        block_tables,
        logits,
        q_bf16.stride(0),
        q_bf16.stride(1),
        q_bf16.stride(2),
        k_cache_bf16.stride(0),
        k_cache_bf16.stride(1),
        k_cache_bf16.stride(2),
        k_cache_bf16.stride(3),
        weights.stride(0),
        weights.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        n_heads=n_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_pool_len=max_pool_len,
        max_pages=max_pages,
        num_warps=4,
    )

    return logits


def fused_topk_paged(
    q_bf16: torch.Tensor,
    k_cache_bf16: torch.Tensor,
    weights: torch.Tensor,
    pool_seqlens: torch.Tensor,
    block_tables: torch.Tensor,
    max_pool_len: int,
    n_pool_topk: int,
    page_size: int = 128,
) -> torch.Tensor:
    """Fused paged MQA logits + causal mask + topk selection.

    Computes logits with the Triton paged kernel, then selects pools using
    ``torch.topk``.

    Returns pool-level topk indices: (num_q, n_pool_topk) int32.

    Args:
        q_bf16: (num_q, n_heads, head_dim) BF16
        k_cache_bf16: (num_pages, page_size, 1, head_dim) BF16
        weights: (num_q, n_heads) FP32
        pool_seqlens: (num_q,) int32
        block_tables: (num_q, max_pages) int32
        max_pool_len: max pool entries
        n_pool_topk: number of top pools to select
        page_size: page size of the index cache
    """
    num_q = q_bf16.shape[0]

    topk_indices = torch.full(
        (num_q, n_pool_topk), -1, dtype=torch.int32, device=q_bf16.device
    )

    actual_topk = min(n_pool_topk, max_pool_len)
    if actual_topk == 0:
        return topk_indices

    logits = fused_paged_mqa_logits(
        q_bf16,
        k_cache_bf16,
        weights,
        pool_seqlens,
        block_tables,
        max_pool_len,
        page_size,
    )

    topk_logits, topk_pool_indices = torch.topk(
        logits, actual_topk, dim=1, largest=True
    )
    topk_pool_indices = torch.where(
        topk_logits.isneginf(),
        torch.full_like(topk_pool_indices, -1),
        topk_pool_indices,
    )
    topk_indices[:, :actual_topk] = topk_pool_indices.to(torch.int32)

    return topk_indices
