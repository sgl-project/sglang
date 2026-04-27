"""v4 orchestrator — paged pool-K cache (v3 layout) + triton hotspot kernels.

Same control flow as ``fp8_native_hierarchy_paged_mqa_logits_with_pool_cache_v3``
in ``hisa/custom_ops.py``:

    1) tail_only_v3   (tilelang, cheap, ~10 μs)
    2) block-MQA      (→ triton batch_decode_pool_mqa_v3_triton,  1-3.5× faster)
    3) torch.topk     (unchanged)
    4) sparse-paged   (→ triton sparse_paged_mqa_triton,  6-15× faster)

Per-step on decode (B=10, ctx=65K): v4 saves ~12 ms / step vs v3 at the
indexer level (steady-state total ~5 ms vs v3's ~17 ms). Correctness: fp8
ULP drift ≤ 2.6% rel, topk-2048 IoU ≥ 0.997 vs tilelang — within fp8
accumulation noise, no e2e regression expected (verify via
``SGLANG_HISA_VERIFY=1`` when flipping the default).
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_paged_mean_pooling_tail_only_v3_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    batch_decode_pool_mqa_v3_triton,
    batch_pool_mqa_triton,
    paged_mean_pooling_triton,
    sparse_paged_mqa_triton,
    tail_only_v3_triton,
)


def fp8_native_hierarchy_paged_mqa_logits_with_pool_cache_v4(
    q_fp8: torch.Tensor,                # [B, 1, H, D] fp8
    kv_cache_fp8: torch.Tensor,         # [num_blocks, paged_block_size, 1, D+4] uint8
    pool_k_pages: torch.Tensor,         # [num_pool_pages_global, pool_page_size * (D+4)] uint8
    pool_page_tables: torch.Tensor,     # [B, max_pool_pages] i32
    weights: torch.Tensor,              # [B*1, H] f32
    context_lens: torch.Tensor,         # [B] i32 — raw seq_len per request
    block_tables: torch.Tensor,         # [B, max_kv_blocks] i32
    k_block_size: int,
    pool_page_size: int,
    block_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1) Refresh tail pool block in place. Dispatch: tilelang for k>=paged
    # (the well-tested production path), SK16 triton for k<paged where
    # tilelang would assert pooling%paged==0.
    paged_block_size = kv_cache_fp8.shape[1]
    if k_block_size < paged_block_size:
        num_phys = kv_cache_fp8.shape[0]
        kv_cache_flat = kv_cache_fp8.view(num_phys, -1)
        tail_only_v3_triton(
            kv_cache_flat=kv_cache_flat,
            context_lens=context_lens,
            block_tables=block_tables,
            pool_page_tables=pool_page_tables,
            pool_k_pages=pool_k_pages,
            k_block_size=k_block_size,
            paged_block_size=paged_block_size,
            pool_page_size=pool_page_size,
        )
    else:
        fp8_native_paged_mean_pooling_tail_only_v3_interface(
            kv_cache=kv_cache_fp8, context_lens=context_lens,
            block_tables=block_tables,
            pool_page_tables=pool_page_tables,
            pool_k_pages=pool_k_pages,
            k_block_size=k_block_size,
            pool_page_size=pool_page_size,
        )

    # 2) Block-MQA — triton port of paged pool_k_pages reader.
    num_pool_blocks_per_req = (context_lens + k_block_size - 1) // k_block_size
    block_k_indexer_score = batch_decode_pool_mqa_v3_triton(
        q_fp8=q_fp8,
        pool_k_pages=pool_k_pages,
        pool_page_tables=pool_page_tables,
        weights_f32=weights,
        context_lens_pool=num_pool_blocks_per_req,
        pool_page_size=pool_page_size,
    )  # [B, 1, max_pool_pages * pool_page_size] f32

    # 3) Top-k over pool blocks — torch native.
    topk_block_indices = torch.topk(
        block_k_indexer_score,
        k=min(block_topk, block_k_indexer_score.shape[-1]),
        dim=-1, sorted=False,
    ).indices

    # 4) Sparse paged MQA — triton port of the decode hotspot.
    block_sparse_k_indexer_score = sparse_paged_mqa_triton(
        q_fp8=q_fp8,
        kv_cache_fp8=kv_cache_fp8,
        topk_block_index=topk_block_indices,
        kv_block_size=k_block_size,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_tables,
    )
    return block_sparse_k_indexer_score, topk_block_indices


def fp8_native_hierarchy_paged_mqa_logits_triton(
    q_fp8: torch.Tensor,                # [B, 1, H, D] fp8
    kv_cache_fp8: torch.Tensor,         # [num_blocks, paged_block_size, 1, D+4] uint8
    weights: torch.Tensor,              # [B*1, H] f32
    context_lens: torch.Tensor,         # [B] i32 — raw seq_len per request
    block_tables: torch.Tensor,         # [B, max_kv_blocks] i32
    k_block_size: int,
    block_topk: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-triton fp8 hierarchy MQA, no pool cache. Mirrors
    ``fp8_native_hierarchy_paged_mqa_logits`` (tilelang) but every kernel
    is the SK1..SK12 triton variant, so it works for k_block_size in
    {8, 16, 32, 64, 128} (tilelang ones break for k<64).

    Used by ``_get_topk_paged`` whenever ``k_block_size < 64``, regardless
    of pool-cache env-var state — tilelang would assert there.

    Flow (matches v1 baseline, no cache):
      1) paged mean-pool — ``paged_mean_pooling_triton`` (SK2)
      2) block-MQA score — ``batch_pool_mqa_triton`` (SK10) on contiguous blocked_k
      3) torch.topk
      4) sparse paged MQA — ``sparse_paged_mqa_triton`` (SK3/SK12)

    Returns ``(block_sparse_logits[B, 1, topk*k_block_size],
    topk_block_indices[B, 1, topk] int64)``.
    """
    B, seq_q, H, D = q_fp8.shape
    assert seq_q == 1, "decode expects q_len=1"
    max_num_pool = (max_seq_len + k_block_size - 1) // k_block_size

    # 1) Fresh paged mean-pool (no cache).
    blocked_k_fp8, blocked_k_scale, num_pool_blocks = paged_mean_pooling_triton(
        max_num_pooling_blocks=max_num_pool,
        kv_cache=kv_cache_fp8,
        context_lens=context_lens,
        block_tables=block_tables,
        k_block_size=k_block_size,
    )  # blocked_k: [B, max_num_pool, D] fp8 ; scale: [B, max_num_pool] f32

    # 2) Block-MQA on contiguous blocked_k.
    block_k_indexer_score = batch_pool_mqa_triton(
        q_fp8=q_fp8,
        blocked_k_fp8=blocked_k_fp8,
        blocked_k_scale=blocked_k_scale,
        weights_f32=weights,
        context_lens=num_pool_blocks,
    )  # [B, 1, max_num_pool] f32

    # 3) Top-k over pool blocks.
    topk_block_indices = torch.topk(
        block_k_indexer_score,
        k=min(block_topk, block_k_indexer_score.shape[-1]),
        dim=-1, sorted=False,
    ).indices  # [B, 1, topk] int64

    # 4) Sparse paged MQA on the chosen K-blocks.
    block_sparse_k_indexer_score = sparse_paged_mqa_triton(
        q_fp8=q_fp8,
        kv_cache_fp8=kv_cache_fp8,
        topk_block_index=topk_block_indices,
        kv_block_size=k_block_size,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_tables,
    )  # [B, 1, topk*k_block_size] f32
    return block_sparse_k_indexer_score, topk_block_indices
