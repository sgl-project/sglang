"""HISA legacy orchestrators — v1-style, slated for removal.

Only ``fp8_native_hierarchy_paged_mqa_logits_no_pool_cache`` lives here for
now. It is reachable from :class:`HisaIndexer` only when the user sets
``SGLANG_HISA_DISABLE_POOL_CACHE=1`` *and* ``k_block_size < 64`` (the
tilelang legacy orchestrator can't handle K<64, so we fall back to this
all-triton no-cache path).

Once the env-var fallback is dropped, this file can be deleted alongside
``tilelang_legacy.py``.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.orchestrator import _stage3_topk_decode
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    batch_pool_mqa_triton,
    paged_mean_pooling_triton,
    sparse_paged_mqa_triton,
)


def fp8_native_hierarchy_paged_mqa_logits_no_pool_cache(
    q_fp8: torch.Tensor,  # [B, 1, H, D] fp8
    kv_cache_fp8: torch.Tensor,  # [num_blocks, paged_block_size, 1, D+4] uint8
    weights: torch.Tensor,  # [B*1, H] f32
    context_lens: torch.Tensor,  # [B] i32 — raw seq_len per request
    block_tables: torch.Tensor,  # [B, max_kv_blocks] i32
    k_block_size: int,
    block_topk: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-triton fp8 hierarchy MQA, no pool cache. Mirrors
    ``fp8_native_hierarchy_paged_mqa_logits_tilelang_legacy`` (tilelang) but every kernel
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
    topk_block_indices = _stage3_topk_decode(block_k_indexer_score, block_topk)

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
