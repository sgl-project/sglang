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
    sparse_paged_mqa_triton,
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
    # 1) Refresh tail pool block in place (tilelang kernel — cheap).
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
