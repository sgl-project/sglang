from typing import Optional

import torch

def quest_retrieval_score_and_combine_indices(
    bs: int,
    seq_lens: torch.Tensor,
    page_size: int,
    req_to_token: torch.Tensor,
    page_k_min: torch.Tensor,
    page_k_max: torch.Tensor,
    queries: torch.Tensor,
    req_pool_indices: torch.Tensor,
    num_recent_pages: int,
    fixed_topk_page_cnt: Optional[int],
    sparsity_ratio: float,
    sparse_mask: torch.Tensor,
    out_indices: torch.Tensor,
    out_lengths: torch.Tensor,
) -> None:
    """
    Call the optimized CUDA kernel for Quest retrieval score calculation and index combination.

    Args:
        bs: Batch size
        seq_lens: Sequence lengths [bs]
        page_size: Page size
        req_to_token: Request to token mapping [req_pool_size, max_tokens]
        page_k_min: Page key min values [total_pages, kv_heads, head_dim]
        page_k_max: Page key max values [total_pages, kv_heads, head_dim]
        queries: Queries [bs, q_heads, head_dim]
        req_pool_indices: Request pool indices [bs]
        num_recent_pages: Number of recent pages to always include
        fixed_topk_page_cnt: Fixed number of top-k pages (optional)
        sparsity_ratio: Sparsity ratio for dynamic top-k
        sparse_mask: Sparse mask [bs] (optional, pass empty tensor if not used)
        out_indices: Output indices tensor [bs, max_out] (int32)
        out_lengths: Output lengths tensor [bs] (int32)
    """

    return torch.ops.sgl_kernel.quest_retrieval_score_and_combine_indices.default(
        bs,
        seq_lens,
        page_size,
        req_to_token,
        page_k_min,
        page_k_max,
        queries,
        req_pool_indices,
        num_recent_pages,
        fixed_topk_page_cnt,
        sparsity_ratio,
        sparse_mask,
        out_indices,
        out_lengths,
    )


def update_sparse_metadata(
    page_table: torch.Tensor,
    physical_pages: torch.Tensor,
    valid_lengths: torch.Tensor,
    sparse_mask: torch.Tensor,
    cache_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    original_cache_seqlens: torch.Tensor,
    page_size: int,
) -> None:
    """
    Call the optimized CUDA kernel for Quest sparse metadata update.

    Args:
        page_table: Page table [bs, max_selected, max_tokens]
        physical_pages: Physical pages [bs, max_selected, max_tokens]
        valid_lengths: Valid lengths [bs]
        sparse_mask: Sparse mask [bs] (optional, pass empty tensor if not used)
        cache_seqlens: Cache sequence lengths [bs]
        seq_lens: Sequence lengths [bs]
        original_cache_seqlens: Original cache sequence lengths [bs]
        page_size: Page size
    """

    return torch.ops.sgl_kernel.update_sparse_metadata.default(
        page_table,
        physical_pages,
        valid_lengths,
        sparse_mask,
        cache_seqlens,
        seq_lens,
        original_cache_seqlens,
        page_size,
    )
