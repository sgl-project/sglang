import torch
from typing import Optional

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
        out_lengths
    )

def invoke_sparse_diff_cuda_kernel(
    page_table: torch.Tensor,
    last_top_k: torch.Tensor,
    last_page_ids: torch.Tensor,
    curr_top_k: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    valid_lengths: torch.Tensor,
    sparse_mask: torch.Tensor,
    req_to_tokens_host: torch.Tensor,
    physical_pages: torch.Tensor,
    load_tokens: torch.Tensor,
    load_tokens_host: torch.Tensor,
    cache_seqlens: torch.Tensor,
    original_cache_seqlens: torch.Tensor,
    layer_id: int,
    page_size: int,
) -> None:
    """
    Call the optimized CUDA kernel for Quest sparse metadata update with LRU diff.
    
    Args:
        page_table: Page table [bs, pt_stride]
        last_top_k: Last step's top-k pages [num_reqs, num_layers, hot_buffer_len]
        last_page_ids: Last step's page IDs [num_reqs, num_layers, hot_buffer_len]
        curr_top_k: Current top-k pages [bs, top_k]
        req_pool_indices: Request pool indices [bs]
        seq_lens: Sequence lengths [bs]
        valid_lengths: Valid lengths [bs]
        sparse_mask: Sparse mask [bs]
        req_to_tokens_host: Mapping from request to host tokens
        physical_pages: Output tensor for physical page indices [bs, top_k]
        load_tokens: Output tensor for tokens to load to device [bs, top_k * page_size]
        load_tokens_host: Output tensor for host tokens source [bs, top_k * page_size]
        cache_seqlens: Cache sequence lengths [bs]
        original_cache_seqlens: Original cache sequence lengths [bs]
        layer_id: Layer ID
        page_size: Page size
    """
    
    return torch.ops.sgl_kernel.invoke_sparse_diff_cuda_kernel.default(
        page_table,
        last_top_k,
        last_page_ids,
        curr_top_k,
        req_pool_indices,
        seq_lens,
        valid_lengths,
        sparse_mask,
        req_to_tokens_host,
        physical_pages,
        load_tokens,
        load_tokens_host,
        cache_seqlens,
        original_cache_seqlens,
        layer_id,
        page_size
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
    Call the optimized CUDA kernel to update page_table and cache_seqlens.
    """
    return torch.ops.sgl_kernel.update_sparse_metadata.default(
        page_table,
        physical_pages,
        valid_lengths,
        sparse_mask,
        cache_seqlens,
        seq_lens,
        original_cache_seqlens,
        page_size
    )
