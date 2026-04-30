from typing import Optional

import torch


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        # TODO: implement faster cuda kernels for large vocab sizes
        return torch.topk(values, topk, dim=dim)


def fast_topk_v2(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key whose layout is either ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert (
        topk == 2048
    ), "fast_topk_v2 is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk(score, topk_indices, lengths, row_starts)
    return topk_indices


def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,  # NOTE: page size should be 1
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor and then transform the topk indices
    to indices to the page table (page_size = 1)
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key whose layout is either ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        page_table_size_1: The page table tensor of shape (Batch, topk)
        cu_seqlens_q: The cumulative sequence lengths tensor of shape (Batch + 1)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor. It's only used for cases where the key is
            ragged, i.e. during extend and draft extend.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert (
        topk == 2048
    ), "fast_topk_transform_fused is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    src_page_table = page_table_size_1
    dst_page_table = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk_transform_fused(
        score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts
    )
    return dst_page_table


def deepseek_v4_topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """
    Performs the DeepSeek-V4 indexer top-k selection (TopK = 512) and writes
    the paged physical slot indices into ``page_indices``. Optionally also
    writes the row-relative raw token positions into ``raw_indices`` for
    hisparse capture.

    Args:
        scores: float32 ``[B, max_seq_len]`` indexer logits, contiguous on dim 1.
        seq_lens: int32 ``[B]``, true KV length per batch row.
        page_table: int32 ``[B, num_pages]``, logical->physical page table,
            contiguous on dim 1.
        page_indices: int32 ``[B, 512]``, output buffer, contiguous. Filled
            with paged physical slots; -1 for padding entries.
        page_size: power-of-two page size (e.g. 256 for V4 decode).
        raw_indices: optional int32 ``[B, 512]``, output buffer, contiguous.
            If provided, filled with raw absolute token positions (or -1 for
            padding entries).
    """
    assert scores.dim() == 2
    assert page_indices.dim() == 2 and page_indices.size(1) == 512
    if raw_indices is not None:
        assert raw_indices.dim() == 2 and raw_indices.size(1) == 512
    torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
        scores, seq_lens, page_table, page_indices, page_size, raw_indices
    )


def fast_topk_transform_ragged_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,  # ragged kv
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor and then transform the topk indices to
    indices to ragged kv (non-paged). This function is only used for extend,
    not including draft extend.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key which can be ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        topk_indices_offset: The offset of topk indices in ragged kv of shape (B)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor. It can be None if only the fast path is triggered,
            in the case of all values in lengths <= topk (not checked in the kernel,
            guaranteed by the caller).
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert (
        topk == 2048
    ), "fast_topk_transform_ragged_fused is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    topk_indices_ragged = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk_transform_ragged_fused(
        score, lengths, topk_indices_ragged, topk_indices_offset, row_starts
    )
    return topk_indices_ragged
