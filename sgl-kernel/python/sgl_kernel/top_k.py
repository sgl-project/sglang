import torch


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        # TODO: implement faster cuda kernels for large vocab sizes
        return torch.topk(values, topk, dim=dim)


def fast_topk_v2(score: torch.Tensor, lengths: torch.Tensor, topk: int) -> torch.Tensor:
    assert (
        topk == 2048
    ), "fast_topk_v2 is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk(score, topk_indices, lengths)
    return topk_indices


def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,  # NOTE: page size should be 1
    cu_seqlens_q: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """
    Transform topk indices to indices to the page table (page_size = 1)
    """
    assert (
        topk == 2048
    ), "fast_topk_transform_fused is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    src_page_table = page_table_size_1
    dst_page_table = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk_transform_fused(
        score, lengths, dst_page_table, src_page_table, cu_seqlens_q
    )
    return dst_page_table


def fast_topk_transform_ragged_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,  # ragged kv
    topk: int,
) -> torch.Tensor:
    """
    Transform topk indices to indices to ragged kv (non-paged)
    """
    assert (
        topk == 2048
    ), "fast_topk_transform_fused_ragged is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    topk_indices_ragged = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk_transform_ragged_fused(
        score, lengths, topk_indices_ragged, topk_indices_offset
    )
    return topk_indices_ragged
