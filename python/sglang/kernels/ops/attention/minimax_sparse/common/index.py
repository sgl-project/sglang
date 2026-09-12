import torch


def topk_index_reduce(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Reduces a specific dimension by computing the union of all top-k indices along that dimension.
    The resulting tensor will have the 'dim' removed, and the last dimension expanded.

    Example:
        Input:  [10, num_heads, seq_len, max_topk] with dim=0
        Output: [num_heads, seq_len, 10 * max_topk] (Left-aligned, padded with -1)

    Args:
        tensor (torch.Tensor): Input tensor of shape [..., dim_size, ..., max_topk].
        dim (int): The dimension to reduce (collapse).

    Returns:
        torch.Tensor: Reduced tensor with shape [..., new_max_topk].
                      Where new_max_topk = dim_size * original_max_topk.
    """
    # 1. Shape Transformation
    # We need to merge the target 'dim' with the last dimension 'max_topk'.
    # Step A: Move the target dim to the second-to-last position (-2).
    # e.g., [10, H, S, K] (dim=0) -> [H, S, 10, K]
    tensor_permuted = torch.movedim(tensor, source=dim, destination=-2)

    # Step B: Flatten the last two dimensions.
    # e.g., [H, S, 10, K] -> [H, S, 10 * K]
    combined = tensor_permuted.flatten(start_dim=-2)

    # --- The following logic is identical to 'topk_index_union' ---

    # 2. Sort row-wise.
    # Groups identical values together. -1 padding sorts to the left.
    sorted_vals, _ = combined.sort(dim=-1)

    # 3. Deduplication (Delta Check).
    # Keep value if it differs from the previous one.
    is_new_element = sorted_vals[..., 1:] != sorted_vals[..., :-1]

    # First column is always new
    first_col_true = torch.ones_like(sorted_vals[..., :1], dtype=torch.bool)
    non_duplicate_mask = torch.cat([first_col_true, is_new_element], dim=-1)

    # 4. Filter.
    # Valid if non-duplicate AND not -1 padding.
    valid_mask = non_duplicate_mask & (sorted_vals != -1)

    # 5. Packing (Left-Alignment).
    # Move valid elements to the left.
    sort_idx = torch.argsort((~valid_mask).int(), dim=-1, stable=True)
    result = torch.gather(sorted_vals, -1, sort_idx)

    # 6. Re-masking the right side.
    # Fill garbage values on the right with -1.
    valid_count = valid_mask.sum(dim=-1, keepdim=True)
    total_cols = result.size(-1)

    # Broadcasting check:
    # valid_count shape: [..., 1]
    # idx_range shape:   [total_cols]
    # result shape:      [..., total_cols]
    idx_range = torch.arange(total_cols, device=tensor.device)

    final_result = torch.where(idx_range < valid_count, result, -1)

    return final_result
