"""Convert a decode cross-attention mask to a paged KV read selection."""

import torch


def filter_cross_attention_kv_indices(
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    custom_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select visible keys for one decode query per request.

    ``custom_mask`` concatenates one encoder-length boolean row per request.
    ``kv_indices`` may include extra CUDA graph capacity after the packed rows.
    The returned indices preserve key order, and the returned indptr includes
    requests with zero encoder tokens or zero visible keys.

    Run this while preparing metadata outside CUDA graph capture/replay.
    """
    visible = custom_mask.to(device=kv_indices.device, dtype=torch.bool)
    visible_prefix = torch.cat(
        (
            kv_indptr.new_zeros(1),
            visible.cumsum(dim=0, dtype=kv_indptr.dtype),
        )
    )
    return kv_indices[: visible.numel()][visible], visible_prefix[kv_indptr]
