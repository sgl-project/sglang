"""Read pages or compact logical tokens from the NPU MLA cache."""

from collections.abc import Sequence

import torch


def build_mla_prefix_token_indices(
    prefix_lens: Sequence[int], page_size: int, *, device: torch.device
) -> torch.Tensor:
    """Select valid tokens from request-major, whole-page prefix gathers.

    Each request owns ceil(prefix_len / page_size) gathered pages. Its valid
    tokens start at that physical page offset, not at sum(previous lengths).
    Build this once per batch and reuse it for latent and RoPE at every layer.
    """
    indices = []
    page_offset = 0
    for prefix_len in prefix_lens:
        if prefix_len:
            indices.append(
                torch.arange(page_offset, page_offset + prefix_len, dtype=torch.int64)
            )
        page_offset += (prefix_len + page_size - 1) // page_size * page_size
    if not indices:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.cat(indices).to(device=device)


def gather_mla_cache_pages(
    cache: torch.Tensor, block_ids: torch.Tensor, *, is_nz: bool
) -> torch.Tensor:
    """Return selected pages in logical [blocks, page_size, 1, head_dim] order.

    NZ buffers retain that public shape, but their physical contents are
    [blocks, head_dim // 16, page_size, 16]. Restore token-major order before
    projecting cached latent vectors or concatenating their RoPE features.
    """
    pages = torch.index_select(cache, 0, block_ids)
    if not is_nz:
        return pages
    page_size, head_dim = cache.shape[1], cache.shape[-1]
    return (
        pages.view(block_ids.numel(), head_dim // 16, page_size, 16)
        .permute(0, 2, 1, 3)
        .reshape(block_ids.numel(), page_size, 1, head_dim)
    )


def gather_mla_cache_tokens(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    is_nz: bool,
) -> torch.Tensor:
    """Return [sum(prefix_lens), 1, head_dim], excluding every page tail.

    Restore NZ token order before compacting. Truncating the concatenated
    pages would retain padding between requests and shift their boundaries.
    """
    if token_indices.numel() == 0:
        return cache.new_empty((0, 1, cache.shape[-1]))
    pages = gather_mla_cache_pages(cache, block_ids, is_nz=is_nz)
    return torch.index_select(pages.flatten(0, 1), 0, token_indices)
