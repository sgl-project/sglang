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
    [blocks, head_dim // D0, page_size, D0], where D0 spans 32 bytes.
    Restore token-major order before
    projecting cached latent vectors or concatenating their RoPE features.
    """
    # Gather backing bytes, because index_select need not implement FP8.
    backing = cache.view(torch.uint8) if cache.dtype == torch.float8_e4m3fn else cache
    pages = torch.index_select(backing, 0, block_ids)
    if not is_nz:
        return pages.view(cache.dtype)
    page_size, head_dim = cache.shape[1], cache.shape[-1]
    tile_size = 32 if cache.dtype == torch.float8_e4m3fn else 16
    return (
        pages.view(block_ids.numel(), head_dim // tile_size, page_size, tile_size)
        .permute(0, 2, 1, 3)
        .reshape(block_ids.numel(), page_size, 1, head_dim)
        .view(cache.dtype)
    )


def gather_mla_cache_tokens(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    token_indices: torch.Tensor | None,
    *,
    is_nz: bool,
) -> torch.Tensor:
    """Return [sum(prefix_lens), 1, head_dim], excluding every page tail.

    Restore NZ token order before compacting. Truncating the concatenated
    pages would retain padding between requests and shift their boundaries.
    Pass None when all request lengths are page-aligned to skip compaction.
    """
    if block_ids.numel() == 0 or (
        token_indices is not None and token_indices.numel() == 0
    ):
        return cache.new_empty((0, 1, cache.shape[-1]))
    pages = gather_mla_cache_pages(cache, block_ids, is_nz=is_nz)
    tokens = pages.flatten(0, 1)
    if token_indices is None:
        return tokens
    # Match page gathering: FP8 copies use bytes until cache dequantization.
    backing = (
        tokens.view(torch.uint8) if tokens.dtype == torch.float8_e4m3fn else tokens
    )
    return torch.index_select(backing, 0, token_indices).view(cache.dtype)
