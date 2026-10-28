"""Read logical pages from the explicit PA-NZ storage of the NPU MLA cache."""

import torch


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
