"""Read logical pages from the explicit PA-NZ storage of the NPU MLA cache."""

import torch


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
