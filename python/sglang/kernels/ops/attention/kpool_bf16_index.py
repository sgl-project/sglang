"""Pooled page-table construction for the NPU BF16 index cache."""

import torch

BLOCK_SIZE_K = 128


def build_pooled_page_table(
    block_tables: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Subsample physical KV block IDs for the NPU pooled index cache.

    The caller supplies 128-token KV page IDs. Select the first column in each
    pool-sized page group, preserving its physical block ID; this does not
    convert the table to 64-token pages or compact the physical addresses.
    """
    assert BLOCK_SIZE_K % pool_size == 0
    idx = torch.arange(0, block_tables.shape[-1], pool_size, device=block_tables.device)
    return block_tables[..., idx]
