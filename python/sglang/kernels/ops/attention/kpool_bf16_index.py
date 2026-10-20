"""Pooled page-table construction for the NPU BF16 index cache."""

import torch

BLOCK_SIZE_K = 128


def build_pooled_page_table_64(
    page_table_64: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Pack one logical pool page into the first token page of each page group."""
    assert BLOCK_SIZE_K % pool_size == 0
    idx = torch.arange(
        0, page_table_64.shape[-1], pool_size, device=page_table_64.device
    )
    return page_table_64[..., idx]
