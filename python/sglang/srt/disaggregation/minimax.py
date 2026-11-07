"""Page selection for MiniMax index-K transfers."""

import numpy as np
import torch

from sglang.srt.mem_cache.common import kv_to_page_indices


def minimax_index_transfer_pages(
    token_indices: torch.Tensor, *, page_size: int, prefix_len: int
) -> np.ndarray:
    """Transfer only the suffix; cached or restoring prefix slots must stay untouched."""
    if (
        page_size <= 0
        or not 0 <= prefix_len <= len(token_indices)
        or prefix_len % page_size
    ):
        raise ValueError("MiniMax transfer prefix must be in range and page-aligned")
    return kv_to_page_indices(token_indices[prefix_len:], page_size)
