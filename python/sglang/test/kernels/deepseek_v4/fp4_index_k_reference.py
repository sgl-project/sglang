"""Pure-torch readers for the DeepSeek-V4 FP4 index-K cache layouts."""

from typing import Tuple

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
    _KV_BLOCK_SIZE,
    index_k_page8_views,
)


def read_fp4_index_k_split(
    payload: torch.Tensor,
    scale: torch.Tensor,
    slots: torch.Tensor,
    *,
    page_size: int,
    page8: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of store_fp4_index_k_cache_split: (payload int8 [n, 64], scales
    int32 [n] with chunk c's e8m0 byte at bits 8c..8c+7), the layout of
    quantize_fp4_indexer_tensor."""
    assert page_size == _KV_BLOCK_SIZE, "the 16 x 4 slot transpose spans a 64-slot page"
    slots = slots.to(torch.int64)
    if page8:
        kv8, ks8 = index_k_page8_views(payload, scale)
        page, t = slots // 8, slots % 8
        kv8 = kv8.view(-1, 4, 4, 2, 16)  # [page8, t % 4, chunk, t // 4, 16]
        rows = kv8[page, t % 4, :, t // 4, :].reshape(-1, 64).view(torch.int8)
        sf = ks8.view(-1, 4, 8)[page, :, t].to(torch.int32)  # [n, 4]
        packed = sf[:, 0] | (sf[:, 1] << 8) | (sf[:, 2] << 16) | (sf[:, 3] << 24)
        return rows, packed
    page, off = slots // page_size, slots % page_size
    rows = payload.view(torch.uint8)[page, 0, :, off, :]  # [n, 4, 16]
    rows = rows.reshape(-1, 64).view(torch.int8)
    shuffled = (off % 16) * 4 + off // 16
    sf = scale[page, 0, :, shuffled].to(torch.int32)  # [n, 4]
    packed = sf[:, 0] | (sf[:, 1] << 8) | (sf[:, 2] << 16) | (sf[:, 3] << 24)
    return rows, packed
