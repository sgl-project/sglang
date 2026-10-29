from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import _HEAD_DIM, _KV_BLOCK_SIZE
from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    quantize_fp4_indexer_tensor,
)


# FlyDSL split payload / scale layout; bytes equal AITER's fp4 writer up to the sign of zero
@triton.jit
def _store_fp4_index_k_split_kernel(
    k_fp4,
    k_sf,
    payload,
    scale,
    loc,
    page_size: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE8: tl.constexpr,
):
    token_id = tl.program_id(0)
    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size

    # payload [page, 1, 4 chunks, page_size, 16 bytes]: chunk c holds dims [32c, 32c + 32).
    offsets = tl.arange(0, BLOCK)
    chunk = offsets // 16
    byte = offsets - chunk * 16
    k = tl.load(k_fp4 + token_id * BLOCK + offsets)
    sf = tl.load(k_sf + token_id)
    sf_offsets = tl.arange(0, 4)
    sf_bytes = ((sf >> (sf_offsets * 8)) & 0xFF).to(tl.uint8)
    if PAGE8:
        # aiter's row-group 8-slot page: payload [t % 4][chunk][t // 4][16 B], scales [chunk][t]
        page8 = cache_loc // 8
        t = cache_loc % 8
        tl.store(
            payload
            + page8 * (8 * BLOCK)
            + ((t % 4) * 4 + chunk) * 32
            + (t // 4) * 16
            + byte,
            k,
        )
        tl.store(scale + page8 * 32 + sf_offsets * 8 + t, sf_bytes)
        return
    tl.store(
        payload
        + page * (4 * page_size * 16)
        + chunk * (page_size * 16)
        + page_offset * 16
        + byte,
        k,
    )
    # scale [page, 1, 4, page_size]: the slot axis is a 16 x 4 tile transposed (the FlyDSL K ABI)
    shuffled = (page_offset % 16) * 4 + page_offset // 16
    tl.store(
        scale + page * (4 * page_size) + sf_offsets * page_size + shuffled, sf_bytes
    )


def store_fp4_index_k_cache_split(
    input: torch.Tensor,
    payload: torch.Tensor,
    scale: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool = False,
    page8: bool = False,
) -> None:
    """Quantize input [n, 128] to fp4 (per-32 ue8m0) and scatter row i to slot
    loc[i] of the split FlyDSL K layout (payload [pages, 1, 4, page_size, 16],
    scale [pages, 1, 4, page_size] with the slot axis 16 x 4 transposed); with page8
    each page holds eight 8-slot pages of aiter's row-group layout (pack_kv_cache)."""
    assert input.shape[-1] == _HEAD_DIM
    assert page_size == _KV_BLOCK_SIZE, "the 16 x 4 slot transpose spans a 64-slot page"
    assert payload.shape[1:] == (1, 4, page_size, 16), payload.shape
    assert scale.shape[1:] == (1, 4, page_size), scale.shape
    k_fp4, k_sf = quantize_fp4_indexer_tensor(input.contiguous(), rne=rne)
    n_tokens = k_fp4.shape[0]
    if n_tokens == 0:
        return
    _store_fp4_index_k_split_kernel[(n_tokens,)](
        k_fp4.view(torch.uint8),
        k_sf,
        payload.view(torch.uint8),
        scale,
        loc,
        page_size,
        BLOCK=64,
        PAGE8=page8,
    )
