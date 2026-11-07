from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    INDEX_K_SLOT_BYTES,
    e2m1_decode,
)
from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    quantize_fp4_indexer_tensor,
)


@triton.jit
def _store_fp4_index_k_cache_kernel(
    k_fp4,
    k_sf,
    cache,
    loc,
    page_size: tl.constexpr,
    cache_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size

    k = tl.load(k_fp4 + token_id * BLOCK + offsets)
    tl.store(cache + page * cache_stride + page_offset * BLOCK + offsets, k)

    sf = tl.load(k_sf + token_id)
    sf_offsets = tl.arange(0, 4)
    sf_bytes = (sf >> (sf_offsets * 8)) & 0xFF
    tl.store(
        cache + page * cache_stride + page_size * BLOCK + page_offset * 4 + sf_offsets,
        sf_bytes,
    )


def store_fp4_index_k_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool = False,
) -> None:
    assert input.shape[-1] == 128
    k_fp4, k_sf = quantize_fp4_indexer_tensor(input.contiguous(), rne=rne)
    n_tokens = input.numel() // input.shape[-1]
    assert k_fp4.shape == (n_tokens, 64)
    assert k_sf.shape == (n_tokens,)
    assert cache.shape[1] == page_size * INDEX_K_SLOT_BYTES

    if n_tokens == 0:
        return
    _store_fp4_index_k_cache_kernel[(n_tokens,)](
        k_fp4.view(torch.uint8),
        k_sf,
        cache,
        loc,
        page_size,
        cache.stride(0),
        BLOCK=64,
    )


@triton.jit
def _gather_fp4_index_k_kernel(
    cache,
    cache_stride,
    slots,
    payload,
    scales,
    n,
    page_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # int32 view of a page: page_size * 16 payload words, then page_size scale words.
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid = rows < n
    slot = tl.load(slots + rows, valid, 0).to(tl.int64)
    page = slot // page_size
    offset = slot - page * page_size
    page_base = cache + page * cache_stride
    words = tl.arange(0, 16)
    payload_words = tl.load(
        page_base[:, None] + offset[:, None] * 16 + words[None, :], valid[:, None]
    )
    tl.store(
        payload + rows[:, None].to(tl.int64) * 16 + words[None, :],
        payload_words,
        valid[:, None],
    )
    tl.store(scales + rows, tl.load(page_base + page_size * 16 + offset, valid), valid)


def gather_fp4_index_k(
    cache: torch.Tensor, slots: torch.Tensor, *, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed fp4 rows at ``slots`` of the index-K ``cache`` (``[pages, page_size *
    68]`` uint8, ``store_fp4_index_k_cache``'s layout): (payload int8 ``[n, 64]``,
    ue8m0 scales packed int32 ``[n]``), the layout ``quantize_fp4_indexer_tensor``
    returns."""
    assert cache.shape[1] == page_size * INDEX_K_SLOT_BYTES and cache.stride(1) == 1
    # The kernel addresses slots as `slots + rows`, so a strided view would be read
    # as if its elements were packed.
    assert slots.is_contiguous()
    words = cache.view(torch.int32)
    n = slots.shape[0]
    payload = torch.empty((n, 16), device=cache.device, dtype=torch.int32)
    scales = torch.empty((n,), device=cache.device, dtype=torch.int32)
    if n > 0:
        block_rows = 64
        _gather_fp4_index_k_kernel[(triton.cdiv(n, block_rows),)](
            words,
            words.stride(0),
            slots,
            payload,
            scales,
            n,
            page_size,
            BLOCK_ROWS=block_rows,
            num_warps=4,
        )
    return payload.view(torch.int8), scales


@triton.jit
def _gather_fp4_index_k_fp8_kernel(
    cache,
    cache_stride,
    slots,
    out,
    out_scales,
    n,
    page_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    valid = rows < n
    slot = tl.load(slots + rows, valid, 0).to(tl.int64)
    page = slot // page_size
    offset = slot - page * page_size
    page_base = cache + page * cache_stride
    words = tl.arange(0, 16)
    payload = tl.load(
        page_base[:, None] + offset[:, None] * 16 + words[None, :], valid[:, None], 0
    )
    # One int32 word holds four packed bytes in memory order, low byte first;
    # byte b carries codes 2b (low nibble) and 2b + 1 (high nibble).
    codes = tl.reshape(
        (payload[:, :, None] >> (4 * tl.arange(0, 8))[None, None, :]) & 0xF,
        (BLOCK_ROWS, 128),
    )
    values = e2m1_decode(codes)
    sf = tl.load(page_base + page_size * 16 + offset, valid, 0)
    exps = (sf[:, None] >> (8 * tl.arange(0, 4))[None, :]) & 0xFF
    # One exponent per 32-element block; rescale onto the largest so every value
    # stays on the e2m1 grid, exactly representable in fp8 e4m3.
    max_exp = tl.max(exps, 1)
    block_scale = tl.exp2((exps - max_exp[:, None]).to(tl.float32))
    values = tl.reshape(
        tl.reshape(values, (BLOCK_ROWS, 4, 32)) * block_scale[:, :, None],
        (BLOCK_ROWS, 128),
    )
    tl.store(
        out + rows[:, None] * 128 + tl.arange(0, 128)[None, :],
        values.to(tl.float8e4nv),
        valid[:, None],
    )
    tl.store(out_scales + rows, tl.exp2((max_exp - 127).to(tl.float32)), valid)


def gather_fp4_index_k_fp8(
    cache: torch.Tensor, slots: torch.Tensor, *, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 rows at ``slots`` of the fp4 index-K ``cache``:
    (e4m3 ``[n, 128]``, per-token fp32 scale ``[n]``). Each token's 32-element
    block exponents are rescaled onto their maximum, which is the returned
    scale; the unscaled values are on the e2m1 grid and so cast exactly."""
    assert cache.shape[1] == page_size * INDEX_K_SLOT_BYTES and cache.stride(1) == 1
    assert slots.is_contiguous()
    words = cache.view(torch.int32)
    n = slots.shape[0]
    out = torch.empty((n, 128), device=cache.device, dtype=torch.float8_e4m3fn)
    out_scales = torch.empty((n,), device=cache.device, dtype=torch.float32)
    if n > 0:
        block_rows = 32
        _gather_fp4_index_k_fp8_kernel[(triton.cdiv(n, block_rows),)](
            words,
            words.stride(0),
            slots,
            out,
            out_scales,
            n,
            page_size,
            BLOCK_ROWS=block_rows,
            num_warps=4,
        )
    return out, out_scales
