"""Copy disjoint physical page envelopes without a gathered KV temporary."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Callers pass page-id slices at any element offset (compaction launches a long
# move in chunks); each program loads only one scalar id, so specializing on
# their alignment would only compile more variants, and those mid-serving.
@triton.jit(do_not_specialize_on_alignment=["src_pages", "dst_pages"])
def _copy_pages_kernel(
    buf,
    src_pages,
    dst_pages,
    page_words,
    page_stride,
    layer_stride,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYERS: tl.constexpr = 1,
    STRIDED_TILES: tl.constexpr = False,
):
    pair = tl.program_id(0)
    src = tl.load(src_pages + pair * SRC_STRIDE).to(tl.int64)
    dst = tl.load(dst_pages + pair * DST_STRIDE).to(tl.int64)
    if src != dst:
        if STRIDED_TILES:
            for layer in tl.range(LAYERS):
                for tile in tl.range(
                    tl.program_id(1), tl.cdiv(page_words, BLOCK), tl.num_programs(1)
                ):
                    _copy_page_tile(
                        buf + layer.to(tl.int64) * layer_stride,
                        src,
                        dst,
                        page_stride,
                        page_words,
                        tile,
                        BLOCK,
                    )
        else:
            _copy_page_tile(
                buf, src, dst, page_stride, page_words, tl.program_id(1), BLOCK
            )


@triton.jit
def _copy_page_tile(buf, src, dst, page_stride, page_words, tile, BLOCK: tl.constexpr):
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < page_words
    values = tl.load(buf + src * page_stride + offsets, mask=mask)
    tl.store(buf + dst * page_stride + offsets, values, mask=mask)


_BLOCK = 2048


def copy_pages(
    raw: torch.Tensor,
    dst_pages: torch.Tensor,
    src_pages: torch.Tensor,
    num_pages: int,
    page_bytes: int,
) -> None:
    """Copy src -> dst on the current stream; dst and src sets must be disjoint,
    but for pairs naming one page for both, which copy nothing.

    The allocator supplies unique destinations in free space. Overlapping float
    shifts are already split into ordered singleton calls by the allocator.
    Reinterpreting the buffer is a view; the GPU copy has no KV-sized scratch.
    """
    count = dst_pages.numel()
    assert count == src_pages.numel()
    if count == 0:
        return
    assert raw.dtype == torch.uint8 and raw.is_contiguous()
    if not raw.is_cuda:
        env = raw[: num_pages * page_bytes].view(num_pages, page_bytes)
        env[dst_pages] = env[src_pages]
        return
    # Actual MHA/MLA pages are 8-byte aligned. Keep unaligned layouts valid.
    wide = page_bytes % 8 == 0 and raw.storage_offset() % 8 == 0
    words = raw[: num_pages * page_bytes].view(torch.int64 if wide else torch.uint8)
    page_words = page_bytes // (8 if wide else 1)
    _copy_pages_kernel[(count, triton.cdiv(page_words, _BLOCK))](
        words,
        src_pages,
        dst_pages,
        page_words,
        page_words,
        0,
        SRC_STRIDE=src_pages.stride(0),
        DST_STRIDE=dst_pages.stride(0),
        BLOCK=_BLOCK,
    )


def copy_layered_pages(
    state: torch.Tensor, *, src: torch.Tensor, dst: torch.Tensor
) -> None:
    """Copy disjoint state rows in a [layers, slots, ...payload] tensor."""
    if state.numel() == 0 or dst.numel() == 0:
        return
    if not state.is_cuda:
        keep = src != dst
        state[:, dst[keep]] = state[:, src[keep]]
        return

    layers, slots = state.shape[:2]
    rows = state.view(layers, slots, -1)
    assert rows.stride(-1) == 1, "state row payload must be contiguous"
    wide = all(
        value * rows.element_size() % 8 == 0
        for value in (rows.storage_offset(), *rows.stride()[:2], rows.shape[-1])
    )
    words = rows.view(torch.int64 if wide else torch.uint8)
    # Bound the launch size even when most pairs are no-ops on a wide beam.
    tiles = min(triton.cdiv(words.shape[-1], _BLOCK), 8)
    _copy_pages_kernel[(dst.numel(), tiles)](
        words,
        src,
        dst,
        words.shape[-1],
        words.stride(1),
        words.stride(0),
        SRC_STRIDE=src.stride(0),
        DST_STRIDE=dst.stride(0),
        BLOCK=_BLOCK,
        LAYERS=layers,
        STRIDED_TILES=True,
    )
