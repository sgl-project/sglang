"""Zero whole page envelopes of the unified pool by physical page id.

The pool is viewed as int64 words (MHA and MLA page envelopes are
8-byte-aligned), one wide element per lane; grid = (num_pages, page word
blocks).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_pages_kernel(
    buf_ptr,  # int64 view of the raw pool buffer
    pages_ptr,  # int64 [M] physical page ids to zero
    page_words,  # int64 words per page envelope
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    blk = tl.program_id(1)
    pg = tl.load(pages_ptr + m).to(tl.int64)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < page_words
    tl.store(buf_ptr + pg * page_words + offs, 0, mask=mask)


_BLOCK = 2048


def zero_pages(
    raw: torch.Tensor,
    pages: torch.Tensor,
    num_pages: int,
    page_bytes: int,
) -> None:
    """Zero the listed physical PAGE envelopes of the uint8 pool `raw`."""
    m = int(pages.numel())
    if m == 0:
        return
    assert raw.dtype == torch.uint8, f"expected uint8 pool, got {raw.dtype}"
    if not raw.is_cuda:
        raw[: num_pages * page_bytes].view(num_pages, page_bytes)[pages] = 0
        return
    assert page_bytes % 8 == 0, f"page_bytes {page_bytes} not int64-aligned"
    page_words = page_bytes // 8
    words = raw[: num_pages * page_bytes].view(torch.int64)
    grid = (m, triton.cdiv(page_words, _BLOCK))
    _zero_pages_kernel[grid](words, pages.to(torch.int64), page_words, BLOCK=_BLOCK)


@triton.jit
def _zero_pages_multi_kernel(
    buf_ptrs,  # int64 [B] base address of each page-major buffer
    buf_page_words,  # int64 [B] int64 words per page in each buffer
    pages_ptr,  # int64 [M] page ids to zero
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    b = tl.program_id(1)
    blk = tl.program_id(2)
    page_words = tl.load(buf_page_words + b)
    if blk * BLOCK >= page_words:
        return
    base = tl.load(buf_ptrs + b).to(tl.pointer_type(tl.int64))
    pg = tl.load(pages_ptr + m).to(tl.int64)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < page_words
    tl.store(base + pg * page_words + offs, 0, mask=mask)


def zero_pages_multi(
    buf_ptrs: torch.Tensor,
    buf_page_words: torch.Tensor,
    max_page_words: int,
    pages: torch.Tensor,
) -> None:
    """Zero the listed pages in every page-major buffer with one launch."""
    m = int(pages.numel())
    if m == 0:
        return
    grid = (m, int(buf_ptrs.numel()), triton.cdiv(max_page_words, _BLOCK))
    _zero_pages_multi_kernel[grid](
        buf_ptrs, buf_page_words, pages.to(torch.int64), BLOCK=_BLOCK
    )
