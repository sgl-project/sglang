"""Zero whole page envelopes of the unified pool by physical page id, alone or
in the launch that binds the pages to their virtual ids.

The pool is viewed as int64 words (MHA and MLA page envelopes are
8-byte-aligned), one wide element per lane; grid = (num_pages, page word
blocks).
"""

from __future__ import annotations

from typing import Optional

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


# Page-id slices start at any element offset and the frontier start moves with
# every take; specializing on either would compile new variants mid-serving.
@triton.jit(
    do_not_specialize=["start_page"],
    do_not_specialize_on_alignment=["virtual_ptr", "pages_ptr"],
)
def _bind_zero_pages_kernel(
    buf_ptr,  # int64 view of the raw pool buffer
    virtual_ptr,  # [M] virtual page ids to bind
    pages_ptr,  # [M] physical page ids, read when HAS_PAGES
    out_ptr,  # int64 [M] physical page ids, written when not HAS_PAGES
    v2p_ptr,  # int64 virtual_to_physical table
    p2v_ptr,  # int64 physical_to_virtual table
    start_page,  # first page of the ascending run, when not HAS_PAGES
    page_words,  # int64 words per page envelope
    HAS_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0)
    blk = tl.program_id(1)
    if HAS_PAGES:
        pg = tl.load(pages_ptr + m).to(tl.int64)
    else:
        pg = start_page.to(tl.int64) + m
    if blk == 0:
        v = tl.load(virtual_ptr + m).to(tl.int64)
        tl.store(v2p_ptr + v, pg)
        tl.store(p2v_ptr + pg, v)
        if not HAS_PAGES:
            tl.store(out_ptr + m, pg)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    tl.store(buf_ptr + pg * page_words + offs, 0, mask=offs < page_words)


def bind_zero_pages(
    raw: torch.Tensor,
    virtual_pages: torch.Tensor,
    v2p: torch.Tensor,
    p2v: torch.Tensor,
    *,
    num_pages: int,
    page_bytes: int,
    start_page: Optional[int] = None,
    pages: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bind ``virtual_pages`` to physical pages in ``v2p``/``p2v`` and zero
    those pages' envelopes of the uint8 pool ``raw``, in one launch. The
    physical pages are ``pages``, or the ascending run from ``start_page``;
    returns them."""
    assert (start_page is None) != (pages is None), "give start_page or pages"
    m = int(virtual_pages.numel())
    device = virtual_pages.device
    if m == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    assert raw.dtype == torch.uint8, f"expected uint8 pool, got {raw.dtype}"
    if not raw.is_cuda:
        if pages is None:
            pages = torch.arange(
                start_page, start_page + m, dtype=torch.int64, device=device
            )
        v = virtual_pages.to(torch.int64)
        p = pages.to(torch.int64)
        v2p[v] = p
        p2v[p] = v
        raw[: num_pages * page_bytes].view(num_pages, page_bytes)[p] = 0
        return pages
    assert page_bytes % 8 == 0, f"page_bytes {page_bytes} not int64-aligned"
    page_words = page_bytes // 8
    words = raw[: num_pages * page_bytes].view(torch.int64)
    out = torch.empty(m, dtype=torch.int64, device=device) if pages is None else pages
    _bind_zero_pages_kernel[(m, triton.cdiv(page_words, _BLOCK))](
        words,
        virtual_pages,
        out,
        out,
        v2p,
        p2v,
        0 if start_page is None else start_page,
        page_words,
        HAS_PAGES=pages is not None,
        BLOCK=_BLOCK,
    )
    return out
