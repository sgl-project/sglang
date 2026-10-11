"""Copy disjoint physical page envelopes without a gathered KV temporary."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_pages_kernel(
    buf,
    src_pages,
    dst_pages,
    page_words,
    SRC_STRIDE: tl.constexpr,
    DST_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0)
    src = tl.load(src_pages + pair * SRC_STRIDE).to(tl.int64)
    dst = tl.load(dst_pages + pair * DST_STRIDE).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < page_words
    values = tl.load(buf + src * page_words + offsets, mask=mask)
    tl.store(buf + dst * page_words + offsets, values, mask=mask)


_BLOCK = 2048


def copy_pages(
    raw: torch.Tensor,
    dst_pages: torch.Tensor,
    src_pages: torch.Tensor,
    num_pages: int,
    page_bytes: int,
) -> None:
    """Copy src -> dst on the current stream; dst and src sets must be disjoint.

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
        SRC_STRIDE=src_pages.stride(0),
        DST_STRIDE=dst_pages.stride(0),
        BLOCK=_BLOCK,
    )
