"""Addressing a per-layer KV buffer's rows by token id when its pages are not
packed: a paged ``(pages, page_size, 1, row)`` buffer keeps token ``t``'s row at

    (t // page_size) * page_stride + (t % page_size) * row_stride

Kernels take ``PAGE_SIZE`` (1 for a flat run of rows) and both strides.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def paged_layout(buffer: torch.Tensor) -> Optional[Tuple[int, int, int]]:
    """``(page_size, page_stride, row_stride)`` in elements when ``buffer`` is
    a paged buffer whose pages are not packed, else None."""
    if buffer.dim() != 4:
        return None
    pages, page_size, heads, _ = buffer.shape
    if page_size == 1 or buffer.stride(0) == page_size * buffer.stride(1):
        return None
    assert heads == 1 and buffer.stride(-1) == 1, (
        f"a paged KV buffer holds one contiguous row per token; got shape "
        f"{tuple(buffer.shape)} strides {buffer.stride()}"
    )
    return page_size, buffer.stride(0), buffer.stride(1)


@triton.jit
def paged_row_offset(loc, page_stride, row_stride, PAGE_SIZE: tl.constexpr):
    """Element offset of token ``loc``'s row; see the module docstring."""
    if PAGE_SIZE == 1:
        return loc * row_stride
    return (loc // PAGE_SIZE) * page_stride + (loc % PAGE_SIZE) * row_stride
