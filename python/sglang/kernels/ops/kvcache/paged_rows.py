"""Addressing a per-layer KV buffer's rows by token id when its pages are not
packed back to back.

A layer's rows are usually a flat run of rows, where token ``t``'s row is row
``t``. A paged buffer ``(pages, page_size, 1, row)`` whose pages lie further
apart than ``page_size`` rows (each page entry also holds other layers' rows
and other parts) keeps token ``t``'s row at

    (t // page_size) * page_stride + (t % page_size) * row_stride

Kernels that read or write rows by token id take ``PAGE_SIZE`` (1 for a flat
run) and the two strides; with ``PAGE_SIZE`` 1 the offset is ``t * row_stride``,
the address and the code a flat run always had.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def paged_layout(buffer: torch.Tensor) -> Optional[Tuple[int, int, int]]:
    """``(page_size, page_stride, row_stride)`` in ``buffer``'s elements when
    it is a paged buffer whose pages are not packed, else None (a flat run of
    rows, addressed by its row stride as before)."""
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
