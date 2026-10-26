"""Paged views over per-slot KV buffers.

Attention kernels take a paged ``[num_pages, page_size, ...]`` cache. These
helpers split a pool buffer's slot dim into pages from the buffer's own
strides, so a strided per-layer view keeps its slot stride at every page size.
"""

import torch


def paged_view(flat: torch.Tensor, page_size: int) -> torch.Tensor:
    """``[slots, ...]`` -> ``[slots // page_size, page_size, ...]``, as a view.

    Built with ``as_strided``, not ``view``: dim 1 must carry the slot stride
    at every page size, and ``view`` gives a size-1 dim (``page_size == 1``)
    the row stride instead, which differs from the slot stride whenever the
    per-layer view is strided.
    """
    num_slots = int(flat.shape[0])
    assert num_slots % page_size == 0, (num_slots, page_size)
    slot_stride = flat.stride(0)
    return flat.as_strided(
        (num_slots // page_size, page_size, *flat.shape[1:]),
        (slot_stride * page_size, slot_stride, *flat.stride()[1:]),
    )


def paged_kv_view(
    buf: torch.Tensor, page_size: int, head_num: int, head_dim: int
) -> torch.Tensor:
    """An MHA pool buffer as ``[num_pages, page_size, head_num, head_dim]``.

    A slot-major buffer has its rows shaped and its pages split from its own
    strides; an HND buffer (``[num_pages, head_num, page_size, head_dim]``) is
    already page-split and only has its head and token dims swapped.
    """
    if buf.dim() <= 3:
        return paged_view(buf.view(buf.shape[0], head_num, head_dim), page_size)
    assert tuple(buf.shape[1:]) == (head_num, page_size, head_dim), (
        f"expected an HND buffer [pages, {head_num}, {page_size}, {head_dim}], "
        f"got {tuple(buf.shape)}"
    )
    return buf.permute(0, 2, 1, 3)


def paged_row_view(flat: torch.Tensor, page_size: int) -> torch.Tensor:
    """``[slots, ...]`` -> ``[slots // page_size, page_size, row_elems]``, as a view."""
    return paged_view(flat.view(int(flat.shape[0]), -1), page_size)
