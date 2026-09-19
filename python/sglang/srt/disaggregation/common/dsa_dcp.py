"""Byte descriptors for DSA index-K transfer into an interleaved DCP cache."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np


def build_dsa_dcp_transfer_blocks(
    src_ptrs,
    dst_ptrs,
    src_item_lens,
    dst_item_lens,
    src_pages,
    dst_pages,
    *,
    page_size: int,
    dcp_size: int,
    dcp_rank: int,
) -> list[tuple[int, int, int]]:
    """Transfer full source pages, including padding in the final page.

    Pointer lists must already be paired by layer. CUDA index-K pages contain
    [page_size * 128 key bytes][page_size * 4 scale bytes], not 132-byte rows.
    Destination page ids are physical (widened allocator ids / DCP page size).
    """
    if not (len(src_ptrs) == len(dst_ptrs) == len(src_item_lens) == len(dst_item_lens)):
        raise ValueError("DSA DCP index-cache entries must be paired by layer")
    if page_size != 64 or dcp_size <= 1 or not 0 <= dcp_rank < dcp_size:
        raise ValueError("DSA DCP requires page size 64 and a valid DCP rank")
    src_pages = np.asarray(src_pages, dtype=np.int64)
    dst_pages = np.asarray(dst_pages, dtype=np.int64)
    offsets = np.arange(dcp_rank, len(src_pages) * page_size, dcp_size)
    local = offsets // dcp_size
    if len(local) and local[-1] >= len(dst_pages) * page_size:
        raise ValueError("DSA DCP destination has too few index-cache pages")
    src_page = src_pages[offsets // page_size]
    src_slot = offsets % page_size
    dst_page = dst_pages[local // page_size]
    dst_slot = local % page_size
    page_bytes = page_size * 132
    blocks = []
    for src, dst, src_len, dst_len in zip(
        src_ptrs, dst_ptrs, src_item_lens, dst_item_lens
    ):
        if src_len == dst_len == 0:  # Layer reuses an earlier layer's top-k.
            continue
        if src_len != page_bytes or dst_len != page_bytes:
            raise ValueError("DSA DCP requires CUDA index-K pages with 128-byte keys")
        for width, base in ((128, 0), (4, page_size * 128)):
            src_offsets = src_page * page_bytes + base + src_slot * width
            dst_offsets = dst_page * page_bytes + base + dst_slot * width
            blocks.extend(
                (int(src + s), int(dst + d), width)
                for s, d in zip(src_offsets, dst_offsets)
            )
    return blocks


def iter_dsa_dcp_transfer_batches(
    src_ptrs,
    dst_ptrs,
    src_item_lens,
    dst_item_lens,
    src_pages,
    dst_pages,
    *,
    page_size: int,
    dcp_size: int,
    dcp_rank: int,
    max_dst_pages_per_batch: int = 128,
) -> Iterator[list[tuple[int, int, int]]]:
    """Bound descriptor construction and submission for long DSA transfers.

    Each batch covers one layer and at most 128 destination pages (16,384
    descriptors for 64-token pages). Start source slices at multiples of
    dcp_size pages so resetting the local position preserves the cyclic shard
    assignment, including when dcp_size does not divide page_size.
    """
    if not (len(src_ptrs) == len(dst_ptrs) == len(src_item_lens) == len(dst_item_lens)):
        raise ValueError("DSA DCP index-cache entries must be paired by layer")
    if page_size != 64 or dcp_size <= 1 or not 0 <= dcp_rank < dcp_size:
        raise ValueError("DSA DCP requires page size 64 and a valid DCP rank")
    if max_dst_pages_per_batch <= 0:
        raise ValueError("max_dst_pages_per_batch must be positive")
    src_pages = np.asarray(src_pages, dtype=np.int64)
    dst_pages = np.asarray(dst_pages, dtype=np.int64)
    local_tokens = max(
        0, (len(src_pages) * page_size + dcp_size - 1 - dcp_rank) // dcp_size
    )
    if local_tokens > len(dst_pages) * page_size:
        raise ValueError("DSA DCP destination has too few index-cache pages")
    src_pages_per_batch = max_dst_pages_per_batch * dcp_size
    for src, dst, src_len, dst_len in zip(
        src_ptrs, dst_ptrs, src_item_lens, dst_item_lens
    ):
        if src_len == dst_len == 0:
            continue
        if src_len != page_size * 132 or dst_len != page_size * 132:
            raise ValueError("DSA DCP requires CUDA index-K pages with 128-byte keys")
        for start in range(0, len(src_pages), src_pages_per_batch):
            dst_start = start // dcp_size
            blocks = build_dsa_dcp_transfer_blocks(
                [src],
                [dst],
                [src_len],
                [dst_len],
                src_pages[start : start + src_pages_per_batch],
                dst_pages[dst_start : dst_start + max_dst_pages_per_batch],
                page_size=page_size,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
            if blocks:
                yield blocks


class DSAPackPlan(NamedTuple):
    src_ptr: int
    src_pages: np.ndarray
    num_tokens: int
    packed_bytes: int
    blocks: list[tuple[int, int, int]]


def iter_dsa_dcp_pack_plans(
    src_ptrs,
    dst_ptrs,
    src_item_lens,
    dst_item_lens,
    src_pages,
    dst_pages,
    *,
    page_size: int,
    dcp_size: int,
    dcp_rank: int,
    pack_ptr: int,
    pack_bytes: int,
) -> Iterator[DSAPackPlan]:
    """Plan one layer at a time in an existing, bounded GPU scratch buffer.

    Packed pages retain the native [64 keys][64 scales] layout. Only valid
    regions of a partial final page are sent; its unused slots are untouched.
    Source batches start at multiples of dcp_size pages, preserving cyclic
    ownership even when dcp_size does not divide 64.
    """
    if not (len(src_ptrs) == len(dst_ptrs) == len(src_item_lens) == len(dst_item_lens)):
        raise ValueError("DSA DCP index-cache entries must be paired by layer")
    if page_size != 64 or dcp_size <= 1 or not 0 <= dcp_rank < dcp_size:
        raise ValueError("DSA DCP requires page size 64 and a valid DCP rank")
    page_bytes = page_size * 132
    # At most 16.5 MiB of the already registered buffer and, for DCP8,
    # 128 KiB of source page IDs. Never resize the GPU pack buffer.
    capacity = min(pack_bytes // page_bytes, 2048)
    if capacity <= 0:
        raise ValueError("DSA pack buffer must fit at least one page")
    src_pages = np.asarray(src_pages, dtype=np.int64)
    dst_pages = np.asarray(dst_pages, dtype=np.int64)
    local_tokens = max(
        0, (src_pages.size * page_size + dcp_size - 1 - dcp_rank) // dcp_size
    )
    if local_tokens > dst_pages.size * page_size:
        raise ValueError("DSA DCP destination has too few index-cache pages")
    for src, dst, src_len, dst_len in zip(
        src_ptrs, dst_ptrs, src_item_lens, dst_item_lens
    ):
        if src_len == dst_len == 0:
            continue
        if src_len != page_bytes or dst_len != page_bytes:
            raise ValueError("DSA DCP requires CUDA index-K pages with 128-byte keys")
        for start in range(0, src_pages.size, capacity * dcp_size):
            pages = src_pages[start : start + capacity * dcp_size]
            n = max(0, (pages.size * page_size + dcp_size - 1 - dcp_rank) // dcp_size)
            if n == 0:
                continue
            full, tail = divmod(n, page_size)
            dst_start = start // dcp_size
            dest = dst_pages[dst_start : dst_start + full]
            blocks = []
            # Merge whole pages only when destination addresses are contiguous.
            if full:
                boundaries = np.r_[0, np.flatnonzero(np.diff(dest) != 1) + 1, full]
                for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                    blocks.append(
                        (
                            int(pack_ptr + lo * page_bytes),
                            int(dst + dest[lo] * page_bytes),
                            int((hi - lo) * page_bytes),
                        )
                    )
            if tail:
                source = pack_ptr + full * page_bytes
                target = int(dst + dst_pages[dst_start + full] * page_bytes)
                blocks.extend(
                    (
                        (source, target, tail * 128),
                        (source + 8192, target + 8192, tail * 4),
                    )
                )
            yield DSAPackPlan(
                int(src), pages, n, (full + bool(tail)) * page_bytes, blocks
            )
