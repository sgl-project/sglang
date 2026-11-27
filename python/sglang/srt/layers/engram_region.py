"""Placement math for putting the DeepSeek-V4.1 Engram tables in one file-backed
mapping (a regular file on tmpfs, a dev-dax char device, or a CXL DAX region).

The whole TP group shares one backing file, so every table needs a deterministic,
aligned, non-overlapping slice of it. This module computes only those slices; it
imports nothing beyond the standard library, so it can be unit tested without
CUDA, a distributed context or a model checkpoint.

Regions are laid out back to back in the order the row counts are given, so
offsets are unique and regions never overlap by construction. Callers pass the
counts in a stable order (ascending layer id) so that every rank derives the same
offsets without communicating.
"""

from __future__ import annotations

from typing import Sequence

STORE_ALIGN = 2 * 1024 * 1024
"""Region alignment for the file-backed table store, in bytes.

dev-dax rejects a mapping whose offset or length is not a multiple of the 2 MiB
region alignment, and tmpfs/DAX filesystems fold 2 MiB-aligned regions into huge
pages far more readily than arbitrary ones, so one value is used everywhere.
"""


def align_up(n: int, align: int) -> int:
    """Round ``n`` up to a multiple of ``align`` (``align`` must be a power of two)."""
    assert align > 0 and align & (align - 1) == 0, align
    return (n + align - 1) & ~(align - 1)


def engram_row_bytes(dim: int, block_size: int, *, mxfp4: bool = False) -> int:
    """Bytes one table row occupies: the weights followed by their e8m0 scales.

    A row holds ``dim`` fp8 bytes, or ``dim // 2`` bytes when the checkpoint keeps
    the table packed as MXFP4, plus one e8m0 scale per ``block_size`` weights.
    """
    weights = dim // 2 if mxfp4 else dim
    return weights + dim // block_size


def engram_region_offsets(
    row_counts: Sequence[int],
    dim: int,
    block_size: int,
    align: int = STORE_ALIGN,
    *,
    mxfp4: bool = False,
    replicas: Sequence[int] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return ``((offset, region_len), ...)``, one region per ``(table, replica)``.

    Each region is large enough for its rows and padded to ``align``; regions are
    laid out back to back, so they never overlap and every offset is
    ``align``-aligned.

    ``replicas`` gives how many separate regions each table needs: one when every
    rank maps the whole table, and one per TP rank when each rank owns only its own
    shard. Each rank's region is a distinct slice, so two ranks writing their shards
    cannot land on the same bytes. Omitted, every table gets a single region.
    """
    counts = list(row_counts)
    per_table = list(replicas) if replicas is not None else [1] * len(counts)
    if len(per_table) != len(counts):
        raise ValueError(
            f"replicas has {len(per_table)} entries for {len(counts)} row counts"
        )
    row_bytes = engram_row_bytes(dim, block_size, mxfp4=mxfp4)
    regions, offset = [], 0
    for rows, count in zip(counts, per_table):
        region_len = align_up(max(1, rows) * row_bytes, align)
        for _ in range(count):
            regions.append((offset, region_len))
            offset += region_len
    return tuple(regions)


def engram_table_offsets(
    row_counts: Sequence[int],
    dim: int,
    block_size: int,
    tp_rank: int,
    tp_size: int,
    align: int = STORE_ALIGN,
    *,
    mxfp4: bool = False,
    replicated: bool = False,
) -> tuple[int, ...]:
    """The offset of this rank's region for each table, in the given order.

    ``replicated`` is the layout where every rank maps the whole table (one region
    per table, shared). Otherwise each rank gets its own region per table, laid out
    table by table: ``(table, rank)`` regions in order, so rank ``r``'s slice of
    table ``t`` sits at ``(t * tp_size + r)`` in the layout.
    """
    regions = engram_region_offsets(
        row_counts,
        dim,
        block_size,
        align,
        mxfp4=mxfp4,
        replicas=None if replicated else [tp_size] * len(row_counts),
    )
    stride = 1 if replicated else tp_size
    return tuple(
        regions[table * stride + (0 if replicated else tp_rank)][0]
        for table in range(len(row_counts))
    )


def engram_total_bytes(regions: Sequence[tuple[int, int]]) -> int:
    """Bytes the regions occupy together, i.e. the size the backing file needs."""
    return max((offset + length for offset, length in regions), default=0)


def max_shard_rows(
    num_embeddings: int,
    column_bounds: Sequence[int],
    tp_size: int,
    shard_range,
) -> int:
    """The row count of the largest shard this table splits into over ``tp_size``.

    A store region must be sized by the largest shard rather than by the local
    one: with an uneven split the ranks' shards differ by a row, and sizing each
    region on the local count would let the offsets drift apart and overlap.
    """
    return max(
        shard_range(num_embeddings, column_bounds, rank, tp_size)[1]
        - shard_range(num_embeddings, column_bounds, rank, tp_size)[0]
        for rank in range(tp_size)
    )
