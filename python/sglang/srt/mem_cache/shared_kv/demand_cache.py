"""Model-neutral GPU row cache used by Shared-KV kernel integrations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEMAND_CACHE_TAG_BYTES = 8
DEMAND_CACHE_STATS_COUNTERS = 5
_EPOCH_BASE = 1 << 30
_MAX_EPOCH = (1 << 31) - 1


@dataclass(frozen=True)
class RowDemandCacheView:
    rows: torch.Tensor
    tags: torch.Tensor
    stats: torch.Tensor | None
    epoch: int
    ways: int


def row_demand_cache_bytes(*, rows: int, row_bytes: int) -> int:
    if rows <= 0 or row_bytes <= 0:
        raise ValueError("row demand-cache dimensions must be positive")
    return rows * (row_bytes + DEMAND_CACHE_TAG_BYTES)


class TransientRowDemandCache:
    """A phase-local row cache whose tags are invalidated by an epoch.

    The storage and lifecycle are model-neutral. The consuming kernel owns the
    tag encoding, hash, fill, and release/acquire protocol.
    """

    def __init__(
        self,
        *,
        rows: int,
        row_bytes: int,
        ways: int,
        device: str | torch.device,
        collect_stats: bool,
    ) -> None:
        if ways <= 0:
            raise ValueError("row demand-cache ways must be positive")
        sets, remainder = divmod(rows, ways)
        if remainder or sets <= 0 or sets & (sets - 1):
            raise ValueError(
                "row demand-cache rows must form a power-of-two number of sets"
            )
        if row_bytes <= 0 or row_bytes % 16:
            raise ValueError("row demand-cache row bytes must be 16-byte aligned")

        self.num_rows = rows
        self.row_bytes = row_bytes
        self.ways = ways
        self.num_sets = sets
        self.rows = torch.empty((rows, row_bytes), dtype=torch.uint8, device=device)
        self.tags = torch.zeros((sets, ways), dtype=torch.int64, device=device)
        self.stats = (
            torch.zeros(
                DEMAND_CACHE_STATS_COUNTERS,
                dtype=torch.int64,
                device=device,
            )
            if collect_stats
            else None
        )
        self._epoch = _EPOCH_BASE - 1

    def next_view(self) -> RowDemandCacheView:
        self._epoch += 1
        if self._epoch > _MAX_EPOCH:
            self.tags.zero_()
            self._epoch = _EPOCH_BASE
        return RowDemandCacheView(
            rows=self.rows,
            tags=self.tags,
            stats=self.stats,
            epoch=self._epoch,
            ways=self.ways,
        )

    @property
    def allocated_bytes(self) -> int:
        stats_bytes = self.stats.nbytes if self.stats is not None else 0
        return (
            row_demand_cache_bytes(rows=self.num_rows, row_bytes=self.row_bytes)
            + stats_bytes
        )
