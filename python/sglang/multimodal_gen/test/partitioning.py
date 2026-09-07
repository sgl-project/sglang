from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartitionItem:
    kind: str
    item_id: str
    est_time: float
    used_fallback_estimate: bool = False


def partition_items_by_lpt(
    items: list[PartitionItem], num_partitions: int
) -> list[list[PartitionItem]]:
    if not items or num_partitions <= 0:
        return []

    sorted_items = sorted(
        items,
        key=lambda item: (-item.est_time, item.kind, item.item_id),
    )
    partitions: list[list[PartitionItem]] = [[] for _ in range(num_partitions)]
    partition_sums = [0.0] * num_partitions

    for item in sorted_items:
        min_idx = partition_sums.index(min(partition_sums))
        partitions[min_idx].append(item)
        partition_sums[min_idx] += item.est_time

    return partitions


def assign_partition(
    items: list[PartitionItem], partition_id: int, num_partitions: int
) -> list[PartitionItem]:
    """Return the LPT slice of ``items`` owned by ``partition_id``.

    The LPT pass is deterministic, so shards that each call this with the same
    item list cover the list exactly once between them.
    """
    partitions = partition_items_by_lpt(items, num_partitions)
    if partition_id < 0 or partition_id >= len(partitions):
        return []
    return partitions[partition_id]
