"""Shard assignment invariants for the diffusion suites.

Lanes that hardcode ``--total-partitions`` (AMD) must still schedule the whole
suite when standalone files outnumber the shards, and the shards must agree on
who runs what without talking to each other.
"""

import pytest

from sglang.multimodal_gen.test.partitioning import PartitionItem, assign_partition
from sglang.multimodal_gen.test.run_suite import build_local_partition_assignment
from sglang.multimodal_gen.test.server.gpu_cases import (
    PARAMETRIZED_CASE_GROUPS,
    STANDALONE_FILES,
)


def _items(*est_times: float) -> list[PartitionItem]:
    return [
        PartitionItem(kind="case", item_id=f"case-{idx}", est_time=est_time)
        for idx, est_time in enumerate(est_times)
    ]


def _expected_work(suite: str) -> tuple[list[str], list[str]]:
    case_ids = [
        case.id
        for _, case_group in PARAMETRIZED_CASE_GROUPS[suite]
        for case in case_group
    ]
    return case_ids, list(STANDALONE_FILES.get(suite, []))


def test_assign_partition_covers_every_item_once():
    items = _items(300.0, 120.0, 600.0, 60.0, 180.0)
    assigned = [
        item.item_id for rank in range(3) for item in assign_partition(items, rank, 3)
    ]
    assert sorted(assigned) == sorted(item.item_id for item in items)


def test_assign_partition_is_empty_outside_the_shard_range():
    items = _items(300.0, 120.0)
    assert assign_partition(items, 5, 3) == []
    assert assign_partition(items, -1, 3) == []
    assert assign_partition(items, 0, 0) == []


@pytest.mark.parametrize("suite", sorted(PARAMETRIZED_CASE_GROUPS))
@pytest.mark.parametrize("total_partitions", [1, 2, 3, 4, 8])
def test_suite_is_fully_scheduled_for_any_shard_count(suite, total_partitions):
    expected_case_ids, expected_standalone_files = _expected_work(suite)

    scheduled_case_ids: list[str] = []
    scheduled_standalone_files: list[str] = []
    for partition_id in range(total_partitions):
        assignment = build_local_partition_assignment(
            suite=suite,
            partition_id=partition_id,
            total_partitions=total_partitions,
        )
        scheduled_case_ids.extend(assignment.case_ids)
        scheduled_standalone_files.extend(assignment.standalone_files)

    # More standalone files than shards used to abort the run; they now share
    # shards with the parametrized cases instead.
    assert sorted(scheduled_case_ids) == sorted(expected_case_ids)
    assert sorted(scheduled_standalone_files) == sorted(expected_standalone_files)
