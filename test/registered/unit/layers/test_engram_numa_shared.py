import os

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import pytest

import sglang.srt.layers.engram as engram
from sglang.srt.layers.engram import (
    _HostTable,
    _NumaGroupPlan,
    _NumaRankInfo,
    _partition_rows,
    _plan_numa_group,
)


def _topology(nodes, hosts=None, pci_ids=None):
    hosts = hosts or ["worker"] * len(nodes)
    pci_ids = pci_ids or [f"0000:{rank + 0x10:02x}:00.0" for rank in range(len(nodes))]
    return [
        _NumaRankInfo(rank, hosts[rank], rank, pci_ids[rank], node, len(nodes))
        for rank, node in enumerate(nodes)
    ]


def test_two_numa_groups_choose_local_owner_and_rank():
    ranks = _topology([0, 0, 0, 0, 1, 1, 1, 1])
    assert _plan_numa_group(ranks, 0) == _NumaGroupPlan(0, 0, 0, 4, ((0, 0), (1, 4)))
    assert _plan_numa_group(ranks, 5) == _NumaGroupPlan(1, 4, 1, 4, ((0, 0), (1, 4)))


def test_four_numa_groups_choose_one_owner_per_node():
    ranks = _topology([0, 0, 1, 1, 2, 2, 3, 3])
    assert _plan_numa_group(ranks, 7) == _NumaGroupPlan(
        3, 6, 1, 2, ((0, 0), (1, 2), (2, 4), (3, 6))
    )


def test_single_numa_degrades_to_one_owner():
    ranks = _topology([0, 0, 0, 0])
    assert _plan_numa_group(ranks, 2) == _NumaGroupPlan(0, 0, 2, 4, ((0, 0),))


def test_numa_group_load_ranges_cover_full_table_once():
    ranges = [_partition_rows(11, rank, 4) for rank in range(4)]
    assert ranges == [(0, 2), (2, 5), (5, 8), (8, 11)]
    assert [row for start, end in ranges for row in range(start, end)] == list(
        range(11)
    )


@pytest.mark.parametrize(
    "ranks, error",
    [
        (_topology([0, -1]), "unresolved NUMA"),
        (_topology([0, 1], hosts=["worker-a", "worker-b"]), "one host"),
        (
            _topology([0, 1], pci_ids=["0000:10:00.0", "0000:10:00.0"]),
            "duplicate GPU PCI",
        ),
    ],
)
def test_invalid_topology_fails_closed(ranks, error):
    with pytest.raises(RuntimeError, match=error):
        _plan_numa_group(ranks, 0)


class _OneRankGroup:
    rank_in_group = 0
    world_size = 1

    def all_gather_object(self, value):
        return [value]

    def barrier(self):
        return None


@pytest.mark.skipif(
    not hasattr(os, "memfd_create"), reason="numa_shared memfd is Linux-only"
)
def test_mbind_failure_aborts_table_creation(monkeypatch):
    plan = _NumaGroupPlan(0, 0, 0, 1, ((0, 0),))
    monkeypatch.setattr(engram, "_discover_numa_group", lambda group: plan)

    def fail_bind(*args):
        raise RuntimeError("mbind failed")

    monkeypatch.setattr(engram, "bind_memory_to_node", fail_bind)
    with pytest.raises(RuntimeError, match="mbind failed"):
        _HostTable("numa_shared", 4096, "engram_test", _OneRankGroup())
