"""Capability selection for custom all-reduce v2.

`can_use_custom_all_reduce_v2()` answers one question: is this process group one
custom AR v2 can serve? Two shapes qualify, and they are gated differently.

* **Intra-node** defers to the node-local NVLink/P2P capability check, which
  covers the cudaIpc graph-input path as well as the eager one.
* **Multi-node (MNNVL)** is gated on the group being a single NVLink fabric
  clique (one NVL72 / MNNVL domain): such a clique shares one address space
  across nodes, so the symm-mem workspace and fabric peer VAs are valid
  group-wide.

Multi-node is deliberately *not* gated on the caching allocator being
VMM-backed. That probe's only consumer is graph zero-copy input registration,
and `CustomAllReduceV2` already disables graph mode on a multi-node group
outright (`_is_graph_mode_supported`), so the path can never run there.

    python -m pytest test/registered/unit/distributed/test_custom_all_reduce_v2_capability.py -v
"""

from unittest.mock import Mock

import pytest

from sglang.srt.distributed.device_communicators import custom_all_reduce_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _patch_group(monkeypatch, *, world_size, same_node):
    group = object()
    device = object()
    monkeypatch.setattr(
        custom_all_reduce_v2.dist,
        "get_world_size",
        lambda group: world_size,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "get_supported_world_sizes",
        lambda: (world_size,),
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "in_the_same_node_as",
        lambda group, source_rank: [same_node] * world_size,
    )
    return group, device


@pytest.mark.parametrize(
    ("same_node", "has_fabric_clique", "expected"),
    [
        (False, True, True),
        (False, False, False),
        (True, None, True),
    ],
)
def test_topology_capability(monkeypatch, same_node, has_fabric_clique, expected):
    world_size = 8 if same_node else 16
    group, device = _patch_group(
        monkeypatch,
        world_size=world_size,
        same_node=same_node,
    )

    def is_one_clique(group, device):
        if same_node:
            pytest.fail("intra-node groups do not need a fabric clique")
        return has_fabric_clique

    intra_node_capability = Mock(return_value=True)

    monkeypatch.setattr(
        custom_all_reduce_v2,
        "is_one_nvlink_clique",
        is_one_clique,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        intra_node_capability,
    )

    assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is expected
    if same_node:
        intra_node_capability.assert_called_once_with(
            group=group,
            device=device,
            supported_world_size=[world_size],
            cls_name="CustomAllReduceV2",
        )
    else:
        intra_node_capability.assert_not_called()


def test_multinode_is_not_gated_on_the_allocator(monkeypatch):
    """BUG REGRESSION (#36429). Multi-node v2 required a VMM-backed caching
    allocator, but that probe guards graph zero-copy input registration, which a
    multi-node group never runs. The requirement was stale from the moment
    graph mode was disabled for multi-node, and it rejected v2 on every default
    launch -- silently costing ~19% throughput on a GB300 NVL72.

    A default launch is exactly this case: plain cudaMalloc from the caching
    allocator, so `is_vmm_pointer` answers False for every probe. The capability
    answer must not depend on it.
    """
    group, device = _patch_group(monkeypatch, world_size=16, same_node=False)
    monkeypatch.setattr(custom_all_reduce_v2, "is_one_nvlink_clique", lambda g, d: True)
    monkeypatch.setattr(custom_all_reduce_v2, "is_vmm_pointer", lambda ptr: False)

    assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is True


def test_multinode_without_the_clique_is_still_rejected(monkeypatch):
    """The topology half stays a hard requirement: admitting v2 on a cross-node
    group that is not one fabric clique would use fabric peer VAs that are not
    valid group-wide."""
    group, device = _patch_group(monkeypatch, world_size=16, same_node=False)
    monkeypatch.setattr(
        custom_all_reduce_v2, "is_one_nvlink_clique", lambda g, d: False
    )
    monkeypatch.setattr(custom_all_reduce_v2, "is_vmm_pointer", lambda ptr: True)

    assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
