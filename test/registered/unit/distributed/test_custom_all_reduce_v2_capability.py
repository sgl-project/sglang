from unittest.mock import Mock

import pytest

from sglang.srt.distributed.device_communicators import (
    custom_all_reduce_utils,
    custom_all_reduce_v2,
    dpx_hier_all_reduce,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


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
    monkeypatch.setattr(custom_all_reduce_v2, "_is_hip", False)
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


@pytest.mark.parametrize(
    ("hip_version", "expected"),
    [((7, 0, 51831), False), ((7, 1, 0), False), ((7, 2, 0), True), ((7, 17, 0), True)],
)
def test_rocm_runtime_gate(monkeypatch, hip_version, expected):
    """On ROCm a HIP runtime older than MIN_HIP_VERSION disables v2 up front."""
    from sglang.srt.utils import hip_vmm_utils

    group, device = _patch_group(monkeypatch, world_size=8, same_node=True)
    monkeypatch.setattr(custom_all_reduce_v2, "_is_hip", True)
    monkeypatch.setattr(hip_vmm_utils, "hip_runtime_version", lambda: hip_version)
    topology_check = Mock(return_value=True)
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        topology_check,
    )

    assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is expected
    assert topology_check.called is expected


@pytest.mark.parametrize("in_clique", [True, False])
def test_rocm_multinode_uses_fabric_clique(monkeypatch, in_clique):
    """A multi-node ROCm group is admitted only by the gfx1250 fabric probe."""
    from sglang.srt.utils import hip_vmm_utils

    group, device = _patch_group(monkeypatch, world_size=16, same_node=False)
    monkeypatch.setattr(custom_all_reduce_v2, "_is_hip", True)
    monkeypatch.setattr(hip_vmm_utils, "hip_runtime_version", lambda: (7, 17, 0))
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "is_one_nvlink_clique",
        lambda group, device: pytest.fail("ROCm has no NVLink clique to query"),
    )
    from sglang.srt.distributed.device_communicators import gfx1250_fabric

    probe = Mock(return_value=in_clique)
    monkeypatch.setattr(gfx1250_fabric, "probe_fabric_clique", probe)

    assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is in_clique
    probe.assert_called_once_with(group, device)


@pytest.mark.parametrize(
    ("disabled", "has_multicast", "nbytes", "num_blocks", "expected"),
    [
        (False, True, 1024, 4, True),
        (False, True, 1025, 4, False),
        (False, True, 1024, 4096, True),
        (False, True, 1024, 0, False),
        (False, True, 0, 1, False),
        (False, False, 1024, 4, True),
        (True, None, 1024, 4, False),
    ],
)
def test_supports_push(disabled, has_multicast, nbytes, num_blocks, expected):
    """Push is offered within the inclusive slot capacity at any positive grid size."""
    comm = object.__new__(custom_all_reduce_v2.CustomAllReduceV2)
    comm.disabled = disabled
    # A disabled communicator never builds its planes, so these stay unset.
    if not disabled:
        comm.has_multicast = has_multicast
        comm.max_push_size = 1024
    assert comm.supports_push(nbytes=nbytes, num_blocks=num_blocks) is expected


_KB, _MB = 1024, 1024 * 1024


def test_gfx1250_table_enables_hier_only_at_tp16():
    from sglang.srt.distributed.device_communicators.configs import (
        custom_all_reduce_v2 as cfg,
    )

    configs = cfg._rocm_gfx1250_configs.__wrapped__(96)
    assert configs[16].graph.hier_2shot_push == cfg.Range(256 * _KB, 64 * _MB)
    assert configs[16].eager.hier_2shot_push == cfg.Range(256 * _KB, 64 * _MB)
    for world_size in range(2, 9):
        assert configs[world_size].graph.hier_2shot_push.max_bytes == 0


@pytest.mark.parametrize(
    ("max_size", "sizes_own_push", "hier_max", "two_shot"),
    [
        # a pair slot holds half a message and stays within max_size; 2shot_push
        # keeps the sizes above a hier range it no longer covers
        (16 * _MB, False, 32 * _MB, 64 * _MB),
        # hier covers 2shot_push: its slots stay at the hier minimum
        (32 * _MB, False, 64 * _MB, 256 * _KB),
        # push-only, explicit or forced push sizes: no hier planes
        (32 * _MB, True, 0, 64 * _MB),
        # a cap at or below the hier minimum disables it
        (128 * _KB, False, 0, 64 * _MB),
    ],
)
def test_hier_sizing(max_size, sizes_own_push, hier_max, two_shot):
    from sglang.srt.distributed.device_communicators.configs import (
        custom_all_reduce_v2 as cfg,
    )

    config = cfg._rocm_gfx1250_configs.__wrapped__(96)[16]
    got = dpx_hier_all_reduce._hier_max_bytes(
        config, max_size, sizes_own_push=sizes_own_push
    )
    assert got == hier_max
    fitted = dpx_hier_all_reduce._fit_hier(config, got)
    for heuristic in (fitted.graph, fitted.eager):
        assert heuristic.hier_2shot_push.max_bytes == hier_max
        assert heuristic.two_shot_push_threshold == two_shot


def _choice_name(choice):
    if choice is None:
        return None
    return "HIER_2SHOT_PUSH" if choice.use_hier else choice.algo.name


@pytest.mark.parametrize(
    ("nbytes", "expected"),
    [
        (16 * _MB, "HIER_2SHOT_PUSH"),
        (32 * _MB, "HIER_2SHOT_PUSH"),
        # above the clipped hier range, 2shot_push keeps its tuned sizes
        (48 * _MB, "TWO_SHOT_PUSH"),
        (64 * _MB + 32, None),
    ],
)
def test_pick_config_hier_at_default_cap(nbytes, expected):
    from sglang.srt.distributed.device_communicators.configs import (
        custom_all_reduce_v2 as cfg,
    )

    config = cfg._rocm_gfx1250_configs.__wrapped__(96)[16]
    hier_max = dpx_hier_all_reduce._hier_max_bytes(
        config, 16 * _MB, sizes_own_push=False
    )
    comm = object.__new__(custom_all_reduce_v2.CustomAllReduceV2)
    comm.disabled = True
    comm.config = dpx_hier_all_reduce._fit_hier(config, hier_max)
    assert _choice_name(comm._pick_config(nbytes, False)) == expected


@pytest.mark.parametrize(
    ("nbytes", "expected"),
    [
        (256 * _KB, "ONE_SHOT_PUSH"),
        (256 * _KB + 32, "HIER_2SHOT_PUSH"),
        (64 * _MB, "HIER_2SHOT_PUSH"),
        # cannot be halved into whole 16 B vectors; 2shot_push stops at the
        # hier minimum, so NCCL takes it
        (4 * _MB + 16, None),
        (64 * _MB + 32, None),
    ],
)
def test_pick_config_hier_2shot_push(nbytes, expected):
    """With hier_2shot_push enabled, 2shot_push keeps only the sizes below it."""
    from sglang.srt.distributed.device_communicators.configs import (
        custom_all_reduce_v2 as cfg,
    )

    heuristic = cfg.Heuristic(
        256 * _KB,
        0,
        0,
        two_shot_push_threshold=256 * _KB,
        hier_2shot_push=cfg.Range(256 * _KB, 64 * _MB),
    )
    comm = object.__new__(custom_all_reduce_v2.CustomAllReduceV2)
    # dispatch does not look at it; close() at teardown does
    comm.disabled = True
    comm.config = cfg.AllReduceConfig(
        graph=heuristic,
        eager=heuristic,
        num_push_blocks=96,
        num_pull_blocks=32,
        num_mc_blocks=None,
    )
    for use_graph in (False, True):
        assert _choice_name(comm._pick_config(nbytes, use_graph)) == expected


@pytest.mark.parametrize(
    ("cluster_uuid", "peer_host", "expected"),
    [
        # HGX baseboards report a zero cluster UUID and clique 0 on every node.
        (bytes(16), "node-a", True),
        (bytes(16), "node-b", False),
        # An NVL72 rack shares one nonzero cluster UUID across its nodes.
        (b"\x01" * 16, "node-b", True),
    ],
)
def test_zero_cluster_uuid_clique_ends_at_its_node(
    monkeypatch, cluster_uuid, peer_host, expected
):
    monkeypatch.setattr(custom_all_reduce_utils, "_is_hip", False)
    monkeypatch.setattr(
        custom_all_reduce_utils, "_gpu_fabric_clique", lambda device: (cluster_uuid, 0)
    )
    monkeypatch.setattr(custom_all_reduce_utils.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(custom_all_reduce_utils.socket, "gethostname", lambda: "node-a")

    def all_gather_object(gathered, clique, group):
        peer = clique if len(clique) == 2 else (*clique[:2], peer_host)
        gathered[:] = [clique, peer]

    monkeypatch.setattr(
        custom_all_reduce_utils.dist, "all_gather_object", all_gather_object
    )
    # The undecorated function: the CPU runner has no NVML to initialize.
    is_one_nvlink_clique = custom_all_reduce_utils.is_one_nvlink_clique.__wrapped__
    assert is_one_nvlink_clique(object(), object()) is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
