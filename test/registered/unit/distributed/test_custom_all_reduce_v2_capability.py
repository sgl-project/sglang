from types import SimpleNamespace
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
    ("same_node", "has_fabric_clique", "uses_vmm", "expected"),
    [
        (False, True, True, True),
        (False, False, True, False),
        (False, True, False, False),
        (True, None, None, True),
    ],
)
def test_topology_capability(
    monkeypatch, same_node, has_fabric_clique, uses_vmm, expected
):
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

    def is_vmm_backed(device):
        if same_node:
            pytest.fail("intra-node groups do not need VMM")
        return uses_vmm

    intra_node_capability = Mock(return_value=True)

    monkeypatch.setattr(
        custom_all_reduce_v2,
        "is_one_nvlink_clique",
        is_one_clique,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "_is_vmm_backed_allocator",
        is_vmm_backed,
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


def _workspace_owner(monkeypatch):
    monkeypatch.setattr(custom_all_reduce_v2, "_FORCE_PUSH_SIZE_KB", None)
    monkeypatch.setattr(custom_all_reduce_v2.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(
        custom_all_reduce_v2.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(custom_all_reduce_v2.dist, "barrier", Mock())
    return SimpleNamespace(
        disabled=False,
        max_push_size=3 * 1024**2,
        max_pull_size=16 * 1024**2,
        max_size=16 * 1024**2,
        config=object(),
        device=object(),
        group=object(),
        _capture_started=False,
        _graph_mode_allowed=False,
        _graph_counter=0,
        _graph_inputs=[],
        _init_workspace=Mock(),
    )


def test_reserve_push_preserves_policy_and_is_idempotent(monkeypatch):
    comm = _workspace_owner(monkeypatch)
    policy = comm.config
    reserve = custom_all_reduce_v2.CustomAllReduceV2.reserve_push_capacity
    reserve(comm, 4 * 1024**2 - 1)
    assert comm.max_push_size == 4 * 1024**2
    assert comm.max_size == comm.max_pull_size == 16 * 1024**2
    assert comm.config is policy
    comm._init_workspace.assert_called_once_with()
    custom_all_reduce_v2.torch.cuda.synchronize.assert_called_once_with(comm.device)
    custom_all_reduce_v2.dist.barrier.assert_called_once_with(group=comm.group)
    reserve(comm, 2 * 1024**2)
    reserve(comm, 4 * 1024**2)
    comm._init_workspace.assert_called_once_with()


def test_reserve_push_respects_forced_capacity(monkeypatch):
    comm = _workspace_owner(monkeypatch)
    monkeypatch.setattr(custom_all_reduce_v2, "_FORCE_PUSH_SIZE_KB", 3072)
    custom_all_reduce_v2.CustomAllReduceV2.reserve_push_capacity(comm, 4 * 1024**2)
    assert comm.max_push_size == 3 * 1024**2
    comm._init_workspace.assert_not_called()


def test_reserve_push_rejects_growth_after_push_only_capture(monkeypatch):
    comm = _workspace_owner(monkeypatch)
    # Push-only graphs do not register graph input rows.
    comm._capture_started = True
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        custom_all_reduce_v2.CustomAllReduceV2.reserve_push_capacity(comm, 4 * 1024**2)
    assert comm.max_push_size == 3 * 1024**2
    comm._init_workspace.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
