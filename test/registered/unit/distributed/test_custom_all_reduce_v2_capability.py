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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_allocate_symmetric_memory_uses_group_free_form_under_nvshmem(monkeypatch):
    """With torch's NVSHMEM symmetric-memory backend (selected process-wide,
    e.g. by the Cake SP all-gather matmul route) the group-scoped allocation
    form is refused, so the slab is allocated first and rendezvoused on the
    group; the CUDA backend keeps the group-scoped form."""
    import sys
    import types

    import torch

    calls = []
    fake = types.SimpleNamespace(
        backend="NVSHMEM",
        get_backend=lambda device: fake.backend,
        empty=lambda n, dtype, device: calls.append(("empty", n, dtype)) or "tensor",
        rendezvous=lambda t, name: calls.append(("rendezvous", t, name)) or "handle",
        enable_symm_mem_for_group=lambda name: calls.append(("enable", name)),
    )
    strided = Mock(return_value="p2p_tensor")
    rendezvous_p2p = Mock(return_value="p2p_handle")
    fake_c = types.SimpleNamespace(
        _SymmetricMemory=types.SimpleNamespace(
            empty_strided_p2p=strided, rendezvous=rendezvous_p2p
        )
    )
    group = types.SimpleNamespace(group_name="tp")
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "torch.distributed._symmetric_memory", fake)
        m.setitem(sys.modules, "torch._C._distributed_c10d", fake_c)
        tensor, handle = custom_all_reduce_v2._allocate_symmetric_memory(
            4096, device=torch.device("cpu"), group=group
        )
        assert (tensor, handle) == ("tensor", "handle")
        assert calls == [
            ("empty", 4096, torch.uint8),
            ("rendezvous", "tensor", "tp"),
        ]
        strided.assert_not_called()

        fake.backend = "CUDA"
        calls.clear()
        tensor, handle = custom_all_reduce_v2._allocate_symmetric_memory(
            4096, device=torch.device("cpu"), group=group
        )
        assert (tensor, handle) == ("p2p_tensor", "p2p_handle")
        strided.assert_called_once()
        assert strided.call_args.args[4] == "tp"
        assert not calls or calls == [("enable", "tp")]
