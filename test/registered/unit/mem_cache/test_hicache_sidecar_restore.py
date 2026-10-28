"""Derived sidecars must follow every destination of a Mamba host-cache hit."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.page_interleave import PageInterleavePlacement, PageShardSpec
from sglang.srt.mem_cache.unified_cache.components.base import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("active_request", [False, True])
@pytest.mark.parametrize("sidecar_first", [False, True])
def test_sidecar_restores_cache_and_request_slots(active_request, sidecar_first):
    node = UnifiedTreeNode((ComponentType.FULL, ComponentType.MAMBA))
    node.component_data[ComponentType.MAMBA].host_value = torch.tensor([2])
    component = object.__new__(MambaComponent)
    transfers = component.build_hicache_transfers(
        node,
        CacheTransferPhase.LOAD_BACK,
        mamba_pool_idx=torch.tensor(7) if active_request else None,
    )
    sidecar = PoolTransfer(name=PoolName.DRAFT, indices_from_pool=PoolName.MAMBA)
    transfers.insert(0 if sidecar_first else len(transfers), sidecar)

    alloc = Mock(return_value=torch.tensor([4]))
    free = Mock()
    controller = object.__new__(HybridCacheController)
    controller.device = "cpu"
    controller.is_kv_sharded = False
    controller.mem_pool_device_allocator = SimpleNamespace()
    controller.load_queue = []
    controller.mem_pool_host = SimpleNamespace(
        anchor_entry=SimpleNamespace(
            name=PoolName.KV, host_pool=SimpleNamespace(size_per_token=16)
        ),
        entry_map={
            PoolName.MAMBA: SimpleNamespace(
                device_alloc_fn=alloc,
                device_free_fn=free,
                device_evict_fn=None,
                host_pool=SimpleNamespace(size_per_token=8),
            ),
            PoolName.DRAFT: SimpleNamespace(
                host_pool=SimpleNamespace(size_per_token=4)
            ),
        },
    )
    result = controller.load(torch.empty(0, dtype=torch.int64), extra_pools=transfers)
    assert result is not None
    alloc.assert_called_once_with(1)
    free.assert_not_called()

    # Simulate the queued copy on CPU: neither destination may retain stale state.
    host = torch.tensor([10, 20, 30, 40])
    device = torch.full((8,), -1)
    device[sidecar.device_indices] = host[sidecar.host_indices]
    expected = torch.full((8,), -1)
    expected[4] = 30
    if active_request:
        expected[7] = 30
    torch.testing.assert_close(device, expected)
    assert controller._transfer_num_bytes(controller.load_queue[0]) == (
        2 if active_request else 1
    ) * (8 + 4)


@pytest.mark.parametrize("shard_rank, expected_bytes", [(0, 44), (1, 36)])
def test_repeated_source_bytes_count_only_local_slots(shard_rank, expected_bytes):
    controller = object.__new__(HybridCacheController)
    controller.is_kv_sharded = True
    controller.mem_pool_host = SimpleNamespace(
        anchor_entry=SimpleNamespace(
            name=PoolName.KV,
            host_pool=SimpleNamespace(
                size_per_token=16,
                shard_rank=shard_rank,
                placement=PageInterleavePlacement(
                    PageShardSpec(shard_rank, 2, 2, 8, 2)
                ),
            ),
        ),
        entry_map={
            PoolName.SWA: SimpleNamespace(host_pool=SimpleNamespace(size_per_token=8)),
            PoolName.DRAFT_SWA: SimpleNamespace(
                host_pool=SimpleNamespace(size_per_token=4)
            ),
        },
    )
    op = CacheOperation(
        host_indices=torch.tensor([0, 1]),
        device_indices=torch.tensor([4, 5]),
        node_id=0,
        pool_transfers=[
            PoolTransfer(PoolName.SWA, host_indices=torch.tensor([0, 2])),
            PoolTransfer(PoolName.SWA, host_indices=torch.tensor([2, 3])),
            PoolTransfer(PoolName.DRAFT_SWA, indices_from_pool=PoolName.SWA),
        ],
    )
    assert controller._transfer_num_bytes(op) == expected_bytes


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if arg != "-f"]
    raise SystemExit(pytest.main([__file__, "-v", "-x", *args]))
