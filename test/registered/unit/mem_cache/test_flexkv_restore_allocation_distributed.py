"""Real two-rank Gloo allocation rollback with CPU slots and fake transfers."""

import json
import tempfile
from array import array
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_flexkv_hybrid_radix_cache import _make_layerwise_restore
from test_flexkv_radix_cache_load_back import _make_cache

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


def _two_rank_restore_trace(rank, rendezvous, adapter, layerwise):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        if adapter == "ordinary":
            cache, allocator = _make_cache()
            modes = type(cache._mode)
            cache._mode = modes.IP if layerwise else modes.MP
            prefix = torch.arange(10, 14, dtype=torch.int64)
            prefix_key = RadixKey(array("q", range(4)))
            RadixCache.insert(cache, InsertParams(key=prefix_key, value=prefix))
            prefix_node = RadixCache.match_prefix(
                cache, MatchPrefixParams(key=prefix_key)
            ).last_device_node
        else:
            cache, _, _, _ = _make_layerwise_restore()
            allocator = cache.token_to_kv_pool_allocator
            cache._load_markers.clear()
            prefix = torch.arange(10, 14, dtype=torch.int64)
            prefix_node = object()

        connector = cache.flexkv_connector
        connector.enable_layerwise = layerwise
        held_lookups = set()
        connector.release_pending.side_effect = held_lookups.discard

        # Keep one adapter and one process group across failed attempts and
        # successful retries. -1: all ready; 2: both allocations fail.
        for attempt, failed_rank in enumerate((0, -1, 1, -1, 2, -1)):
            local_ready = failed_rank not in (rank, 2)
            all_ready = failed_rank == -1
            handle = CacheRequestHandle("same-rid", attempt)
            tracking_key = json.dumps([handle.rid, attempt], separators=(",", ":"))
            held_lookups.add(tracking_key)
            slots = torch.arange(100 + attempt * 4, 104 + attempt * 4)
            allocation = slots if local_ready else None
            events = []

            def vote(ready):
                assert ready is local_ready
                events.append("vote")
                ready_tensor = torch.tensor(int(ready), dtype=torch.int64)
                dist.all_reduce(ready_tensor, op=dist.ReduceOp.MIN)
                return bool(ready_tensor.item())

            def transfer(key, mapping):
                assert key == tracking_key
                assert events == ["vote"]
                events.append("transfer")
                torch.testing.assert_close(mapping, slots)
                # A real restore would enter connector collectives here.
                dist.barrier()
                held_lookups.remove(key)
                return len(mapping)

            connector.sync_load_ready = vote
            connector.retrieve_kv.side_effect = transfer
            connector.start_load_kv_layerwise.side_effect = lambda key, mapping: (
                transfer(key, mapping),
                0,
            )
            req = SimpleNamespace(
                rid=handle.rid,
                cache_request_handle=handle,
                last_node=prefix_node,
                prefix_indices=prefix.clone(),
                prefix_len=len(prefix),
                kv=SimpleNamespace(cache_protected_len=4),
            )
            if adapter == "ordinary":
                # A distinct tail per attempt avoids turning the next restore
                # into an already-resident full hit after a successful MP load.
                key = RadixKey(
                    array("q", [*range(4), *range(20 + attempt * 4, 24 + attempt * 4)])
                )
                cache._load_markers[tracking_key] = SimpleNamespace(
                    key=key, value_numel=4
                )
                allocator.alloc.side_effect = lambda _size: allocation
            else:
                cache._load_markers[tracking_key] = SimpleNamespace(device_length=4)
                cache._alloc_restore_slots.return_value = allocation

            frees_before = allocator.free.call_count
            restored, node = cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=prefix_node, host_hit_length=4, req=req
                )
            )

            assert events == (["vote", "transfer"] if all_ready else ["vote"])
            assert not held_lookups
            assert tracking_key not in cache._load_markers
            torch.testing.assert_close(req.prefix_indices, prefix)
            if all_ready:
                assert restored == len(slots)
                assert allocator.free.call_count == frees_before
                request_owned = adapter == "hybrid" or layerwise
                assert (tracking_key in cache._restore_leases) is request_owned
            else:
                assert restored == 0
                assert node is prefix_node
                assert tracking_key not in cache._restore_leases
                assert not hasattr(req, "pending_restore_slots")
                assert allocator.free.call_count == frees_before + int(local_ready)
                if local_ready:
                    torch.testing.assert_close(allocator.free.call_args.args[0], slots)
                connector.release_pending.assert_any_call(tracking_key)

            if adapter == "ordinary":
                existing = RadixCache.match_prefix(
                    cache, MatchPrefixParams(key=prefix_key)
                )
                torch.testing.assert_close(
                    cache.path_device_indices(existing.last_device_node), prefix
                )
                assert existing.last_device_node is prefix_node
            # The same group must remain usable after rollback.
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("adapter", ["ordinary", "hybrid"])
@pytest.mark.parametrize("layerwise", [False, True], ids=["mp", "ip"])
def test_two_rank_restore_allocation_rollback_and_retry(adapter, layerwise):
    with tempfile.TemporaryDirectory(prefix="flexkv-allocation-") as directory:
        mp.spawn(
            _two_rank_restore_trace,
            args=(f"file://{directory}/store", adapter, layerwise),
            nprocs=2,
            join=True,
        )
