"""TP2 shared V4 KV, indexer, and compression-state copies with native waits."""

import json
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.mla_host_shared import SharedCPUHostDedupContext
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")


class _DelayedV4HostPool(DeepSeekV4PagedHostPool):
    delay_cycles = 0

    def load_to_device_per_layer_physical(
        self, device_pool, host_indices, device_indices, layer_id, io_backend, **kwargs
    ):
        if layer_id == self.layer_num - 1 and self.delay_cycles:
            torch.cuda._sleep(self.delay_cycles)
        return super().load_to_device_per_layer_physical(
            device_pool, host_indices, device_indices, layer_id, io_backend, **kwargs
        )


def _load_tree(controller, anchor, rows, node_id):
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.cache_controller = controller
    tree.pp_rank = 0
    tree.buffer_pipeline = None
    tree.metrics_collector = None
    tree.ongoing_load_back = {node_id: (object(), None, None)}
    tree.dec_lock_ref = lambda node, params: None
    tree.dec_host_lock_ref = lambda node, params: None
    tree.tree_core = SimpleNamespace(
        write_back_duplicate_reclaim_digest=0,
        finish_load_back=lambda node: anchor.free(rows),
    )
    tree._all_reduce = lambda values, operation: dist.all_reduce(
        values, op=operation, group=controller.mla_dedup.process_group
    )
    return tree


def _v4_rank(rank, init_method):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=90),
    )
    cpu_group = dist.group.WORLD
    model_group = dist.new_group([0, 1], backend="nccl", timeout=timedelta(seconds=90))
    controller, pools = None, []
    try:
        # Use the actual V4 host/controller/wait implementations. Tiny device
        # regions isolate the copy protocol from model loading and attention.
        root = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        root.size = 32
        root.layer_num = 3
        root.start_layer, root.end_layer = 0, 3
        root.device = "cuda"
        anchor = LogicalHostPool(32, 2, layout="page_first_direct")
        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=anchor,
                device_pool=root,
                layer_mapper=lambda layer: layer,
                is_primary_index_anchor=True,
            )
        ]
        regions = []
        for name, mapping, kind in (
            (PoolName.SWA, {0: 0, 1: 1, 2: 2}, "paged"),
            (PoolName.DEEPSEEK_V4_C4, {0: 0, 2: 1}, "paged"),
            (PoolName.DEEPSEEK_V4_C4_INDEXER, {0: 0, 2: 1}, "paged"),
            (PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE, {0: 0, 2: 1}, "paged"),
            (PoolName.DEEPSEEK_V4_C4_STATE, {0: 0, 2: 1}, "state"),
            (PoolName.DEEPSEEK_V4_C4_INDEXER_STATE, {0: 0, 2: 1}, "state"),
            (PoolName.DEEPSEEK_V4_C128, {1: 0}, "paged"),
            (PoolName.DEEPSEEK_V4_C2, {2: 0}, "paged"),
            (PoolName.DEEPSEEK_V4_C2_INDEXER, {2: 0}, "paged"),
            (PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE, {2: 0}, "paged"),
        ):
            common = dict(
                num_host_pages=16,
                layout="page_first_direct",
                pin_memory=True,
                shared_memory_group=cpu_group,
            )
            if kind == "paged":
                buffers = [
                    torch.empty((16, 256), dtype=torch.uint8, device="cuda")
                    for _ in mapping
                ]
                pool = _DelayedV4HostPool(
                    name.value, buffers, item_bytes=256, slot_page_size=2, **common
                )
            else:
                states = [
                    SimpleNamespace(
                        ring_size=2,
                        kv_score_buffer=SimpleNamespace(
                            kv_score=torch.empty(
                                (32, 64), dtype=torch.float32, device="cuda"
                            )
                        ),
                    )
                    for _ in mapping
                ]
                pool = DeepSeekV4StateHostPool(
                    name.value, states, swa_page_size=2, **common
                )
                buffers = pool.device_page_views
            pools.append(pool)
            # Each region has its own local physical page translation.
            offset = rank * (2 if name == PoolName.SWA else 3)
            device_pool = SimpleNamespace(
                device="cuda",
                host_transfer_translate=lambda values, offset=offset: (
                    ((values // 2 + offset) % 14) * 2 + values % 2
                ),
            )
            entries.append(
                PoolEntry(
                    name=name,
                    host_pool=pool,
                    device_pool=device_pool,
                    layer_mapper=mapping.get,
                )
            )
            regions.append((name, pool, device_pool, mapping, buffers))
        allocator = TokenToKVPoolAllocator(
            32, torch.uint8, "cuda", root, need_sort=False
        )
        controller = HybridCacheController(
            allocator,
            HostPoolGroup(entries),
            page_size=2,
            tp_group=cpu_group,
            load_cache_event=threading.Event(),
            io_backend="direct",
            mla_dedup_context=SharedCPUHostDedupContext(cpu_group),
        )
        assert controller.layer_done_counter.on_layer_ready is None
        assert all(pool.allocator.is_shared for pool in pools)
        logical = torch.tensor([4, 5, 10, 11], dtype=torch.int64, device="cuda")
        model_total = torch.ones(1, device="cuda")
        dist.all_reduce(model_total, group=model_group)
        torch.cuda.synchronize()
        for round_id, scenario in enumerate(
            ("normal", "slow_layer", "slow_rank", "cancel")
        ):
            free_before = anchor.available_size()
            allocated = anchor.alloc(4)
            host_rows = allocated.flip(0).reshape(2, 2).flip(1).flatten()
            transfers, expected = [], {}
            for number, (name, pool, device_pool, mapping, buffers) in enumerate(
                regions
            ):
                physical_rows = device_pool.host_transfer_translate(logical)[::2] // 2
                for layer, buffer in enumerate(buffers):
                    value = torch.full_like(
                        buffer[:2], 10 + number * 7 + layer + round_id
                    )
                    expected[(name, layer)] = value
                    buffer.fill_(245)
                    buffer.index_copy_(
                        0,
                        physical_rows,
                        value if rank == 0 else torch.full_like(value, 246),
                    )
                transfers.append(
                    PoolTransfer(
                        name=name,
                        indices_from_pool=PoolName.KV,
                        host_indices=host_rows,
                        device_indices=logical,
                    )
                )
            controller.set_mla_producer_stream(torch.cuda.current_stream())
            controller.write_queue.append(
                CacheOperation(host_rows, logical, round_id, pool_transfers=transfers)
            )
            controller.start_writing()
            ack = controller.ack_write_queue.pop(0)
            assert ack.finish_event.query()
            assert bool(ack.num_bytes) == (rank == 0)
            for _, pool, _, _, buffers in regions:
                for buffer in buffers:
                    buffer.fill_(244)
                if isinstance(pool, _DelayedV4HostPool):
                    pool.delay_cycles = 0
            # Delay only the SWA region's last layer. Earlier model layers
            # must remain usable, and slow readers must retain shared rows.
            pools[0].delay_cycles = (
                500_000_000
                if scenario == "slow_layer"
                else 2_000_000_000
                if rank == 1 and scenario in ("slow_rank", "cancel")
                else 0
            )
            controller.load_fence_stream = torch.cuda.current_stream()
            controller.load_queue.append(
                CacheOperation(host_rows, logical, round_id, pool_transfers=transfers)
            )
            slot = controller.start_loading()
            controller.layer_done_counter.set_consumer(slot)
            ack = controller.ack_load_queue[0]
            tree = _load_tree(controller, anchor, host_rows, round_id)
            if scenario == "slow_rank":
                if rank == 0:
                    ack.finish_event.synchronize()
                states = [None, None]
                dist.all_gather_object(
                    states, ack.finish_event.query(), group=cpu_group
                )
                assert states == [True, False], states
                tree.loading_check()
                assert len(controller.ack_load_queue) == 1
                assert anchor.available_size() == free_before - 4
            if scenario == "cancel":
                controller.reset()
                assert ack.finish_event.query()
                assert not controller.ack_load_queue
            first_compute = torch.cuda.Event(enable_timing=True)
            snapshots = []
            for layer in range(3):
                root.wait_layer_transfer(layer)
                for name, _, device_pool, mapping, buffers in regions:
                    local = mapping.get(layer)
                    if local is None:
                        continue
                    rows = device_pool.host_transfer_translate(logical)[::2] // 2
                    snapshots.append(
                        (buffers[local].index_select(0, rows), expected[(name, local)])
                    )
                total = snapshots[-1][0].float().sum()
                dist.all_reduce(total, group=model_group)
                if layer == 0:
                    first_compute.record()
            torch.cuda.synchronize()
            assert all(torch.equal(actual, value) for actual, value in snapshots)
            if scenario == "slow_layer":
                assert first_compute.elapsed_time(ack.finish_event) > 0
            if scenario == "cancel":
                anchor.free(host_rows)
            else:
                tree.loading_check()
                assert not controller.ack_load_queue
            assert anchor.available_size() == free_before
        if rank == 0:
            print(
                json.dumps({"shared_v4_tp2": "passed", "regions": len(regions)}),
                flush=True,
            )
    finally:
        if controller is not None:
            controller.drain_shared_transfers(synchronize_peers=False)
        torch.cuda.synchronize()
        for pool in reversed(pools):
            pool.destroy()
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA GPUs")
class TestSharedV4CUDA(CustomTestCase):
    def test_v4_regions_use_local_indices_and_native_layer_waits(self):
        with tempfile.TemporaryDirectory(prefix="shared-v4-cuda-") as directory:
            processes = mp.spawn(
                _v4_rank,
                args=(f"file://{directory}/group",),
                nprocs=2,
                join=False,
            )
            try:
                deadline = time.monotonic() + 110
                while not processes.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail("Shared V4 TP2 test exceeded 110 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
