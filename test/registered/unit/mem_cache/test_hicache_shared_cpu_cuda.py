"""Real TP2 shared-host DMA, native layer waits, and reader retirement."""

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
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.srt.mem_cache.mla_host_shared import SharedCPUHostDedupContext
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
    make_dsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")


class _DelayedHostPool(MLATokenToKVPoolHost):
    delay_cycles = 0

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        if layer_id == self.layer_num - 1 and self.delay_cycles:
            # Delay only this real DMA submission; do not replace the copy.
            torch.cuda._sleep(self.delay_cycles)
        return super().load_to_device_per_layer_physical(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )


def _load_tree(controller, host_pool, host_indices, node_id):
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
        finish_load_back=lambda node: host_pool.free(host_indices),
    )
    tree._all_reduce = lambda values, operation: dist.all_reduce(
        values, op=operation, group=controller.mla_dedup.process_group
    )
    return tree


def _run_shared_rank(rank, init_method, with_indexer):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=60),
    )
    cpu_group = dist.group.WORLD
    model_group = dist.new_group([0, 1], backend="nccl", timeout=timedelta(seconds=60))
    controller = None
    host_pools = []
    try:
        fields = dict(
            size=512,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=128,
            qk_rope_head_dim=0,
            layer_num=3,
            device="cuda",
            enable_memory_saver=False,
        )
        if with_indexer:
            pool = DSATokenToKVPool(
                **fields,
                index_head_dim=128,
                kv_cache_dim=128,
                skip_topk_layers=[True, False, False],
            )
        else:
            pool = MLATokenToKVPool(**fields)
        # Whole physical pages differ by rank; logical token order stays fixed.
        pool.host_transfer_translate = lambda indices: (
            ((indices // 64 + rank * 2) % 7) * 64 + indices % 64
        )
        host = _DelayedHostPool(
            pool,
            host_to_device_ratio=2,
            host_size=0,
            page_size=64,
            layout="page_first_direct",
            pin_memory=True,
            shared_memory_group=cpu_group,
        )
        host_pools.append(host)
        entries = [
            PoolEntry(
                name=PoolName.KV,
                host_pool=host,
                device_pool=pool,
                layer_mapper=lambda layer: layer,
                is_primary_index_anchor=True,
            )
        ]
        if with_indexer:
            index_host = DSAIndexerPoolHost(
                make_dsa_indexer_pool_decl(pool),
                host,
                shared_memory_group=cpu_group,
            )
            host_pools.append(index_host)
            entries.append(
                PoolEntry(
                    name=PoolName.INDEXER,
                    host_pool=index_host,
                    device_pool=pool,
                    layer_mapper=lambda layer: layer,
                )
            )
            assert pool.index_k_with_scale_buffer[0].numel() == 0
        allocator = TokenToKVPoolAllocator(
            pool.size, pool.dtype, "cuda", pool, need_sort=False
        )
        controller = HybridCacheController(
            allocator,
            HostPoolGroup(entries),
            page_size=64,
            tp_group=cpu_group,
            load_cache_event=threading.Event(),
            io_backend="direct",
            mla_dedup_context=SharedCPUHostDedupContext(cpu_group),
        )
        assert controller.layer_done_counter.on_layer_ready is None
        assert not controller.mla_dedup.layerwise_overlap
        assert all(p.allocator.is_shared for p in host_pools)
        assert host.allocator.is_owner == (rank == 0)
        logical = torch.cat((torch.arange(64, 128), torch.arange(192, 256)))
        local_rows = pool.host_transfer_translate(logical).cuda()
        local_pages = local_rows[::64] // 64
        # Warm the independent model communicator before checking overlap.
        total = torch.ones(1, device="cuda")
        dist.all_reduce(total, group=model_group)
        torch.cuda.synchronize()
        outcomes = []
        for round_id, scenario in enumerate(
            ("warmup", "warmup", "slow_layer", "slow_rank", "cancel", "reuse")
        ):
            free_before = host.available_size()
            allocated = host.alloc(128)
            assert allocated is not None
            host_rows = torch.cat((allocated[64:], allocated[:64]))
            extras = (
                [
                    PoolTransfer(
                        name=PoolName.INDEXER,
                        indices_from_pool=PoolName.KV,
                        host_indices=host_rows,
                        device_indices=logical,
                    )
                ]
                if with_indexer
                else None
            )
            expected = []
            expected_indexer = {}
            for layer, buffer in enumerate(pool.kv_buffer):
                value = (
                    (torch.arange(128, device="cuda") % 32) + layer * 32 + round_id * 4
                )
                value = value[:, None, None].expand(128, 1, 128).to(buffer.dtype)
                expected.append(value)
                buffer.fill_(-11)
                buffer.index_copy_(
                    0, local_rows, value if rank == 0 else torch.full_like(value, -7)
                )
                if with_indexer and pool.index_k_with_scale_buffer[layer].numel():
                    index_buffer = pool.index_k_with_scale_buffer[layer]
                    selected = torch.stack(
                        [
                            torch.full_like(
                                index_buffer[0], layer * 16 + round_id * 2 + part
                            )
                            for part in range(2)
                        ]
                    )
                    expected_indexer[layer] = selected
                    index_buffer.fill_(249)
                    index_buffer.index_copy_(
                        0,
                        local_pages,
                        selected if rank == 0 else torch.full_like(selected, 248),
                    )
            controller.set_mla_producer_stream(torch.cuda.current_stream())
            controller.write_queue.append(
                CacheOperation(host_rows, logical, round_id, pool_transfers=extras)
            )
            controller.start_writing()
            write_ack = controller.ack_write_queue.pop(0)
            assert write_ack.finish_event.query()
            assert bool(write_ack.num_bytes) == (rank == 0)
            for buffer in pool.kv_buffer:
                buffer.fill_(-13)
            if with_indexer:
                for buffer in pool.index_k_with_scale_buffer:
                    buffer.fill_(247)
            host.delay_cycles = (
                400_000_000
                if scenario == "slow_layer"
                else 2_000_000_000
                if rank == 1 and scenario in ("slow_rank", "cancel")
                else 0
            )
            controller.load_fence_stream = torch.cuda.current_stream()
            controller.load_queue.append(
                CacheOperation(host_rows, logical, round_id, pool_transfers=extras)
            )
            slot = controller.start_loading()
            controller.layer_done_counter.set_consumer(slot)
            ack = controller.ack_load_queue[0]
            tree = _load_tree(controller, host, host_rows, round_id)
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
                assert host.available_size() == free_before - 128
                assert round_id in tree.ongoing_load_back
            if scenario == "cancel":
                # Cancelling bypasses the normal ACK/MIN drain. Reset must
                # still finish the slow reader before shared rows are reused.
                controller.reset()
                assert ack.finish_event.query()
                assert not controller.ack_load_queue
                assert not controller.has_inflight_device_transfers()
            snapshots = []
            first_computed = torch.cuda.Event(enable_timing=True)
            for layer in range(3):
                # Exercise the real MLA getter, not just counter.wait_until().
                selected = pool.get_key_buffer(layer).index_select(0, local_rows)
                snapshots.append((selected, expected[layer]))
                model_total = selected.float().sum()
                dist.all_reduce(model_total, group=model_group)
                if layer == 0:
                    first_computed.record()
                if with_indexer and pool.index_k_with_scale_buffer[layer].numel():
                    selected_indexer = pool.get_index_k_with_scale_buffer(
                        layer
                    ).index_select(0, local_pages)
                    snapshots.append((selected_indexer, expected_indexer[layer]))
            torch.cuda.synchronize()
            for actual, value in snapshots:
                assert torch.equal(actual, value), (rank, round_id, scenario)
            if scenario == "slow_layer":
                # The measured first-layer compute must precede the last DMA.
                # Both events use the same device clock, across real streams.
                assert first_computed.elapsed_time(ack.finish_event) > 0
            if scenario == "cancel":
                host.free(host_rows)
            else:
                tree.loading_check()
                assert not controller.ack_load_queue
                assert not tree.ongoing_load_back
            assert host.available_size() == free_before
            outcomes.append(scenario)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "shared_cpu_tp2": "passed",
                        "with_indexer": with_indexer,
                        "rounds": outcomes,
                    }
                ),
                flush=True,
            )
    finally:
        if controller is not None:
            controller.drain_shared_transfers(synchronize_peers=False)
        torch.cuda.synchronize()
        for host_pool in reversed(host_pools):
            host_pool.destroy()
        dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA GPUs")
class TestHiCacheSharedCPUCUDA(CustomTestCase):
    def test_mla_shared_dma_native_layer_wait_and_reader_retirement(self):
        self._run(False)

    def test_dsa_shared_dma_preserves_empty_indexer_and_page_order(self):
        self._run(True)

    def _run(self, with_indexer):
        with tempfile.TemporaryDirectory() as directory:
            processes = mp.spawn(
                _run_shared_rank,
                args=(f"file://{directory}/group", with_indexer),
                nprocs=2,
                join=False,
            )
            try:
                deadline = time.monotonic() + 100
                while not processes.join(timeout=max(0, deadline - time.monotonic())):
                    if time.monotonic() >= deadline:
                        self.fail("Shared-host TP2 test exceeded 100 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
