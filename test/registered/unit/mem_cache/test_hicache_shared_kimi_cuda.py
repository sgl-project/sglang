"""TP2 real shared MLA/private KDA DMA through the Kimi hybrid builder."""

import json
import tempfile
import threading
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MambaPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import (
    get_context,
    restore_context,
    snapshot_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")


class _DelayedKDAHostPool(MambaPoolHost):
    delay_cycles = 0

    def load_to_device_per_layer_physical(
        self, device_pool, host_indices, device_indices, layer_id, io_backend, **kwargs
    ):
        if layer_id == self.num_mamba_layers - 1 and self.delay_cycles:
            torch.cuda._sleep(self.delay_cycles)
        return super().load_to_device_per_layer_physical(
            device_pool, host_indices, device_indices, layer_id, io_backend, **kwargs
        )


def _load_tree(controller, host, state_host, host_rows, state_rows, node_id):
    tree = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tree.cache_controller = controller
    tree.pp_rank = 0
    tree.buffer_pipeline = None
    tree.metrics_collector = None
    tree.ongoing_load_back = {node_id: (object(), None, None)}
    tree.dec_lock_ref = lambda node, params: None
    tree.dec_host_lock_ref = lambda node, params: None

    def finish(node):
        host.free(host_rows)
        state_host.free(state_rows)

    tree.tree_core = SimpleNamespace(
        write_back_duplicate_reclaim_digest=0, finish_load_back=finish
    )
    tree._all_reduce = lambda values, operation: dist.all_reduce(
        values, op=operation, group=controller.mla_dedup.process_group
    )
    return tree


def _kimi_rank(rank, init_method):
    context_state = snapshot_context()
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
        state_ids = [0, 1, 2, 4, 5, 6]
        state = MambaPool(
            size=8,
            spec_state_size=0,
            cache_params=KimiLinearCacheParams(
                layers=state_ids,
                shape=KimiLinearStateShape.create(
                    tp_world_size=2,
                    num_heads=4,
                    head_dim=16,
                ),
            ),
            mamba_layer_ids=state_ids,
            device="cuda",
        )
        kv = MLATokenToKVPool(
            size=512,
            page_size=64,
            dtype=torch.bfloat16,
            kv_lora_rank=128,
            qk_rope_head_dim=0,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
        )
        kv.host_transfer_translate = lambda indices: (
            ((indices // 64 + rank * 2) % 7) * 64 + indices % 64
        )
        hybrid = HybridLinearKVPool(
            size=512,
            dtype=torch.bfloat16,
            page_size=64,
            head_num=1,
            head_dim=128,
            full_attention_layer_ids=[3, 7],
            device="cuda",
            mamba_pool=state,
            use_mla=True,
            kv_lora_rank=128,
            qk_rope_head_dim=0,
            full_kv_pool=kv,
        )
        allocator = TokenToKVPoolAllocator(
            512, torch.bfloat16, "cuda", hybrid, need_sort=False
        )
        req = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
        req.mamba_pool = state
        req.mamba_allocator = MambaSlotAllocator(8, "cuda")
        req.mamba_map = {layer: index for index, layer in enumerate(state_ids)}
        req.start_layer = 0
        req.layer_transfer_counter = None
        params = SimpleNamespace(
            page_size=64,
            mtp_draft_device_pools=(),
            mtp_draft_mamba_pools=(),
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=req,
            tp_cache_group=cpu_group,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        memory = ServerArgs(
            model_path="dummy",
            device="cuda",
            tp_size=2,
            page_size=64,
            enable_hierarchical_cache=True,
            hicache_ratio=2,
            hicache_size=0,
            hicache_mem_layout="page_first_direct",
            hicache_storage_backend=None,
            enable_mla_hicache_host_dedup=True,
            mla_hicache_host_dedup_mode="shared",
            enable_mla_hicache_layerwise_overlap=False,
            hicache_write_policy="write_through",
            hicache_io_backend="direct",
            hicache_host_memory_mode="cache",
        )
        # Publish the same real memory configuration to the builder and host
        # allocator; patching one imported accessor leaves the allocator unset.
        get_context().set_server_args(memory)
        with mock.patch.object(assembler, "MambaPoolHost", _DelayedKDAHostPool):
            group, controller, _ = assembler.build_hybrid_mamba_stack(
                params=params,
                decls=hybrid.host_pool_decls(),
                mamba_pool=state,
                full_layer_mapping={3: 0, 7: 1},
                start_layer=req.start_layer,
                load_cache_event=threading.Event(),
                storage_backend=None,
                use_mla=True,
            )
        host = group.get_pool(PoolName.KV)
        state_host = group.get_pool(PoolName.MAMBA)
        pools = [host, state_host]
        hybrid.register_layer_transfer_counter(controller.layer_done_counter)
        req.register_layer_transfer_counter(controller.layer_done_counter)
        assert kv.layer_transfer_counter is None
        assert host.allocator.is_shared and host.allocator.is_owner == (rank == 0)
        assert not getattr(state_host.allocator, "is_shared", False)
        assert controller.transfer_layer_id_max == 8
        assert controller.layer_done_counter.on_layer_ready is None

        # Private host slot allocation and GPU physical slots deliberately differ.
        held_private = state_host.alloc(1) if rank == 1 else None
        logical = torch.cat((torch.arange(64, 128), torch.arange(192, 256))).cuda()
        physical = kv.host_transfer_translate(logical)
        source_state = torch.tensor([1 + rank], dtype=torch.int64, device="cuda")
        target_state = torch.tensor([4 + rank], dtype=torch.int64, device="cuda")
        forward = torch.cuda.Stream()
        torch.cuda.synchronize()
        outcomes = []
        for round_id, scenario in enumerate(
            ("normal", "slow_state", "slow_reader", "cancel", "reuse")
        ):
            host_free = host.available_size()
            state_free = state_host.available_size()
            kv_expected = [
                torch.full(
                    (128, 1, 128),
                    11 + layer + round_id,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                for layer in range(2)
            ]
            temporal_expected = (
                state.mamba_cache.temporal[:, :1]
                .clone()
                .fill_(100 + rank * 20 + round_id)
            )
            conv_expected = [
                v[:, :1].clone().fill_(200 + rank * 20 + round_id)
                for v in state.mamba_cache.conv
            ]
            forward.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(forward):
                torch.cuda._sleep(30_000_000)
                for layer, buffer in enumerate(kv.kv_buffer):
                    buffer.index_copy_(
                        0,
                        physical,
                        kv_expected[layer]
                        if rank == 0
                        else torch.full_like(kv_expected[layer], -3),
                    )
                state.mamba_cache.temporal.index_copy_(
                    1, source_state, temporal_expected
                )
                for buffer, value in zip(state.mamba_cache.conv, conv_expected):
                    buffer.index_copy_(1, source_state, value)
            controller.set_mla_producer_stream(forward)
            host_rows = controller.write(
                logical,
                node_id=round_id,
                flush=False,
                extra_pools=[
                    PoolTransfer(name=PoolName.MAMBA, device_indices=source_state)
                ],
            )
            assert host_rows is not None
            state_rows = controller.write_queue[-1].pool_transfers[0].host_indices
            rows_by_rank = [None, None]
            dist.all_gather_object(rows_by_rank, state_rows.tolist(), group=cpu_group)
            assert rows_by_rank[0] != rows_by_rank[1], rows_by_rank
            controller.start_writing()
            ack_write = controller.ack_write_queue.pop(0)
            assert ack_write.finish_event.query()
            assert torch.equal(
                state_host.temporal_buffer.index_select(0, state_rows)
                .squeeze(2)
                .movedim(0, 1),
                temporal_expected.cpu(),
            )
            for value, expected in zip(state_host.conv_buffer, conv_expected):
                assert torch.equal(
                    value.index_select(0, state_rows).squeeze(2).movedim(0, 1),
                    expected.cpu(),
                )

            # A previous forward is still writing the recycled destination slots.
            with torch.cuda.stream(forward):
                torch.cuda._sleep(20_000_000)
                for buffer in kv.kv_buffer:
                    buffer.index_fill_(0, physical, -13)
                state.mamba_cache.temporal.index_fill_(1, target_state, -17)
                for buffer in state.mamba_cache.conv:
                    buffer.index_fill_(1, target_state, -19)
            controller.load_fence_stream = forward
            state_host.delay_cycles = (
                400_000_000
                if scenario == "slow_state"
                else 2_000_000_000
                if rank == 1 and scenario in ("slow_reader", "cancel")
                else 0
            )
            allocator.free_pages = logical.clone()
            loaded = controller.load(
                host_rows,
                node_id=round_id,
                extra_pools=[
                    PoolTransfer(
                        name=PoolName.MAMBA,
                        host_indices=state_rows,
                        device_indices=target_state,
                    )
                ],
            )
            assert torch.equal(loaded, logical)
            slot = controller.start_loading()
            controller.layer_done_counter.set_consumer(slot)
            ack = controller.ack_load_queue[0]
            tree = _load_tree(
                controller, host, state_host, host_rows, state_rows, round_id
            )
            if scenario == "slow_reader":
                if rank == 0:
                    ack.finish_event.synchronize()
                ready = [None, None]
                dist.all_gather_object(ready, ack.finish_event.query(), group=cpu_group)
                assert ready == [True, False], ready
                tree.loading_check()
                assert len(controller.ack_load_queue) == 1
                assert host.available_size() == host_free - 128
                assert state_host.available_size() == state_free - 1
            if scenario == "cancel":
                controller.reset()
                assert ack.finish_event.query()
                assert not controller.has_inflight_device_transfers()

            snapshots = []
            first_compute = torch.cuda.Event(enable_timing=True)
            for layer in range(8):
                if layer in req.mamba_map:
                    cache = req.mamba2_layer_cache(layer)
                    local_layer = req.mamba_map[layer]
                    snapshots.append(
                        (
                            cache.temporal.index_select(0, target_state),
                            temporal_expected[local_layer],
                        )
                    )
                    for value, expected in zip(cache.conv, conv_expected):
                        snapshots.append(
                            (value.index_select(0, target_state), expected[local_layer])
                        )
                    total = snapshots[-1][0].float().sum()
                else:
                    value = hybrid.get_key_buffer(layer).index_select(0, physical)
                    snapshots.append(
                        (
                            value,
                            kv_expected[hybrid.full_attention_layer_id_mapping[layer]],
                        )
                    )
                    total = value.float().sum()
                dist.all_reduce(total, group=model_group)
                if layer == 0:
                    first_compute.record()
            torch.cuda.synchronize()
            for actual, expected in snapshots:
                assert torch.equal(actual, expected), (rank, scenario)
            if scenario == "slow_state":
                assert first_compute.elapsed_time(ack.finish_event) > 0
            if scenario == "cancel":
                host.free(host_rows)
                state_host.free(state_rows)
            else:
                tree.loading_check()
                assert not controller.ack_load_queue and not tree.ongoing_load_back
            assert host.available_size() == host_free
            assert state_host.available_size() == state_free
            outcomes.append(scenario)
        if held_private is not None:
            state_host.free(held_private)
        if rank == 0:
            print(
                json.dumps(
                    {"kimi_shared_mla_private_kda": "passed", "rounds": outcomes}
                ),
                flush=True,
            )
    finally:
        if controller is not None:
            controller.drain_shared_transfers(synchronize_peers=False)
        torch.cuda.synchronize()
        for pool in reversed(pools):
            pool.destroy()
        dist.destroy_process_group()
        restore_context(context_state)


class TestSharedKimiCUDA(CustomTestCase):
    def test_shared_mla_private_kda_native_wait_and_reader_retirement(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Requires two CUDA GPUs")
        with tempfile.TemporaryDirectory(prefix="hicache-shared-kimi-") as root:
            mp.spawn(_kimi_rank, args=(f"file://{root}/gloo",), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
