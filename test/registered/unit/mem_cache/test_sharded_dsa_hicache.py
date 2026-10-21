"""CPU contracts for DSA/MTP on the native page-sharded HiCache stack.

Native host declarations, layer mappings, allocation and L2 resolution run here.
Host payloads and CUDA copies are replaced; GPU kernels require CUDA CI.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import prepare_host_pool_config
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    CacheOperation,
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine, TransferCompletion
from sglang.srt.mem_cache.page_interleave import PageInterleavePlacement, PageShardSpec
from sglang.srt.mem_cache.page_interleave_pool import PageInterleaveDSATokenToKVPool
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.mem_cache.pool_host import dsa as dsa_host
from sglang.srt.mem_cache.pool_host.mla import MLAHostAllocation, MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.page_interleave import page_interleave_host_pool_cls
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PS = 64
TOKENS = 8 * PS


def device_pool(rank, shards, *, draft=False):
    pool = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
    pool.layer_num = 1 if draft else 3
    pool.start_layer = 3 if draft else 0
    pool.end_layer = pool.start_layer + pool.layer_num
    pool.size = TOKENS
    pool.page_size = pool.index_page_size = PS
    pool.dtype = pool.store_dtype = torch.uint8
    pool.kv_lora_rank, pool.qk_rope_head_dim = 512, 64
    pool.kv_cache_dim, pool.index_head_dim = 576, 128
    pool.layer_shard_enabled = False
    pool.skip_topk_layers = [False] if draft else [False, True, False]
    pool.index_key_cache = SimpleNamespace(buffer=[object()] * pool.layer_num)
    pool.shard_rank, pool.shard_size = rank, shards
    pool.placement = PageInterleavePlacement(PageShardSpec(rank, shards, PS, 512, 256))
    return pool


def native_group(rank=0, shards=4, *, mtp=True, layout="layer_first"):
    target = device_pool(rank, shards)
    drafts = (device_pool(rank, shards, draft=True),) if mtp else ()
    config = prepare_host_pool_config(
        decls=target.host_pool_decls(),
        full_layer_mapping={0: 0, 1: 1, 2: 2},
        transfer_layer_id_max=3,
        transfer_page_size=PS,
        packed_draft_device_pools=drafts,
    )
    host_type = page_interleave_host_pool_cls(MLATokenToKVPoolHost)
    anchor = host_type(
        MLAHostAllocation(
            dtype=torch.uint8,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            kv_cache_dim=576,
            size_per_token=576 * (3 + len(drafts)),
            draft_layer_num=len(drafts),
        ),
        placement=target.placement,
        host_to_device_ratio=2,
        host_size=0,
        page_size=PS,
        layout=layout,
        pin_memory=False,
        device_capacity_tokens=TOKENS,
        start_layer=0,
        end_layer=3,
        target_layer_num=3,
    )
    indexer_type = dsa_host.DSAIndexerPoolHost

    def dummy_indexer(*args, **kwargs):
        return indexer_type(*args, **kwargs, is_dummy=True, pin_memory=False)

    with (
        patch.object(dsa_host, "DSAIndexerPoolHost", dummy_indexer),
        patch.object(assembler, "_get_allocator_type", return_value="default"),
    ):
        entries = assembler._build_pool_entries(config=config, root_host_pool=anchor)
    return HostPoolGroup(entries), target, drafts


def controller(group):
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.mem_pool_host = group
    cc.device, cc.io_backend, cc.page_size = "cpu", "kernel", PS
    cc.transfer_layer_id_max = 3
    cc.mem_pool_device_allocator = PageInterleavePoolAllocator(
        TOKENS,
        PS,
        group.anchor_entry.host_pool.shard_size,
        torch.uint8,
        "cpu",
        group.anchor_entry.device_pool,
        False,
    )
    cc.load_queue = []
    cc.move_indices = lambda h, d: (h, d)
    return cc


def plan(cc, pages=7, start=2):
    device = cc.mem_pool_device_allocator.alloc_cyclic(start, pages * PS)
    host, extras = cc.allocate_host_transfers(
        device,
        [PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)],
        start_class=start,
    )
    return host, device, extras


class TestShardedDSAHiCache(unittest.TestCase):
    def test_native_host_capacity_and_packed_layer_maps(self):
        for mtp in (False, True):
            for layout in ("layer_first", "page_first"):
                with self.subTest(mtp=mtp, layout=layout):
                    group, _, _ = native_group(mtp=mtp, layout=layout)
                    kv, index = group.entries
                    self.assertEqual(group.available_size(), 4 * 2 * TOKENS)
                    self.assertEqual(kv.host_pool.size, index.host_pool.size)
                    self.assertEqual(kv.host_pool.layer_num, 3 + int(mtp))
                    self.assertEqual(index.host_pool.layer_num, 2 + int(mtp))
                    self.assertIsNone(index.layer_mapper(1))
                    self.assertEqual(index.layer_mapper(2), 2)
                    self.assertEqual(index.host_pool._host_layer_index(2), 1)
                    self.assertEqual(index.layer_mapper(3), 3 if mtp else None)
                    if mtp:
                        self.assertEqual(index.host_pool._draft_host_layer(3), 2)

    def test_queued_and_direct_l2_paths_localize_all_pools_once(self):
        for shards in (2, 4, 8):
            for rank in range(shards):
                with self.subTest(shards=shards, rank=rank):
                    group, _, drafts = native_group(rank, shards)
                    cc = controller(group)
                    host, device, extras = plan(cc)
                    op = CacheOperation(host, device, node_id=7, pool_transfers=extras)
                    for moved in (
                        cc._move_op_indices(op),
                        cc._move_write_operation(op),
                    ):
                        self.assertIs(moved[0], host)
                        self.assertIs(moved[1], device)
                    placement = group.anchor_entry.host_pool.placement
                    mask = placement.owner_of(host) == rank
                    expect_h = placement.local_index(host[mask])
                    expect_d = placement.local_index(device[mask])
                    for transfers, expected_count in (
                        (cc._l2_transfers(host, device, extras), 2),
                        (cc._l2_load_transfers(host, device, extras), 4),
                    ):
                        self.assertEqual(
                            len(transfers), expected_count if mask.any() else 0
                        )
                        for transfer in transfers:
                            self.assertTrue(
                                torch.equal(transfer.host_indices, expect_h)
                            )
                            self.assertTrue(
                                torch.equal(transfer.device_indices, expect_d)
                            )
                            if transfer.is_draft:
                                self.assertIs(transfer.device_pool, drafts[0])
                    self.assertTrue(torch.equal(extras[0].host_indices, host))
                    self.assertTrue(torch.equal(extras[0].device_indices, device))

    def test_load_reallocates_with_same_owner_sequence(self):
        group, _, _ = native_group()
        cc = controller(group)
        host, device, _ = plan(cc)
        cc.mem_pool_device_allocator.free(device)
        cc.mem_pool_device_allocator.alloc_cyclic(2, PS)
        reloaded = cc.load(
            host,
            extra_pools=[
                PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)
            ],
        )
        self.assertIsNotNone(reloaded)
        self.assertFalse(torch.equal(reloaded, device))
        self.assertTrue(torch.equal(host // PS % 4, reloaded // PS % 4))
        self.assertEqual(len(cc.load_queue), 1)
        for transfer in cc._l2_load_transfers(*cc._move_op_indices(cc.load_queue[0])):
            self.assertTrue(
                torch.equal(
                    transfer.device_indices,
                    reloaded[reloaded // PS % 4 == 0] // (4 * PS) * PS
                    + reloaded[reloaded // PS % 4 == 0] % PS,
                )
            )

    def test_layer_ready_follows_target_indexer_and_packed_draft(self):
        for rank in (0, 3):
            with self.subTest(rank=rank):
                group, _, _ = native_group(rank)
                cc = controller(group)
                host, device, extras = plan(cc, pages=1, start=0)
                events = []
                for entry in group.entries:
                    entry.host_pool.load_to_device_per_layer_physical = (
                        lambda pool, h, d, layer, backend, *, is_draft, name=entry.name: (
                            events.append((name.value, layer, is_draft))
                        )
                    )
                engine = L2TransferEngine.__new__(L2TransferEngine)
                engine.io_backend, engine.host_to_device_stream = "kernel", object()
                completion = TransferCompletion(Mock(), Mock(), False)

                @contextmanager
                def submission(transfers, *args):
                    yield transfers, completion

                engine._submission = submission
                result = engine.submit_host_to_device(
                    cc._l2_load_transfers(host, device, extras),
                    transfer_layer_id_max=3,
                    on_layer_done=lambda layer: events.append(("ready", layer)),
                )
                self.assertIs(result, completion)
                if rank == 0:
                    self.assertEqual(
                        events[:5],
                        [
                            ("kv", 0, False),
                            ("indexer", 0, False),
                            ("kv", 3, True),
                            ("indexer", 3, True),
                            ("ready", 0),
                        ],
                    )
                    self.assertNotIn(
                        ("indexer", 1, False), events[: events.index(("ready", 1))]
                    )
                else:
                    self.assertEqual(events, [("ready", 0), ("ready", 1), ("ready", 2)])

    def test_draft_gather_binds_same_final_controller_counter(self):
        group, target, drafts = native_group()
        counter = object()
        result = assembler.StackBuildResult(
            host_pool_group=group,
            cache_controller=SimpleNamespace(
                layer_done_counter=counter, transfer_layer_id_max=3
            ),
            component_host_pools={ComponentType.FULL: group.anchor_entry.host_pool},
            pools_desc="KV + INDEXER",
        )
        cache = SimpleNamespace(components={ComponentType.FULL: SimpleNamespace()})
        assembler._apply_stack_result(
            cache, target, SimpleNamespace(mtp_draft_device_pools=drafts), result
        )
        self.assertIs(target.layer_transfer_counter, counter)
        self.assertIs(drafts[0].layer_transfer_counter, counter)

    def test_write_ack_keeps_logical_tokens_and_counts_owned_bytes(self):
        for rank in (0, 3):
            with self.subTest(rank=rank):
                group, _, _ = native_group(rank)
                cc = controller(group)
                host, device, extras = plan(cc, pages=1, start=0)
                cc.write_queue = [
                    CacheOperation(host, device, node_id=7, pool_transfers=extras)
                ]
                cc.ack_write_queue = []
                finish = Mock()
                cc.l2_transfer_engine = SimpleNamespace(
                    submit_device_to_host=Mock(
                        return_value=TransferCompletion(Mock(), finish, False)
                    )
                )
                cc.mem_pool_device_allocator.set_hicache_transfer_done_event = Mock()
                cc.start_writing()
                sent = cc.l2_transfer_engine.submit_device_to_host.call_args.args[0]
                self.assertEqual(len(sent), 2 if rank == 0 else 0)
                self.assertEqual(len(cc.ack_write_queue), 1)
                ack = cc.ack_write_queue[0]
                self.assertEqual(ack.num_tokens, PS)
                self.assertEqual(
                    ack.num_bytes, PS * (576 * 4 + 132 * 3) if rank == 0 else 0
                )
                self.assertIs(ack.finish_event, finish)
                cc.mem_pool_device_allocator.set_hicache_transfer_done_event.assert_called_once()

    def test_load_fences_resolved_indices_before_ack_submission(self):
        for rank in (0, 3):
            with self.subTest(rank=rank):
                group, _, _ = native_group(rank)
                cc = controller(group)
                host, device, extras = plan(cc, pages=1, start=0)
                cc.load_queue = [
                    CacheOperation(host, device, node_id=7, pool_transfers=extras)
                ]
                cc.ack_load_queue = []
                cc.load_fence_stream = None
                events = []
                original = cc._l2_load_transfers

                def resolve(*args):
                    events.append("resolve_indices")
                    return original(*args)

                cc._l2_load_transfers = resolve
                start = Mock()
                start.record.side_effect = lambda: events.append("producer_fence")
                producer = SimpleNamespace(start_event=start, complete=Mock())
                cc.layer_done_counter = SimpleNamespace(
                    update_producer=lambda: 0, events=[producer]
                )
                finish = Mock()

                def submit(transfers, **kwargs):
                    events.append("submit_load")
                    self.assertIs(kwargs["start_event"], start)
                    self.assertEqual(len(transfers), 4 if rank == 0 else 0)
                    for layer in range(kwargs["transfer_layer_id_max"]):
                        kwargs["on_layer_done"](layer)
                    return TransferCompletion(start, finish, False)

                cc.l2_transfer_engine = SimpleNamespace(submit_host_to_device=submit)
                cc.mem_pool_device_allocator.set_hicache_transfer_done_event = Mock()
                self.assertEqual(cc.start_loading(), 0)
                self.assertEqual(
                    events, ["resolve_indices", "producer_fence", "submit_load"]
                )
                self.assertEqual(producer.complete.call_count, 3)
                self.assertEqual(len(cc.ack_load_queue), 1)
                self.assertEqual(cc.ack_load_queue[0].num_tokens, PS)
                self.assertIs(cc.ack_load_queue[0].finish_event, finish)


if __name__ == "__main__":
    unittest.main()
