"""CPU contracts for owner-local L2, native sidecars and completion ordering."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocator.page_interleave import PageInterleavePoolAllocator
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.mem_cache.page_interleave_pool import (
    PageInterleaveDSATokenToKVPool,
    PageInterleaveMHATokenToKVPool,
    PageInterleaveMLATokenToKVPool,
)
from sglang.srt.mem_cache.pool_host import PoolEntry
from sglang.srt.mem_cache.sharded_hicache import (
    ShardedDSAHiCacheController,
    ShardedDSAHostPoolGroup,
    local_transfer_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PS = 4


def rows(pages):
    return (
        torch.as_tensor(pages, dtype=torch.int64)[:, None] * PS + torch.arange(PS)
    ).flatten()


def allocator(n=4, pages=8):
    return PageInterleavePoolAllocator(
        pages * PS, PS, n, torch.uint8, "cpu", None, True
    )


def group(n=4, rank=0, pages=8, draft=True):
    target = SimpleNamespace(shard_rank=rank, shard_size=n, page_size=PS, layer_num=2)
    mtp = SimpleNamespace(shard_rank=rank, shard_size=n, page_size=PS, layer_num=1)
    entries = []
    for name, width in ((PoolName.KV, 576), (PoolName.INDEXER, 132)):
        host = SimpleNamespace(
            layout="layer_first",
            page_size=PS,
            device="cpu",
            dtype=torch.uint8,
            size=(pages + 1) * PS,
            logical_size=(pages + 1) * PS,
            can_use_write_back_jit=False,
            size_per_token=width * (3 if draft else 2),
            clear=Mock(),
            layer_num=3 if draft else 2,
        )
        entries.append(
            PoolEntry(
                name=name,
                host_pool=host,
                device_pool=target,
                layer_mapper=lambda i: i if 0 <= i < (3 if draft else 2) else None,
                is_primary_index_anchor=name == PoolName.KV,
                packed_draft_device_pools=(mtp,) if draft else (),
            )
        )
    return ShardedDSAHostPoolGroup(entries, target)


def controller(n=4, rank=0, pages=8, draft=True):
    cc = ShardedDSAHiCacheController.__new__(ShardedDSAHiCacheController)
    cc.mem_pool_host = group(n, rank, pages, draft)
    cc.mem_pool_device_allocator = allocator(n, pages)
    cc.page_size, cc.device, cc.io_backend = PS, "cpu", "kernel"
    cc.transfer_layer_id_max = 2
    cc.write_queue, cc.load_queue = [], []
    cc.start_writing = Mock()
    return cc


def indexer_transfer():
    return PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.KV)


class TestShardedHiCache(unittest.TestCase):
    def test_only_dsa_accepts_a_layer_transfer_counter(self):
        counter = object()
        dsa = PageInterleaveDSATokenToKVPool.__new__(PageInterleaveDSATokenToKVPool)
        dsa.register_layer_transfer_counter(counter)
        self.assertIs(dsa.layer_transfer_counter, counter)
        dsa.register_layer_transfer_counter(None)
        self.assertIsNone(dsa.layer_transfer_counter)
        for pool_cls in (
            PageInterleaveMLATokenToKVPool,
            PageInterleaveMHATokenToKVPool,
        ):
            with self.subTest(pool=pool_cls.__name__):
                pool = pool_cls.__new__(pool_cls)
                with self.assertRaisesRegex(
                    NotImplementedError, "layer-wise KV load-back"
                ):
                    pool.register_layer_transfer_counter(counter)
                pool.register_layer_transfer_counter(None)
                self.assertIsNone(pool.layer_transfer_counter)

    def test_gather_stream_waits_for_local_layer_before_reading_target_or_draft(self):
        # CPU tensors plus recording stream/communicator doubles check the
        # submission ordering, not real CUDA event or NCCL execution.
        for label, start_layer, layer_num, local_layer in (
            ("target", 5, 2, 1),
            ("draft", 78, 1, 0),
        ):
            with self.subTest(pool=label):
                events = []
                active_stream = [None]
                compute_stream = object()
                gather_stream = SimpleNamespace()

                def wait_stream(stream):
                    self.assertIs(stream, compute_stream)
                    self.assertIsNone(active_stream[0])
                    events.append("wait_compute")

                gather_stream.wait_stream = wait_stream

                @contextmanager
                def stream_scope(stream):
                    self.assertIs(stream, gather_stream)
                    self.assertIsNone(active_stream[0])
                    active_stream[0] = stream
                    events.append("enter_gather")
                    try:
                        yield
                    finally:
                        events.append("exit_gather")
                        active_stream[0] = None

                def assert_gather():
                    self.assertIs(active_stream[0], gather_stream)

                def wait_until(layer):
                    assert_gather()
                    self.assertEqual(layer, local_layer)
                    events.append(("wait_layer", layer))

                @contextmanager
                def change_state(*, enable):
                    self.assertTrue(enable)
                    assert_gather()
                    yield

                def all_gather(out, send):
                    assert_gather()
                    self.assertEqual(out.shape[0], 4 * send.shape[0])
                    events.append(("all_gather", send.shape[1]))

                def record(stream):
                    assert_gather()
                    self.assertIs(stream, gather_stream)
                    events.append("ready_record")

                pool = PageInterleaveDSATokenToKVPool.__new__(
                    PageInterleaveDSATokenToKVPool
                )
                pool.start_layer, pool.layer_num = start_layer, layer_num
                pool.shard_rank, pool.shard_size, pool._epoch = 1, 4, 7
                pool.kv_buffer = [
                    torch.arange(32 * 3).reshape(32, 3) for _ in range(layer_num)
                ]
                pool.index_key_cache = SimpleNamespace(
                    buffer=[
                        torch.arange(8 * 11).reshape(8, 11) for _ in range(layer_num)
                    ]
                )
                pool._send_rows = torch.arange(PS, 2 * PS)
                pool._send_pages = torch.tensor([1])
                pool._slots = [
                    SimpleNamespace(
                        tensors={
                            "kv": torch.zeros((4 * PS, 3), dtype=torch.int64),
                            "index_k": torch.zeros((4, 11), dtype=torch.int64),
                        },
                        ready=SimpleNamespace(record=record),
                        resident_key=None,
                    )
                    for _ in range(2)
                ]
                pool.kv_gather_stream = gather_stream
                pool.device_module = SimpleNamespace(
                    current_stream=lambda: compute_stream, stream=stream_scope
                )
                pool.kv_gather_comm = SimpleNamespace(
                    change_state=change_state, all_gather=all_gather
                )
                pool.register_layer_transfer_counter(
                    SimpleNamespace(wait_until=wait_until)
                )
                index_select = torch.index_select

                def checked_index_select(source, dim, indices, *, out):
                    assert_gather()
                    events.append(("index_select", source.shape[1]))
                    return index_select(source, dim, indices, out=out)

                layer_id = start_layer + local_layer
                with patch("torch.index_select", checked_index_select):
                    pool._prefetch_layer(layer_id)
                    self.assertEqual(
                        events,
                        [
                            "wait_compute",
                            "enter_gather",
                            ("wait_layer", local_layer),
                            ("index_select", 3),
                            ("all_gather", 3),
                            ("index_select", 11),
                            ("all_gather", 11),
                            "ready_record",
                            "exit_gather",
                        ],
                    )
                    self.assertEqual(
                        pool._slots[layer_id % 2].resident_key, (layer_id, 7)
                    )
                    before = list(events)
                    pool._prefetch_layer(layer_id)
                    self.assertEqual(events, before)
                    pool._epoch += 1
                    pool._prefetch_layer(layer_id)
                    self.assertEqual(events, before + before)

    def test_alloc_matching_atomic_failure_and_reuse(self):
        for n in (4, 8):
            with self.subTest(cp=n):
                alloc = allocator(n, pages=2)
                source = rows([n * 9 + 2, n * 4 + 2, n * 3 + 1])
                out = alloc.alloc_matching(source)
                self.assertTrue(torch.equal(out // PS % n, source // PS % n))
                before = [t.clone() for t in alloc.class_free_pages]
                self.assertIsNone(alloc.alloc_matching(rows([n + 2, n + 3])))
                for actual, expected in zip(alloc.class_free_pages, before):
                    self.assertTrue(torch.equal(actual, expected))
                alloc.free(out)
                self.assertEqual(alloc.class_free_page_counts(), [2] * n)
                self.assertIsNotNone(alloc.alloc_matching(source))

    def test_invalid_pages_do_not_consume_capacity(self):
        for source in (
            torch.arange(3),
            torch.arange(1, 5),
            rows([0]),
            torch.tensor([16, 17, 19, 18]),
            torch.zeros((4, 4)),
        ):
            with self.subTest(source=source):
                alloc = allocator()
                with self.assertRaises(ValueError):
                    alloc.alloc_matching(source)
                self.assertEqual(alloc.class_free_page_counts(), [8] * 4)

    def test_fragmented_roundtrip_preserves_rotation_and_each_byte(self):
        generator = torch.Generator().manual_seed(137)
        for n in (4, 8):
            for rotation in range(n):
                with self.subTest(cp=n, rotation=rotation):
                    owners = (torch.arange(2 * n + 1) + rotation) % n
                    source = rows(
                        (torch.randperm(20, generator=generator)[: len(owners)] + 10)
                        * n
                        + owners
                    )
                    payload = torch.arange(len(source) * 19).reshape(-1, 19)
                    visited = torch.zeros(len(source), dtype=torch.int64)
                    for rank in range(n):
                        host_alloc, gpu_alloc = allocator(n, 32), allocator(n, 32)
                        host = host_alloc.alloc_matching(source)
                        gpu_alloc.alloc_matching(rows(range(n, 2 * n)))
                        restored = gpu_alloc.alloc_matching(host)
                        h, d = local_transfer_indices(
                            host, source, page_size=PS, rank=rank, size=n
                        )
                        h2, d2 = local_transfer_indices(
                            host, restored, page_size=PS, rank=rank, size=n
                        )
                        data = torch.zeros((33 * PS, 19), dtype=torch.int64)
                        own = source // PS % n == rank
                        data[h] = payload[own]
                        output = torch.zeros_like(data)
                        output[d2] = data[h2]
                        self.assertTrue(torch.equal(output[d2], payload[own]))
                        self.assertFalse(torch.equal(d, d2))
                        self.assertTrue(
                            torch.equal(restored // PS % n, source // PS % n)
                        )
                        visited += own
                    self.assertTrue(torch.equal(visited, torch.ones_like(visited)))

    def test_capacity_reserves_exactly_one_physical_page(self):
        for n in (4, 8):
            host = group(n=n, pages=7)
            self.assertEqual(host.physical_size, 8 * PS)
            self.assertEqual(host.logical_size, n * 7 * PS)
            self.assertEqual(host.available_size(), n * 7 * PS)
            allocated = host.alloc_matching(rows(range(n, 8 * n)))
            self.assertEqual(host.available_size(), 0)
            self.assertTrue(bool((allocated // (n * PS) < 8).all()))
            host.free(allocated, pool=PoolName.KV)
            self.assertEqual(host.available_size(), host.logical_size)

    def test_mismatched_owner_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "owners"):
            local_transfer_indices(rows([4]), rows([5]), page_size=PS, rank=0, size=4)

    def test_empty_owner_keeps_logical_queue_and_ack_metrics(self):
        cc = controller(rank=3)
        source = rows([4, 5])
        host = cc.write(
            source, node_id=17, extra_pools=[indexer_transfer()], flush=False
        )
        op = cc.write_queue[0]
        self.assertEqual(op.node_ids, [17])
        self.assertEqual(len(op.device_indices), len(source))
        self.assertEqual(cc._num_tokens_by_pool(op), {"kv": len(source)})
        self.assertEqual(cc._transfer_num_bytes(op), 0)
        self.assertEqual(cc._l2_transfers(*cc._move_write_operation(op)), [])
        cc.start_writing.assert_not_called()
        restored = cc.load(host, node_id=17, extra_pools=[indexer_transfer()])
        self.assertEqual(len(restored), len(source))
        load_op = cc.load_queue[0]
        self.assertEqual(cc._l2_load_transfers(*cc._move_op_indices(load_op)), [])

    def test_metrics_count_logical_tokens_and_local_bytes(self):
        cc = controller(rank=1)
        cc.write(rows([4, 5, 9, 6]), extra_pools=[indexer_transfer()], flush=False)
        op = cc.write_queue[0]
        self.assertEqual(cc._num_tokens_by_pool(op), {"kv": 4 * PS})
        self.assertEqual(cc._transfer_num_bytes(op), 2 * PS * (576 + 132) * 3)
        local = cc._l2_transfers(*cc._move_write_operation(op))
        self.assertEqual(len(local), 2)
        self.assertTrue(torch.equal(local[0].host_indices, local[1].host_indices))
        self.assertTrue(torch.equal(local[0].device_indices, local[1].device_indices))
        self.assertEqual(len(op.device_indices), 4 * PS)

    def test_layer_ready_follows_target_and_draft_kv_and_indexer(self):
        cc = controller(rank=1)
        host = cc.write(rows([5, 9]), extra_pools=[indexer_transfer()], flush=False)
        cc.load(host, extra_pools=[indexer_transfer()])
        transfers = cc._l2_load_transfers(*cc._move_op_indices(cc.load_queue[0]))
        events = []
        for name, entry in cc.mem_pool_host.entry_map.items():

            def load(pool, h, d, layer, backend, *, is_draft=False, name=name):
                events.append((name, "draft" if is_draft else "target", layer))

            entry.host_pool.load_to_device_per_layer_physical = load

        @contextmanager
        def submission(transfers, *args):
            yield transfers, None

        engine = L2TransferEngine.__new__(L2TransferEngine)
        engine.host_to_device_stream, engine.io_backend = None, "kernel"
        with patch.object(engine, "_submission", submission):
            engine.submit_host_to_device(
                transfers,
                transfer_layer_id_max=2,
                on_layer_done=lambda layer: events.append(("ready", layer)),
            )
        self.assertEqual(
            events,
            [
                (PoolName.KV, "target", 0),
                (PoolName.INDEXER, "target", 0),
                (PoolName.KV, "draft", 2),
                (PoolName.INDEXER, "draft", 2),
                ("ready", 0),
                (PoolName.KV, "target", 1),
                (PoolName.INDEXER, "target", 1),
                ("ready", 1),
            ],
        )

    def test_empty_owner_still_publishes_each_layer_ready(self):
        events = []

        @contextmanager
        def submission(transfers, *args):
            yield transfers, None

        engine = L2TransferEngine.__new__(L2TransferEngine)
        engine.host_to_device_stream, engine.io_backend = None, "kernel"
        with patch.object(engine, "_submission", submission):
            engine.submit_host_to_device(
                [], transfer_layer_id_max=2, on_layer_done=events.append
            )
        self.assertEqual(events, [0, 1])

    def test_rejects_non_kv_derived_sidecar_before_allocating(self):
        cc = controller()
        before = cc.mem_pool_host.available_size()
        with self.assertRaisesRegex(ValueError, "follow KV"):
            cc.write(rows([4]), extra_pools=[PoolTransfer(name=PoolName.INDEXER)])
        self.assertEqual(cc.mem_pool_host.available_size(), before)

    def test_storage_cannot_be_attached_at_runtime(self):
        with self.assertRaisesRegex(ValueError, "CPU L2"):
            controller().attach_storage_backend(storage_backend="file")

    def test_rejects_storage_and_buffer_only_before_cuda_initialization(self):
        for kwargs in (
            {"storage_backend": "file"},
            {"host_memory_mode": "buffer_only"},
            {"io_backend": "direct"},
        ):
            with self.assertRaises(ValueError):
                ShardedDSAHiCacheController(allocator(), group(), **kwargs)

    def test_packed_draft_registers_the_target_layer_counter_once(self):
        host_group = group()
        draft = host_group.entries[0].packed_draft_device_pools[0]
        draft.register_layer_transfer_counter = Mock()
        counter = object()

        def initialize(cc, *args, **kwargs):
            cc.layer_done_counter = counter

        with patch.object(HybridCacheController, "__init__", initialize):
            ShardedDSAHiCacheController(allocator(), host_group)
        draft.register_layer_transfer_counter.assert_called_once_with(counter)

    def test_strategy_keeps_native_packed_pools_and_target_transfer_range(self):
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler

        host_group = group()
        target = host_group.entries[0].device_pool
        target.host_pool_decls = lambda: ()
        draft = host_group.entries[0].packed_draft_device_pools[0]
        decl = SimpleNamespace(
            is_primary=True, pool_name=PoolName.KV, device_pool=target
        )
        config = SimpleNamespace(
            pools=[SimpleNamespace(decl=decl, packed_draft_device_pools=(draft,))],
            transfer_page_size=PS,
        )
        params = SimpleNamespace(
            page_size=PS,
            mtp_draft_device_pools=(draft,),
            token_to_kv_pool_allocator=allocator(),
            tp_cache_group=None,
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            pp_cache_group=None,
        )
        memory = SimpleNamespace(
            hicache_host_memory_mode="cache",
            hicache_io_backend="kernel",
            hicache_mem_layout="page_first",
            hicache_write_policy="write_back",
        )
        with (
            patch.object(assembler, "prepare_host_pool_config", return_value=config),
            patch.object(
                assembler,
                "build_kv_host_pool",
                return_value=host_group.anchor_entry.host_pool,
            ) as build,
            patch.object(
                assembler, "_build_pool_entries", return_value=host_group.entries
            ),
            patch.object(assembler, "get_memory", return_value=memory),
            patch(
                "sglang.srt.mem_cache.sharded_hicache.ShardedDSAHiCacheController"
            ) as cc,
        ):
            result = assembler._DsaStrategy().build(
                cache=None,
                kvcache=target,
                params=params,
                server_args=None,
                load_cache_event=None,
            )
        self.assertIsInstance(result.host_pool_group, ShardedDSAHostPoolGroup)
        self.assertEqual(build.call_args.kwargs["mtp_draft_device_pools"], (draft,))
        self.assertEqual(cc.call_args.kwargs["transfer_layer_id_max"], 2)
        self.assertEqual(result.host_pool_group.logical_size, 8 * PS * 4)

    def test_full_capacity_mismatch_and_mismatched_draft_owner_are_rejected(self):
        for kind in ("capacity", "owner"):
            with self.subTest(kind=kind):
                good = group()
                if kind == "capacity":
                    good.entries[1].host_pool.size -= PS
                else:
                    good.entries[0].packed_draft_device_pools[0].shard_rank = 2
                with self.assertRaises(ValueError):
                    ShardedDSAHostPoolGroup(good.entries, good.entries[0].device_pool)


if __name__ == "__main__":
    unittest.main()
