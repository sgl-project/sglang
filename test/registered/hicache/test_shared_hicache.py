import argparse
import asyncio
import inspect
import json
import threading
import time
import unittest
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.openai.utils import cached_tokens_details_from_dict
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.shared_hicache.manager import (
    SharedHiCacheManager,
    SharedHiCachePlan,
    _SharedHiCachePendingFetch,
)
from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
)
from sglang.srt.mem_cache.shared_hicache.source import (
    ResolvedHostPage,
    resolve_host_pages,
)
from sglang.srt.mem_cache.shared_hicache.transfer import (
    MooncakeSharedHiCacheTransferBackend,
    _get_or_init_mooncake_transfer_engine,
    make_shared_hicache_transfer_backend,
)
from sglang.srt.mem_cache.utils import block_hash_aliases, hash_str_to_int64
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.server_args import ServerArgs


def _make_plan(block_hashes, **overrides):
    plan = {
        "plan_id": "plan-1",
        "request_id": "request-1",
        "target_worker_id": 42,
        "target_dp_rank": 0,
        "source_worker_id": 7,
        "source_dp_rank": 0,
        "source_endpoint": "127.0.0.1:39007",
        "source_medium": StorageMedium.CPU.value,
        "block_hashes": block_hashes,
        "planned_prefix_blocks": len(block_hashes),
        "block_size_tokens": 2,
        "created_at_ms": 1,
        "expires_at_ms": int(time.time() * 1000) + 60_000,
    }
    plan.update(overrides)
    return plan


def _completed_future(result):
    future = Future()
    future.set_result(result)
    return future


class FakeDeviceAllocator:
    def __init__(self):
        self.next_index = 200
        self.freed = []

    def alloc(self, need_size):
        indices = torch.arange(self.next_index, self.next_index + need_size)
        self.next_index += need_size
        return indices

    def free(self, indices):
        self.freed.extend(int(idx) for idx in indices)
        return len(indices)


class FakeHostPool:
    def __init__(self):
        self.pages = {}

    def get_data_page(self, index, flat=True):
        return self.pages[int(index)]


class FakeTree:
    def __init__(self, page_size=2):
        self.page_size = page_size
        self.is_eagle = False
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(array("q"))
        self.device_allocator = FakeDeviceAllocator()
        self.cache_controller = SimpleNamespace(
            mem_pool_host=FakeHostPool(),
            mem_pool_device_allocator=self.device_allocator,
        )
        self.insert_prefix_len = 0
        self.insert_calls = []
        self.locked_nodes = []
        self.unlocked_nodes = []

    def insert_shared_hicache_device_blocks(self, *, key, value):
        self.insert_calls.append((key, value.clone()))
        return SimpleNamespace(prefix_len=self.insert_prefix_len)

    def insert(self, params):
        self.insert_calls.append((params.key, params.value.clone()))
        return SimpleNamespace(prefix_len=self.insert_prefix_len)

    def evict(self, params):
        return SimpleNamespace(num_tokens_evicted=0)

    def inc_lock_ref(self, node):
        self.locked_nodes.append(node)
        node.lock_ref += 1
        return SimpleNamespace(delta=0)

    def dec_lock_ref(self, node):
        self.unlocked_nodes.append(node)
        node.lock_ref -= 1
        return SimpleNamespace(delta=0)


class FakeDirectTransfer:
    name = "mooncake"
    enabled = True
    target_session_id = "target-session"
    target_kv_ptrs = [1]
    target_kv_item_lens = [64]

    def target_descriptor(self):
        return {"backend": self.name, "session_id": self.target_session_id}


class FakeMooncakeEngine:
    def __init__(self, ib_device=None, protocol="rdma", register_returns=None):
        self.ib_device = ib_device
        self.protocol = protocol
        self.register_returns = list(register_returns or [])
        self.registered = []
        self.transfers = []

    def get_session_id(self):
        return "target-session"

    def get_transport_info(self):
        return {
            "protocol": self.protocol,
            "ib_device": self.ib_device,
            "path_hint": (
                "explicit_ib_device"
                if self.protocol == "rdma" and self.ib_device is not None
                else "no_explicit_ib_device"
                if self.protocol == "rdma"
                else self.protocol
            ),
        }

    def register_regions_checked(self, ptrs, lengths, prefer_scalar=True):
        checked = [(int(ptr), int(length)) for ptr, length in zip(ptrs, lengths)]
        self.registered.extend(checked)
        for index, _ in enumerate(checked):
            if self.register_returns:
                ret = self.register_returns.pop(0)
                if ret != 0:
                    return False, f"register_memory_failed:index={index}:ret={ret}"
        return True, "ok"

    def batch_transfer_sync(self, session_id, src_addrs, dst_addrs, lengths):
        self.transfers.append((session_id, src_addrs, dst_addrs, lengths))
        return 0


class FakePrometheusMetric:
    def __init__(self):
        self.calls = []
        self._labels = None

    def labels(self, **labels):
        self._labels = dict(labels)
        return self

    def inc(self, value=1):
        self.calls.append(("inc", self._labels, value))

    def observe(self, value):
        self.calls.append(("observe", self._labels, value))

    def set(self, value):
        self.calls.append(("set", self._labels, value))


class FakeTokenizerManager:
    def __init__(self):
        self.requests = []

    def generate_request(self, obj, _):
        self.requests.append(obj)

        async def _gen():
            yield {"ok": True}

        return _gen()


def _make_manager(tree=None):
    tree = tree or FakeTree()
    manager = SharedHiCacheManager.__new__(SharedHiCacheManager)
    manager.tree_cache = tree
    manager.worker_id = 42
    manager.dp_rank = 0
    manager.timeout_secs = 1.0
    manager.prefetch_stop_policy = "timeout"
    manager.prefetch_timeout_config = None
    manager.direct_transfer = FakeDirectTransfer()
    manager.metrics_collector = None
    manager._fetch_executor = ThreadPoolExecutor(max_workers=1)
    manager._fetch_semaphore = threading.BoundedSemaphore(1)
    manager._pending_fetches = {}
    manager._detached_fetches = set()
    manager._finished_plan_keys = set()
    manager._quarantined_device_indices = []
    manager._quarantined_tokens_by_backend = {}
    manager._source_activity_lock = threading.Lock()
    manager._active_source_resolver_ops = 0
    return manager


def _make_req(plan):
    return SimpleNamespace(
        rid="rid-1",
        fill_ids=array("q", [1, 2, 3, 4, 5]),
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        host_hit_length=0,
        storage_hit_length=0,
        shared_hicache_hit_length=0,
        return_logprob=False,
        logprob_start_len=-1,
        positional_embed_overrides=None,
        extra_key=None,
        shared_hicache_plan=plan,
        last_node=None,
    )


class TestSharedHiCache(unittest.TestCase):
    def test_plan_requires_cpu_source_medium_and_integer_fields(self):
        plan = SharedHiCachePlan.from_dict(_make_plan([11, 22]))

        self.assertEqual(plan.source_medium, StorageMedium.CPU.value)
        self.assertTrue(plan.is_shared_hicache())
        self.assertEqual(plan.source_endpoint, "http://127.0.0.1:39007")

        bad = _make_plan([11], source_medium=StorageMedium.DISK.value)
        with self.assertRaisesRegex(ValueError, "source_medium must be"):
            SharedHiCachePlan.from_dict(bad)

        missing = _make_plan([11])
        del missing["source_medium"]
        with self.assertRaisesRegex(ValueError, "missing source_medium"):
            SharedHiCachePlan.from_dict(missing)

        not_int = _make_plan([11], target_worker_id=True)
        with self.assertRaisesRegex(ValueError, "target_worker_id must be an integer"):
            SharedHiCachePlan.from_dict(not_int)

    def test_source_resolves_protected_hicache_host_pages(self):
        kv_hash = hash_str_to_int64("aa" * 32)
        identity_hash = 123
        node = TreeNode()
        node.host_value = torch.tensor([100, 102], dtype=torch.int64)
        node.hash_value = ["aa" * 32, "bb" * 32]
        tree = FakeTree(page_size=2)
        tree.cache_controller.mem_pool_host.pages[100] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8
        )
        calls = []

        def lookup(wanted_hashes, *, protect=False):
            calls.append((set(wanted_hashes), protect))
            self.assertTrue(protect)
            node.protect_host()
            return {kv_hash: (node, 0, "aa" * 32)}, [node]

        tree.lookup_hicache_host_blocks = lookup
        plan = SharedHiCachePlan.from_dict(
            _make_plan([identity_hash], kv_block_hashes=[kv_hash])
        )

        pages, reason = resolve_host_pages(
            tree, plan, start_block=0, max_blocks=1, worker_id=7, dp_rank=0
        )

        self.assertEqual(reason, "ok")
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].block_hash, identity_hash)
        self.assertEqual(pages[0].data, bytes([1, 2, 3, 4]))
        self.assertEqual(calls, [({kv_hash}, True)])
        self.assertEqual(node.host_ref_counter, 0)

    def test_hiradix_host_index_tracks_aliases_and_stale_entries(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.hicache_host_index_lock = threading.RLock()
        cache.hicache_host_block_index = {}
        node = TreeNode()
        node.host_value = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        node.hash_value = ["aa" * 32, "bb" * 32]

        cache._index_hicache_host_node(node)

        first_hash = hash_str_to_int64("aa" * 32)
        matches, protected = cache.lookup_hicache_host_blocks(
            {first_hash}, protect=True
        )
        for alias in block_hash_aliases(first_hash):
            self.assertIn(alias, matches)
        self.assertEqual(protected, [node])
        self.assertEqual(node.host_ref_counter, 1)
        node.release_host()

        node.hash_value[0] = "cc" * 32
        self.assertEqual(cache.lookup_hicache_host_blocks({first_hash}), {})

        parent = TreeNode()
        parent.value = torch.tensor([9], dtype=torch.int64)
        parent.children = {}
        evicted = TreeNode()
        evicted.parent = parent
        evicted.key = RadixKey(array("q", [5, 6]))
        evicted.host_value = torch.tensor([10, 11], dtype=torch.int64)
        evicted.hash_value = ["dd" * 32]
        parent.children[evicted.key.child_key(cache.page_size)] = evicted
        cache.root_node = TreeNode()
        cache.evictable_host_leaves = [evicted]
        cache.eviction_strategy = SimpleNamespace(get_priority=lambda _: 0)
        cache.cache_controller = SimpleNamespace(
            evict_host=lambda host_value: len(host_value)
        )
        cache._record_remove_event = lambda *_, **__: None
        cache._update_host_leaf_status = lambda *_: None
        cache._update_leaf_status = lambda *_: None

        evicted_hash = hash_str_to_int64("dd" * 32)
        cache._index_hicache_host_node(evicted)
        self.assertNotEqual(cache.lookup_hicache_host_blocks({evicted_hash}), {})

        cache.evict_host(2)
        self.assertEqual(cache.lookup_hicache_host_blocks({evicted_hash}), {})

    def test_manager_stages_direct_transfer_into_local_prefix(self):
        manager = _make_manager()
        plan = _make_plan([11, 22])
        req = _make_req(plan)

        def request_source_transfer(**kwargs):
            self.assertEqual(kwargs["start_block"], 0)
            self.assertEqual(kwargs["target_page_indices"], [100, 101])
            return [
                ResolvedHostPage(11, "", b""),
                ResolvedHostPage(22, "", b""),
            ], "ok"

        manager._request_source_transfer = request_source_transfer
        try:
            first = manager.check_shared_prefix(req)
            self.assertTrue(first.pending)
            pending = manager._pending_fetches[req.rid]
            pending.future.result(timeout=1)

            second = manager.check_shared_prefix(req)
            self.assertFalse(second.pending)
            self.assertEqual(second.staged_tokens, 4)
            self.assertEqual(req.shared_hicache_hit_length, 4)
            self.assertEqual(len(manager.tree_cache.insert_calls), 1)
        finally:
            manager._fetch_executor.shutdown(wait=False, cancel_futures=True)

    def test_manager_quarantines_indeterminate_target_pages(self):
        tree = FakeTree()
        manager = _make_manager(tree)
        plan = SharedHiCachePlan.from_dict(_make_plan([11, 22]))
        req = _make_req(plan.to_dict())
        device_indices = torch.arange(200, 204)
        pending = _SharedHiCachePendingFetch(
            plan=plan,
            plan_offset=0,
            target_start_block=0,
            expected_hashes=plan.planned_hashes,
            future=_completed_future(([], SHARED_HICACHE_DIRECT_TIMEOUT_REASON)),
            device_indices=device_indices,
            backend="mooncake",
            submitted_at=time.perf_counter(),
        )
        manager._pending_fetches[req.rid] = pending

        result = manager.check_shared_prefix(req)

        self.assertEqual(result.staged_tokens, 0)
        self.assertEqual(tree.device_allocator.freed, [])
        self.assertEqual(manager._quarantined_device_indices, [device_indices])

    def test_mooncake_backend_registers_source_and_transfers_pages(self):
        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        host_buffer = torch.zeros((64,), dtype=torch.uint8)
        item_len = source_k[0].nbytes * page_size
        engine = FakeMooncakeEngine(ib_device="mlx5_0")
        tree = SimpleNamespace(
            page_size=page_size,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    kv_buffer=host_buffer,
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )

        backend = MooncakeSharedHiCacheTransferBackend(
            engine=engine,
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_item_lens=[item_len, item_len],
            transfer_parallelism=1,
        )
        backend._register_source_host_pool()
        backend.transfer_pages(
            target_session_id="peer",
            source_page_indices=torch.tensor([1, 2], dtype=torch.int32).numpy(),
            target_page_indices=torch.tensor([3, 4], dtype=torch.int32).numpy(),
            target_kv_ptrs=backend.target_kv_ptrs,
            target_kv_item_lens=backend.target_kv_item_lens,
        )

        self.assertTrue(backend.enabled)
        self.assertIn((int(host_buffer.data_ptr()), int(host_buffer.nbytes)), engine.registered)
        self.assertEqual(backend.target_descriptor()["transport"]["path_hint"], "explicit_ib_device")
        self.assertEqual(len(engine.transfers), 1)
        session_id, src_addrs, dst_addrs, lengths = engine.transfers[0]
        self.assertEqual(session_id, "peer")
        self.assertEqual(lengths, [item_len * 2, item_len * 2])
        self.assertEqual(src_addrs, [int(source_k.data_ptr()) + item_len, int(source_v.data_ptr()) + item_len])
        self.assertEqual(dst_addrs, [1_000_000 + item_len * 3, 2_000_000 + item_len * 3])

    def test_mooncake_from_scheduler_fails_fast_on_checked_registration_error(self):
        engine = FakeMooncakeEngine(register_returns=[0, -202])
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                shared_hicache_config={"transfer_backend": "mooncake"},
                mooncake_ib_device=None,
                tp_size=1,
                pp_size=1,
                attn_cp_size=1,
            ),
            gpu_id=0,
            tree_cache=SimpleNamespace(page_size=2),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(
                    get_contiguous_buf_infos=lambda: ([1, 2], [128, 128], [8, 8])
                )
            ),
        )

        with patch(
            "sglang.srt.mem_cache.shared_hicache.transfer._get_or_init_mooncake_transfer_engine",
            return_value=engine,
        ):
            backend = MooncakeSharedHiCacheTransferBackend.from_scheduler(scheduler)

        self.assertIsNone(backend)
        self.assertEqual(engine.registered, [(1, 128), (2, 128)])

    def test_mooncake_engine_init_uses_scheduler_parallel_state_gpu_id(self):
        calls = []
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(gpu_id=3),
            server_args=SimpleNamespace(mooncake_ib_device="mlx5_0"),
        )

        with (
            patch(
                "sglang.srt.distributed.device_communicators.mooncake_transfer_engine.get_mooncake_transfer_engine",
                return_value=None,
            ),
            patch(
                "sglang.srt.distributed.device_communicators.mooncake_transfer_engine.init_mooncake_transfer_engine",
                side_effect=lambda ip, gpu_id, ib_device: calls.append(
                    (ip, gpu_id, ib_device)
                )
                or "engine",
            ),
            patch(
                "sglang.srt.utils.network.get_local_ip_auto",
                return_value="127.0.0.1",
            ),
        ):
            engine = _get_or_init_mooncake_transfer_engine(scheduler)

        self.assertEqual(engine, "engine")
        self.assertEqual(calls, [("127.0.0.1", 3, "mlx5_0")])

    def test_make_transfer_backend_rejects_unsupported_topology_when_requested(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                shared_hicache_config={"transfer_backend": "mooncake"},
                tp_size=2,
                pp_size=1,
                attn_cp_size=1,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "tp_size=1"):
            make_shared_hicache_transfer_backend(scheduler)

    def test_server_args_enable_shared_hicache_requires_hicache_and_worker_id(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        missing_hicache = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-shared-hicache",
                "--shared-hicache-worker-id",
                "7",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--enable-hierarchical-cache"):
            ServerArgs.from_cli_args(missing_hicache)

        missing = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-hierarchical-cache",
                "--enable-shared-hicache",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--shared-hicache-worker-id"):
            ServerArgs.from_cli_args(missing)

        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-hierarchical-cache",
                "--enable-shared-hicache",
                "--shared-hicache-worker-id",
                "7",
                "--shared-hicache-config",
                json.dumps(
                    {
                        "control": {"endpoint": "127.0.0.1:39007"},
                        "transfer_backend": "mooncake",
                        "timeout_secs": 2.5,
                    }
                ),
            ]
        )
        server_args = ServerArgs.from_cli_args(args)

        self.assertTrue(server_args.enable_shared_hicache)
        self.assertTrue(server_args.enable_hierarchical_cache)
        self.assertEqual(server_args.shared_hicache_worker_id, 7)
        self.assertEqual(server_args.shared_hicache_config["worker_id"], 7)
        self.assertEqual(
            server_args.shared_hicache_config["control_endpoint"],
            "127.0.0.1:39007",
        )
        self.assertEqual(server_args.shared_hicache_config["transfer_backend"], "mooncake")
        self.assertEqual(server_args.shared_hicache_config["timeout_secs"], 2.5)

    def test_server_args_rejects_static_peer_config_and_unknown_backend(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        static_peers = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-hierarchical-cache",
                "--shared-hicache-worker-id",
                "7",
                "--shared-hicache-config",
                json.dumps({"peer_endpoints": {"8": "127.0.0.1:39008"}}),
            ]
        )
        with self.assertRaisesRegex(ValueError, "static endpoint maps"):
            ServerArgs.from_cli_args(static_peers)

        bad_backend = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-hierarchical-cache",
                "--shared-hicache-worker-id",
                "7",
                "--shared-hicache-config",
                json.dumps({"transfer_backend": "nixl"}),
            ]
        )
        with self.assertRaisesRegex(ValueError, "transfer_backend"):
            ServerArgs.from_cli_args(bad_backend)

    def test_cached_tokens_details_and_metrics_use_shared_hicache_names(self):
        req = SimpleNamespace(
            cached_tokens=12,
            cached_tokens_device=8,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            cached_tokens_shared_hicache=4,
        )
        streamer = SimpleNamespace(enable_hicache_storage=lambda: False)

        details = SchedulerOutputStreamer.get_cached_tokens_details(streamer, req)
        self.assertEqual(details, {"device": 8, "host": 0, "shared_hicache": 4})
        self.assertEqual(
            cached_tokens_details_from_dict(details).model_dump(exclude_none=True),
            details,
        )

        collector = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        collector.labels = {"model_name": "dummy"}
        collector.shared_hicache_requests_total = FakePrometheusMetric()
        collector.shared_hicache_tokens_total = FakePrometheusMetric()
        collector.shared_hicache_wait_seconds = FakePrometheusMetric()
        collector.shared_hicache_insert_seconds = FakePrometheusMetric()
        collector.shared_hicache_transfer_bytes_total = FakePrometheusMetric()
        collector.observe_shared_hicache(
            backend="mooncake",
            outcome="hit",
            reason="ok",
            tokens=4,
            wait_ms=12.5,
            insert_ms=0.25,
            transfer_bytes=2 * 1024 * 1024,
        )

        labels = {
            "model_name": "dummy",
            "backend": "mooncake",
            "outcome": "hit",
            "reason": "ok",
        }
        self.assertEqual(collector.shared_hicache_requests_total.calls, [("inc", labels, 1)])
        self.assertEqual(collector.shared_hicache_tokens_total.calls, [("inc", labels, 4)])
        self.assertEqual(
            collector.shared_hicache_wait_seconds.calls,
            [("observe", labels, 0.0125)],
        )
        self.assertEqual(
            collector.shared_hicache_insert_seconds.calls,
            [("observe", labels, 0.00025)],
        )
        self.assertEqual(
            collector.shared_hicache_transfer_bytes_total.calls,
            [("inc", labels, 2 * 1024 * 1024)],
        )

    def test_generate_req_batch_preserves_shared_hicache_plan_metadata(self):
        plan_0 = _make_plan([1])
        plan_1 = _make_plan([2], plan_id="plan-2")
        req = GenerateReqInput(
            text=["hello", "world"],
            sampling_params=[{}, {}],
            rid=["r0", "r1"],
            shared_hicache_plan=[plan_0, plan_1],
        )

        req.normalize_batch_and_arguments()

        self.assertEqual(req[0].shared_hicache_plan, plan_0)
        self.assertEqual(req[1].shared_hicache_plan, plan_1)

    def test_engine_async_generate_forwards_shared_hicache_plan(self):
        plan = _make_plan([11])
        engine = object.__new__(Engine)
        engine.server_args = SimpleNamespace(dp_size=1)
        engine.tokenizer_manager = FakeTokenizerManager()

        self.assertIn(
            "shared_hicache_plan",
            inspect.signature(Engine.async_generate).parameters,
        )
        result = asyncio.run(
            engine.async_generate(
                input_ids=[1, 2],
                sampling_params={"max_new_tokens": 1},
                shared_hicache_plan=plan,
            )
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(engine.tokenizer_manager.requests[0].shared_hicache_plan, plan)


if __name__ == "__main__":
    unittest.main()
