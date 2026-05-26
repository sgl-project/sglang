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
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.hicache_host_index import HiCacheHostBlockIndex
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.shared_hicache.manager import SharedHiCacheManager
from sglang.srt.mem_cache.shared_hicache.pending import SharedHiCachePendingFetch
from sglang.srt.mem_cache.shared_hicache.config import SharedHiCacheConfig
from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
    SharedHiCachePlan,
)
from sglang.srt.mem_cache.shared_hicache.scheduler_mixin import (
    SharedHiCacheSchedulerMixin,
)
from sglang.srt.mem_cache.shared_hicache.service import format_control_endpoint
from sglang.srt.mem_cache.shared_hicache.source import (
    ResolvedHostPage,
    handle_source_transfer,
    resolve_host_pages,
)
from sglang.srt.mem_cache.shared_hicache.target import SharedHiCacheTarget
from sglang.srt.mem_cache.shared_hicache.transfer import (
    NixlSharedHiCacheTransferBackend,
    SharedHiCacheTransferBackend,
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
        "source_worker_id": 7,
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


class FakeDirectTransfer(SharedHiCacheTransferBackend):
    name = "nixl"

    def __init__(self):
        super().__init__(
            target_session_id="target-session",
            target_kv_ptrs=[1],
            target_kv_item_lens=[64],
        )

    @property
    def enabled(self):
        return True

    def transfer_pages(self, **kwargs):
        pass


class FakeNixlAgent:
    def __init__(self, name="agent", metadata=b"agent-metadata"):
        self.name = name
        self.metadata = metadata
        self.backends = []
        self.registered = []
        self.remote_agents = []
        self.xfer_desc_calls = []
        self.initialized = []
        self.transfers = []
        self.released = []

    def get_plugin_list(self):
        return ["UCX"]

    def create_backend(self, backend, backend_params):
        self.backends.append((backend, dict(backend_params)))

    def register_memory(self, addrs, mem_type=None):
        normalized = [tuple(int(x) if isinstance(x, int) else x for x in item) for item in addrs]
        self.registered.append((mem_type, normalized))
        return [("registered", mem_type, normalized)]

    def get_agent_metadata(self):
        return self.metadata

    def add_remote_agent(self, metadata):
        self.remote_agents.append(metadata)

    def get_xfer_descs(self, reqs, mem_type=None):
        rows = reqs.tolist() if hasattr(reqs, "tolist") else reqs
        self.xfer_desc_calls.append((mem_type, rows))
        return ("descs", mem_type, rows)

    def initialize_xfer(self, *args):
        self.initialized.append(args)
        return f"handle-{len(self.initialized)}"

    def transfer(self, handle):
        self.transfers.append(handle)
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released.append(handle)


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


class FakeSharedHiCacheReq:
    def __init__(self, rid, shared_hicache_plan=True):
        self.rid = rid
        self.shared_hicache_plan = shared_hicache_plan
        self.shared_hicache_max_prefix_len = None
        self.init_calls = []

    def init_next_round_input(self, tree_cache=None, cow_mamba=None):
        self.init_calls.append(
            {
                "tree_cache": tree_cache,
                "cow_mamba": cow_mamba,
                "max_prefix_len": self.shared_hicache_max_prefix_len,
            }
        )


class FakeSharedHiCacheScheduler(SharedHiCacheSchedulerMixin):
    def __init__(
        self,
        manager,
        tp_size=2,
        *,
        allocatable_reqs=2,
        prefill_max_requests=None,
    ):
        self.shared_hicache_manager = manager
        self.ps = SimpleNamespace(tp_size=tp_size)
        self.tp_cpu_group = object()
        self.tree_cache = object()
        self.server_args = SimpleNamespace(prefill_max_requests=prefill_max_requests)
        self.allocatable_reqs = allocatable_reqs
        self.chunked_req = None
        self.enable_priority_preemption = False

    def get_num_allocatable_reqs(self, running_bs):
        return self.allocatable_reqs


class FakeSharedHiCacheScheduleManager:
    def __init__(self, *, plans, results):
        self.plans = dict(plans)
        self.results = dict(results)
        self.prepared = []
        self.released = []

    def has_reuse_plan(self, req):
        return self.plans.get(req.rid, False)

    def prepare_reuse(self, req):
        self.prepared.append(req.rid)
        result = self.results[req.rid]
        if isinstance(result, Exception):
            raise result
        return result

    def release_request(self, rid):
        self.released.append(str(rid))


def _make_manager(tree=None):
    tree = tree or FakeTree()
    manager = SharedHiCacheManager.__new__(SharedHiCacheManager)
    manager.tree_cache = tree
    manager.worker_id = 42
    manager._set_parallel_metadata(
        {
            "tp_rank": 0,
            "tp_size": 1,
            "pp_rank": 0,
            "pp_size": 1,
            "attn_cp_rank": 0,
            "attn_cp_size": 1,
        },
    )
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
    manager._finished_plan_prefix_lens = {}
    manager.target_cache = SharedHiCacheTarget(tree_cache=tree, metrics_collector=None)
    manager.source_service = None
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
        shared_hicache_plan=SharedHiCachePlan.coerce(plan),
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

    def test_plan_accepts_tp_rank_metadata(self):
        plan = SharedHiCachePlan.from_dict(
            _make_plan(
                [11],
                source_tp_rank=1,
                source_tp_size=2,
                target_tp_rank=1,
                target_tp_size=2,
            )
        )

        self.assertEqual(plan.source_tp_rank, 1)
        self.assertEqual(plan.source_tp_size, 2)
        self.assertEqual(plan.target_tp_rank, 1)
        self.assertEqual(plan.target_tp_size, 2)

        with self.assertRaisesRegex(ValueError, "source_tp_rank"):
            SharedHiCachePlan.from_dict(
                _make_plan([11], source_tp_rank=2, source_tp_size=2)
            )

    def test_control_endpoint_formats_tp_rank_fields(self):
        endpoint = format_control_endpoint(
            "127.0.0.1:391{tp_rank}",
            {
                "tp_rank": 1,
            },
        )

        self.assertEqual(endpoint, "http://127.0.0.1:3911")

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
            tree,
            plan,
            start_block=0,
            max_blocks=1,
            worker_id=7,
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
        cache.hicache_host_index = HiCacheHostBlockIndex(cache.page_size)
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

    def test_shared_hicache_device_insert_does_not_write_through(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        captured = []

        def insert(params):
            captured.append(params)
            return SimpleNamespace(prefix_len=0)

        cache.insert = insert
        key = RadixKey(array("q", [1, 2]))
        value = torch.tensor([10, 11], dtype=torch.int64)

        result = cache.insert_shared_hicache_device_blocks(key=key, value=value)

        self.assertEqual(result.prefix_len, 0)
        self.assertEqual(len(captured), 1)
        self.assertIsInstance(captured[0], InsertParams)
        self.assertTrue(captured[0].chunked)
        self.assertIs(captured[0].key, key)
        self.assertTrue(torch.equal(captured[0].value, value))

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
            first = manager.prepare_reuse(req)
            self.assertTrue(first.pending)
            pending = manager._pending_fetches[req.rid]
            pending.future.result(timeout=1)

            second = manager.prepare_reuse(req)
            self.assertFalse(second.pending)
            self.assertEqual(second.staged_tokens, 4)
            self.assertEqual(req.shared_hicache_hit_length, 4)
            self.assertEqual(len(manager.tree_cache.insert_calls), 1)
        finally:
            manager._fetch_executor.shutdown(wait=False, cancel_futures=True)

    def test_manager_releases_unusable_direct_transfer_target_pages(self):
        cases = (
            (SHARED_HICACHE_DIRECT_TIMEOUT_REASON, [], True),
            (
                "direct_transfer_failed:NIXL direct KV transfer failed",
                [200, 201, 202, 203],
                False,
            ),
        )
        for reason, expected_freed, expect_quarantine in cases:
            with self.subTest(reason=reason):
                tree = FakeTree()
                manager = _make_manager(tree)
                plan = SharedHiCachePlan.from_dict(_make_plan([11, 22]))
                req = _make_req(plan.to_dict())
                device_indices = torch.arange(200, 204)
                pending = SharedHiCachePendingFetch(
                    plan=plan,
                    plan_offset=0,
                    target_start_block=0,
                    expected_hashes=plan.planned_hashes,
                    future=_completed_future(([], reason)),
                    device_indices=device_indices,
                    backend="nixl",
                    submitted_at=time.perf_counter(),
                )
                manager._pending_fetches[req.rid] = pending

                result = manager.prepare_reuse(req)

                self.assertEqual(result.staged_tokens, 0)
                self.assertEqual(tree.device_allocator.freed, expected_freed)
                self.assertEqual(
                    manager.target_cache.quarantined_device_indices,
                    [device_indices] if expect_quarantine else [],
                )

    def test_nixl_backend_registers_source_and_transfers_pages(self):
        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        host_buffer = torch.zeros((64,), dtype=torch.uint8)
        item_len = source_k[0].nbytes * page_size
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
        source_agent = FakeNixlAgent("source-agent", b"source-metadata")
        target_agent = FakeNixlAgent("target-agent", b"target-metadata")
        source_backend = NixlSharedHiCacheTransferBackend(
            agent=source_agent,
            agent_name="source-agent",
            backend_name="UCX",
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_item_lens=[item_len, item_len],
            target_registered=True,
            gpu_id=1,
            transfer_parallelism=1,
        )
        target_backend = NixlSharedHiCacheTransferBackend(
            agent=target_agent,
            agent_name="target-agent",
            backend_name="UCX",
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_item_lens=[item_len, item_len],
            target_registered=True,
            gpu_id=2,
            transfer_parallelism=1,
        )
        source_backend._register_source_host_pool()

        source_backend.transfer_pages(
            target_session_id=target_backend.target_session_id,
            source_page_indices=torch.tensor([1, 2], dtype=torch.int32).numpy(),
            target_page_indices=torch.tensor([3, 4], dtype=torch.int32).numpy(),
            target_kv_ptrs=target_backend.target_kv_ptrs,
            target_kv_item_lens=target_backend.target_kv_item_lens,
            target_metadata=target_backend.target_descriptor(),
        )

        self.assertTrue(source_backend.enabled)
        self.assertEqual(source_agent.remote_agents, [b"target-metadata"])
        self.assertEqual(source_agent.xfer_desc_calls[0][0], "DRAM")
        self.assertEqual(source_agent.xfer_desc_calls[1][0], "VRAM")
        self.assertEqual(source_agent.xfer_desc_calls[0][1][0][2], 0)
        self.assertEqual(source_agent.xfer_desc_calls[1][1][0][2], 2)
        self.assertEqual(source_agent.initialized[0][0], "WRITE")
        self.assertEqual(source_agent.initialized[0][3], "target-agent")
        self.assertEqual(source_agent.transfers, ["handle-1"])
        self.assertEqual(source_agent.released, ["handle-1"])

    def test_make_transfer_backend_rejects_pp_and_cp_when_requested(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                shared_hicache_config={"transfer_backend": "nixl"},
                tp_size=1,
                pp_size=2,
                attn_cp_size=1,
            )
        )

        with self.assertRaisesRegex(RuntimeError, "PP/CP are deferred"):
            make_shared_hicache_transfer_backend(scheduler)

        scheduler.server_args.pp_size = 1
        scheduler.server_args.attn_cp_size = 2
        with self.assertRaisesRegex(RuntimeError, "PP/CP are deferred"):
            make_shared_hicache_transfer_backend(scheduler)

    def test_source_transfer_rejects_wrong_tp_rank_metadata(self):
        plan = SharedHiCachePlan.from_dict(
            _make_plan(
                [11],
                source_tp_size=2,
                target_tp_size=2,
            )
        )
        response = handle_source_transfer(
            payload={
                "plan": plan.to_dict(),
                "start_block": 0,
                "max_blocks": 1,
                "target_session_id": "target-session",
                "transfer_backend": "nixl",
                "target_metadata": {
                    "backend": "nixl",
                    "session_id": "target-session",
                    "tp_rank": 1,
                    "tp_size": 2,
                },
                "target_kv_ptrs": [1],
                "target_kv_item_lens": [64],
                "target_page_indices": [0],
            },
            transfer_backend=FakeDirectTransfer(),
            tree_cache=FakeTree(),
            worker_id=7,
            tp_rank=0,
            tp_size=2,
        )

        self.assertFalse(response["ok"])
        self.assertIn("wrong_source_tp_rank_for_target", response["reason"])

    def test_manager_formats_plan_endpoint_by_current_tp_rank(self):
        manager = _make_manager()
        manager._set_parallel_metadata(
            {
                "tp_rank": 1,
                "tp_size": 2,
            },
        )
        plan = SharedHiCachePlan.from_dict(
            _make_plan(
                [11],
                source_endpoint="127.0.0.1:3900{source_tp_rank}",
                source_tp_size=2,
                target_tp_size=2,
            )
        )

        self.assertEqual(
            manager._candidate_endpoints_for_plan(plan),
            ["http://127.0.0.1:39001"],
        )

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
                        "transfer_backend": "nixl",
                        "timeout_secs": 2.5,
                    }
                ),
            ]
        )
        server_args = ServerArgs.from_cli_args(args)

        self.assertTrue(server_args.enable_shared_hicache)
        self.assertTrue(server_args.enable_hierarchical_cache)
        self.assertEqual(server_args.shared_hicache_worker_id, 7)
        self.assertIsInstance(server_args.shared_hicache_config, SharedHiCacheConfig)
        self.assertEqual(server_args.shared_hicache_config.worker_id, 7)
        self.assertEqual(
            server_args.shared_hicache_config.control_endpoint,
            "http://127.0.0.1:39007",
        )
        self.assertEqual(server_args.shared_hicache_config.transfer_backend, "nixl")
        self.assertEqual(server_args.shared_hicache_config.timeout_secs, 2.5)

    def test_manager_uses_server_arg_worker_id_after_dynamo_identity_update(self):
        server_args = SimpleNamespace(
            enable_shared_hicache=True,
            shared_hicache_worker_id=99,
            shared_hicache_config=SharedHiCacheConfig(
                worker_id=2,
                control_endpoint=None,
                timeout_secs=1.0,
                transfer_backend="auto",
            ),
            hicache_storage_prefetch_policy="timeout",
            tp_size=2,
            pp_size=1,
            attn_cp_size=1,
            ep_size=1,
        )
        scheduler = SimpleNamespace(
            server_args=server_args,
            enable_hierarchical_cache=True,
            tree_cache=SimpleNamespace(
                lookup_hicache_host_blocks=lambda *_args, **_kwargs: ({}, []),
                insert_shared_hicache_device_blocks=lambda **_kwargs: None,
                prefetch_timeout_config=None,
            ),
            ps=SimpleNamespace(
                tp_rank=1,
                tp_size=2,
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
            ),
            enable_metrics=False,
        )

        with patch(
            "sglang.srt.mem_cache.shared_hicache.manager.make_shared_hicache_transfer_backend",
            return_value=None,
        ):
            manager = SharedHiCacheManager.from_scheduler(scheduler)

        self.assertIsNotNone(manager)
        try:
            self.assertEqual(manager.worker_id, 99)
            self.assertEqual(manager.tp_rank, 1)
            self.assertEqual(manager.tp_size, 2)
        finally:
            manager.shutdown()

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

        nixl_backend = parser.parse_args(
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
        self.assertEqual(
            ServerArgs.from_cli_args(nixl_backend).shared_hicache_config.transfer_backend,
            "nixl",
        )

        bad_backend = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-hierarchical-cache",
                "--shared-hicache-worker-id",
                "7",
                "--shared-hicache-config",
                json.dumps({"transfer_backend": "bogus"}),
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
            backend="nixl",
            outcome="hit",
            reason="ok",
            tokens=4,
            wait_ms=12.5,
            insert_ms=0.25,
            transfer_bytes=2 * 1024 * 1024,
        )

        labels = {
            "model_name": "dummy",
            "backend": "nixl",
            "outcome": "hit",
            "reason": "ok",
        }
        self.assertEqual(
            collector.shared_hicache_requests_total.calls, [("inc", labels, 1)]
        )
        self.assertEqual(
            collector.shared_hicache_tokens_total.calls, [("inc", labels, 4)]
        )
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
            rid="r",
            shared_hicache_plan=[plan_0, plan_1],
        )

        req.normalize_batch_and_arguments()

        self.assertEqual(req[0].shared_hicache_plan, SharedHiCachePlan.coerce(plan_0))
        self.assertEqual(req[1].shared_hicache_plan, SharedHiCachePlan.coerce(plan_1))

    def test_generate_req_rejects_shared_hicache_parallel_sampling(self):
        req = GenerateReqInput(
            text="hello",
            sampling_params={"n": 2},
            shared_hicache_plan=_make_plan([1]),
        )

        with self.assertRaisesRegex(ValueError, "parallel_sample_num > 1"):
            req.normalize_batch_and_arguments()

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

    def test_scheduler_batches_shared_hicache_tp_coordination(self):
        manager = FakeSharedHiCacheScheduleManager(
            plans={"rid-1": True, "rid-2": True, "rid-3": False},
            results={
                "rid-1": SimpleNamespace(pending=False, prefix_len=8),
                "rid-2": SimpleNamespace(pending=True, prefix_len=0),
            },
        )
        scheduler = FakeSharedHiCacheScheduler(manager, tp_size=2)
        reqs = [
            FakeSharedHiCacheReq("rid-1"),
            FakeSharedHiCacheReq("rid-2"),
            FakeSharedHiCacheReq("rid-3"),
        ]
        all_reduce_shapes = []

        def all_reduce(tensor, op, group):
            self.assertIs(group, scheduler.tp_cpu_group)
            all_reduce_shapes.append(tuple(tensor.shape))
            if op == torch.distributed.ReduceOp.SUM:
                tensor.mul_(scheduler.ps.tp_size)
            elif op == torch.distributed.ReduceOp.MIN:
                tensor[0] = 6
            else:
                raise AssertionError(f"unexpected op: {op}")

        with patch("torch.distributed.all_reduce", side_effect=all_reduce):
            pending_rids = scheduler._prepare_shared_hicache_for_schedule_batch(reqs)

        self.assertEqual(pending_rids, {"rid-2"})
        self.assertEqual(all_reduce_shapes, [(3,), (2,), (1,)])
        self.assertEqual(manager.prepared, ["rid-1", "rid-2"])
        self.assertEqual(reqs[0].shared_hicache_max_prefix_len, 6)
        self.assertEqual(reqs[1].shared_hicache_max_prefix_len, 0)
        self.assertIsNone(reqs[2].shared_hicache_max_prefix_len)

        with patch(
            "torch.distributed.all_reduce",
            side_effect=AssertionError("final init should not all-reduce"),
        ):
            scheduler._init_next_round_input_with_shared_hicache_tp_sync(reqs[0])

        self.assertEqual(reqs[0].init_calls[-1]["max_prefix_len"], 6)
        self.assertIsNone(reqs[0].shared_hicache_max_prefix_len)

    def test_scheduler_bounds_shared_hicache_prepare_to_candidate_prefix(self):
        manager = FakeSharedHiCacheScheduleManager(
            plans={f"rid-{i}": True for i in range(5)},
            results={
                f"rid-{i}": SimpleNamespace(pending=False, prefix_len=8)
                for i in range(5)
            },
        )
        scheduler = FakeSharedHiCacheScheduler(manager, tp_size=1, allocatable_reqs=2)
        reqs = [FakeSharedHiCacheReq(f"rid-{i}") for i in range(5)]

        candidates = scheduler._shared_hicache_schedule_candidates(reqs, running_bs=0)
        pending_rids = scheduler._prepare_shared_hicache_for_schedule_batch(candidates)

        self.assertEqual(pending_rids, set())
        self.assertEqual([req.rid for req in candidates], ["rid-0", "rid-1", "rid-2"])
        self.assertEqual(manager.prepared, ["rid-0", "rid-1", "rid-2"])
        self.assertEqual(reqs[3].init_calls, [])
        self.assertEqual(reqs[4].init_calls, [])

    def test_scheduler_uses_prefill_max_requests_for_shared_hicache_prefix(self):
        scheduler = FakeSharedHiCacheScheduler(
            FakeSharedHiCacheScheduleManager(plans={}, results={}),
            tp_size=1,
            allocatable_reqs=8,
            prefill_max_requests=2,
        )
        reqs = [FakeSharedHiCacheReq(f"rid-{i}") for i in range(5)]

        candidates = scheduler._shared_hicache_schedule_candidates(reqs, running_bs=0)

        self.assertEqual([req.rid for req in candidates], ["rid-0", "rid-1", "rid-2"])

    def test_scheduler_falls_back_on_divergent_shared_hicache_plan(self):
        manager = FakeSharedHiCacheScheduleManager(
            plans={"rid-1": True},
            results={"rid-1": SimpleNamespace(pending=False, prefix_len=8)},
        )
        scheduler = FakeSharedHiCacheScheduler(manager, tp_size=2)
        req = FakeSharedHiCacheReq("rid-1")

        def all_reduce(tensor, op, group):
            self.assertEqual(op, torch.distributed.ReduceOp.SUM)
            self.assertEqual(tensor.tolist(), [1])

        with patch("torch.distributed.all_reduce", side_effect=all_reduce):
            pending_rids = scheduler._prepare_shared_hicache_for_schedule_batch([req])

        self.assertEqual(pending_rids, set())
        self.assertIsNone(req.shared_hicache_plan)
        self.assertEqual(manager.prepared, [])
        self.assertEqual(manager.released, ["rid-1"])


if __name__ == "__main__":
    unittest.main()
