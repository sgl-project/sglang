import time
import threading
import unittest
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.hicache_host_index import HiCacheHostBlockIndex
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.shared_hicache.manager import SharedHiCacheManager
from sglang.srt.mem_cache.shared_hicache.pending import SharedHiCachePendingFetch
from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
    SharedHiCachePlan,
)
from sglang.srt.mem_cache.shared_hicache.scheduler_mixin import (
    SharedHiCacheSchedulerMixin,
)
from sglang.srt.mem_cache.shared_hicache.service import (
    _decode_control_payload,
    _encode_control_payload,
)
from sglang.srt.mem_cache.shared_hicache.source import (
    ResolvedHostPage,
    execute_source_transfer_request,
    parse_source_transfer_request,
    resolve_host_pages,
)
from sglang.srt.mem_cache.shared_hicache.target import SharedHiCacheTarget
from sglang.srt.mem_cache.shared_hicache.transfer import (
    NixlSharedHiCacheTransferBackend,
    SharedHiCacheTransferBackend,
    build_shared_hicache_transfer_notification,
)
from sglang.srt.mem_cache.utils import block_hash_aliases, hash_str_to_int64
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector


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


class FakeTransferHandle:
    def __init__(self, result, status=KVPoll.Success, pending_polls=0):
        self._result = result
        self._status = status
        self._pending_polls = pending_polls
        self.done_at = time.perf_counter() if pending_polls == 0 else 0.0

    def poll(self):
        if self._pending_polls > 0:
            self._pending_polls -= 1
            if self._pending_polls == 0:
                self.done_at = time.perf_counter()
            return KVPoll.Transferring
        if self.done_at <= 0:
            self.done_at = time.perf_counter()
        return self._status

    def done(self):
        return self.poll() in (KVPoll.Success, KVPoll.Failed)

    def result(self):
        self.done()
        return self._result


class FakeDeviceAllocator:
    def __init__(self):
        self.next_index = 200
        self.freed = []
        self.fail_alloc = False

    def alloc(self, need_size):
        if self.fail_alloc:
            return None
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
        self.evict_count = 0

    def insert_shared_hicache_device_blocks(self, *, key, value):
        self.insert_calls.append((key, value.clone()))
        return SimpleNamespace(prefix_len=self.insert_prefix_len)

    def insert(self, params):
        self.insert_calls.append((params.key, params.value.clone()))
        return SimpleNamespace(prefix_len=self.insert_prefix_len)

    def evict(self, params):
        self.evict_count += 1
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
        self.calls = []
        self.dropped_notifications = []

    @property
    def enabled(self):
        return True

    def transfer_pages(self, **kwargs):
        self.calls.append(kwargs)

    def drop_target_transfer_notification(self, transfer_id):
        self.dropped_notifications.append(str(transfer_id))


class FakeNixlAgent:
    def __init__(self, name="agent", metadata=b"agent-metadata"):
        self.name = name
        self.metadata = metadata
        self.registered = []
        self.remote_agents = []
        self.xfer_desc_calls = []
        self.initialized = []
        self.transfers = []
        self.released = []
        self.notifications = {}

    def register_memory(self, addrs, mem_type=None):
        normalized = [
            tuple(int(x) if isinstance(x, int) else x for x in item) for item in addrs
        ]
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

    def get_new_notifs(self):
        notifications = self.notifications
        self.notifications = {}
        return notifications


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


class FakeSharedHiCacheReq:
    def __init__(
        self, rid, shared_hicache_plan=True, local_prefix_len=0, host_hit_length=0
    ):
        self.rid = rid
        self.shared_hicache_plan = shared_hicache_plan
        self.shared_hicache_max_prefix_len = None
        self.local_prefix_len = local_prefix_len
        self.host_hit_length = host_hit_length
        self.prefix_indices = torch.arange(local_prefix_len, dtype=torch.int64)
        self.init_calls = []

    def init_next_round_input(self, tree_cache=None, cow_mamba=None):
        self.init_calls.append(
            {
                "tree_cache": tree_cache,
                "cow_mamba": cow_mamba,
                "max_prefix_len": self.shared_hicache_max_prefix_len,
            }
        )
        prefix_len = self.local_prefix_len
        if self.shared_hicache_max_prefix_len is not None:
            prefix_len = min(prefix_len, self.shared_hicache_max_prefix_len)
        self.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)


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
    manager.direct_transfer = FakeDirectTransfer()
    manager.metrics_collector = None
    manager._source_transfer_executor = ThreadPoolExecutor(max_workers=1)
    manager._source_transfer_capacity = threading.BoundedSemaphore(2)
    manager._source_transfer_lock = threading.Lock()
    manager._source_transfers = {}
    manager._target_transfer_lock = threading.Lock()
    manager._target_transfer_completions = {}
    manager._active_target_transfers = set()
    manager._target_transfer_capacity = None
    manager._pending_fetches = {}
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
    def test_control_payload_uses_json_bytes(self):
        payload = {
            "kind": "shared_hicache_transfer_request",
            "transfer_id": "transfer-1",
            "target_page_indices": [1, 2],
        }

        encoded = _encode_control_payload(payload)

        self.assertIsInstance(encoded, bytes)
        self.assertNotIn(b"\x80", encoded[:1])
        self.assertEqual(_decode_control_payload([encoded]), payload)

    def test_plan_uses_canonical_schema_only(self):
        plan = SharedHiCachePlan.from_dict(
            _make_plan([11], source_tp_rank=0, source_tp_size=1)
        )

        self.assertEqual(plan.source_medium, StorageMedium.CPU.value)
        self.assertEqual(plan.source_endpoint, "tcp://127.0.0.1:39007")
        self.assertEqual(plan.block_hashes, (11,))

        with self.assertRaisesRegex(ValueError, "block_hash must be an integer"):
            SharedHiCachePlan.from_dict(_make_plan([{"block_hash": 11}]))

    def test_hiradix_cpu_events_maintain_host_index(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.enable_kv_cache_events = False
        cache.hicache_host_index = HiCacheHostBlockIndex(cache.page_size)
        node = TreeNode()
        node.host_value = torch.tensor([0, 1], dtype=torch.int64)
        node.hash_value = ["aa" * 32]

        cache._record_store_event(node, medium=StorageMedium.CPU)

        block_hash = hash_str_to_int64("aa" * 32)
        matches, protected = cache.lookup_hicache_host_blocks(
            {block_hash}, protect=True
        )
        for alias in block_hash_aliases(block_hash):
            self.assertIn(alias, matches)
        self.assertEqual(protected, [node])
        self.assertEqual(node.host_ref_counter, 1)
        node.release_host()

        cache._record_remove_event(node, medium=StorageMedium.CPU)
        self.assertEqual(cache.lookup_hicache_host_blocks({block_hash}), {})

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
        self.assertEqual(
            pages, [ResolvedHostPage(identity_hash, "aa" * 32, bytes([1, 2, 3, 4]))]
        )
        self.assertEqual(calls, [({kv_hash}, True)])
        self.assertEqual(node.host_ref_counter, 0)

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
        self.assertIsInstance(captured[0], InsertParams)
        self.assertTrue(captured[0].chunked)

    def test_target_direct_transfer_allocation_does_not_evict(self):
        tree = FakeTree()
        tree.device_allocator.fail_alloc = True
        target = SharedHiCacheTarget(tree_cache=tree, metrics_collector=None)

        self.assertIsNone(target.alloc_device_indices(4))
        self.assertEqual(tree.evict_count, 0)

    def test_manager_stages_direct_transfer_into_local_prefix(self):
        manager = _make_manager()
        req = _make_req(_make_plan([11, 22]))

        def submit_direct_transfer(plan, *, start_block, max_blocks, token_count):
            self.assertEqual(start_block, 0)
            self.assertEqual(max_blocks, 2)
            self.assertEqual(token_count, 4)
            return (
                FakeTransferHandle(
                    (
                        [
                            ResolvedHostPage(11, "", b""),
                            ResolvedHostPage(22, "", b""),
                        ],
                        "ok",
                    )
                ),
                torch.arange(200, 204),
            )

        manager._submit_direct_transfer = submit_direct_transfer
        try:
            first = manager.prepare_reuse(req)
            self.assertTrue(first.pending)

            second = manager.prepare_reuse(req)
            self.assertFalse(second.pending)
            self.assertEqual(second.staged_tokens, 4)
            self.assertEqual(req.shared_hicache_hit_length, 4)
            self.assertEqual(len(manager.tree_cache.insert_calls), 1)
        finally:
            manager._source_transfer_executor.shutdown(wait=False, cancel_futures=True)

    def test_manager_releases_failed_direct_transfer_target_pages(self):
        for reason, expected_freed, expect_quarantine in (
            (SHARED_HICACHE_DIRECT_TIMEOUT_REASON, [], True),
            ("direct_transfer_failed", [200, 201, 202, 203], False),
        ):
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
                    transfer=FakeTransferHandle(([], reason), status=KVPoll.Failed),
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
                manager._source_transfer_executor.shutdown(
                    wait=False, cancel_futures=True
                )

    def test_manager_rejects_wrong_target_tp_rank(self):
        manager = _make_manager()
        manager._set_parallel_metadata(
            {
                "tp_rank": 1,
                "tp_size": 2,
                "pp_rank": 0,
                "pp_size": 1,
                "attn_cp_rank": 0,
                "attn_cp_size": 1,
                "attn_tp_rank": 1,
                "attn_tp_size": 2,
                "attn_dp_rank": 0,
                "attn_dp_size": 1,
            }
        )
        plan = SharedHiCachePlan.from_dict(
            _make_plan(
                [11],
                source_tp_rank=0,
                source_tp_size=2,
                target_tp_rank=0,
                target_tp_size=2,
            )
        )

        self.assertEqual(
            manager._validate_plan(plan),
            "wrong_target_tp_rank:plan=0:local=1",
        )

    def test_source_transfer_immediate_completion_does_not_leave_future(self):
        class ImmediateExecutor:
            def submit(self, fn, **kwargs):
                future = Future()
                future.set_result(
                    {
                        "ok": False,
                        "reason": "missing_first_block",
                        "transferred_blocks": 0,
                        "block_size_tokens": 2,
                    }
                )
                return future

        manager = _make_manager()
        manager._source_transfer_executor.shutdown(wait=False, cancel_futures=True)
        manager._source_transfer_executor = ImmediateExecutor()
        manager.source_service = SimpleNamespace(
            send=lambda _endpoint, _payload: None,
        )
        plan = SharedHiCachePlan.from_dict(
            _make_plan([11], source_worker_id=manager.worker_id)
        )
        payload = {
            "kind": "shared_hicache_transfer_request",
            "transfer_id": "transfer-1",
            "target_control_endpoint": "tcp://127.0.0.1:49999",
            "plan": plan.to_dict(),
            "start_block": 0,
            "max_blocks": 1,
            "target_session_id": "target-session",
            "transfer_backend": "nixl",
            "target_metadata": {
                "backend": "nixl",
                "session_id": "target-session",
                "tp_rank": 0,
                "tp_size": 1,
            },
            "target_kv_ptrs": [1],
            "target_kv_item_lens": [64],
            "target_page_indices": [0],
        }

        response = manager._handle_source_transfer(payload)

        self.assertTrue(response["accepted"])
        with manager._source_transfer_lock:
            self.assertNotIn("transfer-1", manager._source_transfers)

    def test_manager_source_transfer_submit_returns_before_transfer_completes(self):
        tree = FakeTree()
        manager = _make_manager(tree)
        plan = SharedHiCachePlan.from_dict(
            _make_plan([11], source_worker_id=manager.worker_id)
        )
        node = TreeNode()
        node.host_value = torch.tensor([0, 2], dtype=torch.int64)
        node.hash_value = ["aa" * 32]
        transfer_started = threading.Event()
        release_transfer = threading.Event()

        def lookup(wanted_hashes, *, protect=False):
            node.protect_host()
            return {11: (node, 0, "aa" * 32)}, [node]

        def blocking_transfer(**kwargs):
            transfer_started.set()
            release_transfer.wait(timeout=2)
            manager.direct_transfer.calls.append(kwargs)

        tree.lookup_hicache_host_blocks = lookup
        manager.direct_transfer.transfer_pages = blocking_transfer
        sent_messages = []
        manager.source_service = SimpleNamespace(
            send=lambda endpoint, payload: sent_messages.append(
                (endpoint, dict(payload))
            )
        )
        payload = {
            "kind": "shared_hicache_transfer_request",
            "transfer_id": "transfer-1",
            "target_control_endpoint": "tcp://127.0.0.1:49999",
            "plan": plan.to_dict(),
            "start_block": 0,
            "max_blocks": 1,
            "target_session_id": "target-session",
            "transfer_backend": "nixl",
            "target_metadata": {
                "backend": "nixl",
                "session_id": "target-session",
                "tp_rank": 0,
                "tp_size": 1,
            },
            "target_kv_ptrs": [1],
            "target_kv_item_lens": [64],
            "target_page_indices": [0],
        }

        try:
            response = manager._handle_source_transfer(payload)

            self.assertTrue(response["accepted"])
            self.assertEqual(response["transfer_id"], "transfer-1")
            self.assertTrue(transfer_started.wait(timeout=1))
            with manager._source_transfer_lock:
                self.assertIn("transfer-1", manager._source_transfers)

            release_transfer.set()
            for _ in range(100):
                if sent_messages:
                    break
                time.sleep(0.01)

            self.assertEqual(sent_messages[0][0], "tcp://127.0.0.1:49999")
            status = sent_messages[0][1]
            self.assertEqual(status["kind"], "shared_hicache_transfer_done")
            self.assertTrue(status["ok"])
            self.assertEqual(status["transferred_blocks"], 1)
            self.assertEqual(len(manager.direct_transfer.calls), 1)
            self.assertEqual(node.host_ref_counter, 0)
            with manager._source_transfer_lock:
                self.assertNotIn("transfer-1", manager._source_transfers)
        finally:
            release_transfer.set()
            manager._source_transfer_executor.shutdown(wait=False, cancel_futures=True)

    def test_source_transfer_rejects_wrong_tp_rank_metadata(self):
        plan = SharedHiCachePlan.from_dict(
            _make_plan(
                [11],
                source_tp_rank=0,
                source_tp_size=2,
                target_tp_rank=1,
                target_tp_size=2,
            )
        )
        request, error = parse_source_transfer_request(
            payload={
                "transfer_id": "transfer-1",
                "target_control_endpoint": "tcp://127.0.0.1:49999",
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
        )
        self.assertIsNone(error)

        response = execute_source_transfer_request(
            request=request,
            transfer_backend=FakeDirectTransfer(),
            tree_cache=FakeTree(),
            worker_id=7,
            tp_rank=0,
            tp_size=2,
            attn_tp_size=2,
        )

        self.assertFalse(response["ok"])
        self.assertIn("wrong_source_tp_rank_for_target", response["reason"])

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
            agent=FakeNixlAgent("source-target-agent", b"source-target-metadata"),
            agent_name="source-target-agent",
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

        with patch(
            "sglang.srt.mem_cache.shared_hicache.transfer._create_nixl_agent",
            return_value=(source_agent, "source-agent", "UCX"),
        ):
            source_backend.transfer_pages(
                target_session_id=target_backend.target_session_id,
                source_page_indices=torch.tensor([1, 2], dtype=torch.int32).numpy(),
                target_page_indices=torch.tensor([3, 4], dtype=torch.int32).numpy(),
                target_kv_ptrs=target_backend.target_kv_ptrs,
                target_kv_item_lens=target_backend.target_kv_item_lens,
                target_metadata=target_backend.target_descriptor(),
                notification="shared-hicache-notification",
            )

        self.assertTrue(source_backend.enabled)
        self.assertEqual(source_agent.registered[0][0], "DRAM")
        self.assertEqual(source_agent.remote_agents, [b"target-metadata"])
        self.assertEqual(source_agent.xfer_desc_calls[0][0], "DRAM")
        self.assertEqual(source_agent.xfer_desc_calls[1][0], "VRAM")
        self.assertEqual(source_agent.initialized[0][0], "WRITE")
        self.assertEqual(source_agent.initialized[0][3], "target-agent")
        self.assertEqual(source_agent.initialized[0][4], b"shared-hicache-notification")
        self.assertEqual(source_agent.transfers, ["handle-1"])
        self.assertEqual(source_agent.released, ["handle-1"])

    def test_nixl_backend_polls_target_transfer_notifications(self):
        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        host_buffer = torch.zeros((64,), dtype=torch.uint8)
        item_len = source_k[0].nbytes * page_size
        tree = SimpleNamespace(
            page_size=page_size,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    kv_buffer=host_buffer,
                    k_data_refs=[source_k],
                    v_data_refs=[source_k],
                )
            ),
        )
        target_agent = FakeNixlAgent("target-agent", b"target-metadata")
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
        target_agent.notifications = {
            "source-agent": [
                build_shared_hicache_transfer_notification(
                    transfer_id="transfer-1",
                    transferred_blocks=3,
                    reason="partial",
                ).encode("utf-8")
            ]
        }

        self.assertEqual(
            target_backend.pop_target_transfer_notification("transfer-1"),
            (3, "partial"),
        )
        self.assertIsNone(target_backend.pop_target_transfer_notification("transfer-1"))

        target_backend.drop_target_transfer_notification("transfer-2")
        target_agent.notifications = {
            "source-agent": [
                build_shared_hicache_transfer_notification(
                    transfer_id="transfer-2",
                    transferred_blocks=1,
                    reason="ok",
                ).encode("utf-8")
            ]
        }
        self.assertIsNone(target_backend.pop_target_transfer_notification("transfer-2"))

    def test_nixl_backend_uses_per_thread_source_agents_for_concurrent_transfers(self):
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
        target_backend = NixlSharedHiCacheTransferBackend(
            agent=FakeNixlAgent("target-agent", b"target-metadata"),
            agent_name="target-agent",
            backend_name="UCX",
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_item_lens=[item_len, item_len],
            target_registered=True,
            gpu_id=2,
            transfer_parallelism=1,
        )
        shared_in_flight = {"current": 0, "max": 0}
        shared_lock = threading.Lock()

        class BlockingNixlAgent(FakeNixlAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def transfer(self, handle):
                with shared_lock:
                    shared_in_flight["current"] += 1
                    shared_in_flight["max"] = max(
                        shared_in_flight["max"], shared_in_flight["current"]
                    )
                try:
                    self.transfers.append(handle)
                    time.sleep(0.05)
                finally:
                    with shared_lock:
                        shared_in_flight["current"] -= 1
                return "DONE"

        source_backend = NixlSharedHiCacheTransferBackend(
            agent=FakeNixlAgent("source-target-agent", b"source-target-metadata"),
            agent_name="source-target-agent",
            backend_name="UCX",
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_item_lens=[item_len, item_len],
            target_registered=True,
            gpu_id=1,
            transfer_parallelism=1,
        )
        source_backend._register_source_host_pool()
        self.assertTrue(source_backend.enabled)
        source_agents = [
            BlockingNixlAgent("source-agent-1", b"source-metadata-1"),
            BlockingNixlAgent("source-agent-2", b"source-metadata-2"),
        ]

        def run_transfer():
            source_backend.transfer_pages(
                target_session_id=target_backend.target_session_id,
                source_page_indices=torch.tensor([1, 2], dtype=torch.int32).numpy(),
                target_page_indices=torch.tensor([3, 4], dtype=torch.int32).numpy(),
                target_kv_ptrs=target_backend.target_kv_ptrs,
                target_kv_item_lens=target_backend.target_kv_item_lens,
                target_metadata=target_backend.target_descriptor(),
            )

        start = time.perf_counter()
        with patch(
            "sglang.srt.mem_cache.shared_hicache.transfer._create_nixl_agent",
            side_effect=[
                (source_agents[0], "source-agent-1", "UCX"),
                (source_agents[1], "source-agent-2", "UCX"),
            ],
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(run_transfer) for _ in range(2)]
                for future in futures:
                    future.result(timeout=3)
        elapsed = time.perf_counter() - start

        self.assertEqual(shared_in_flight["max"], 2)
        self.assertLess(elapsed, 0.095)
        for source_agent in source_agents:
            self.assertEqual(source_agent.registered[0][0], "DRAM")
            self.assertEqual(source_agent.remote_agents, [b"target-metadata"])
            self.assertEqual(
                [call[0] for call in source_agent.xfer_desc_calls],
                ["DRAM", "VRAM"],
            )
            self.assertEqual(source_agent.transfers, ["handle-1"])
            self.assertEqual(source_agent.released, ["handle-1"])

    def test_hiradix_shared_hicache_write_check_skips_empty_local_queue(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.enable_shared_hicache = True
        cache.ongoing_write_through = {}
        cache.cache_controller = SimpleNamespace(ack_write_queue=[])
        cache._all_reduce_attn_groups = Mock(side_effect=AssertionError)

        cache.writing_check()

        cache._all_reduce_attn_groups.assert_not_called()

    def test_hiradix_shared_hicache_write_check_does_not_drain_without_local_ack_queue(
        self,
    ):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.enable_shared_hicache = True
        cache.ongoing_write_through = {123: SimpleNamespace(id=123)}
        cache.cache_controller = SimpleNamespace(ack_write_queue=[])
        cache._all_reduce_attn_groups = Mock(side_effect=AssertionError)

        cache.writing_check()

        cache._all_reduce_attn_groups.assert_not_called()
        self.assertEqual(list(cache.ongoing_write_through), [123])

    def test_hiradix_shared_hicache_write_check_drains_local_queue(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.enable_shared_hicache = True
        cache.enable_storage = False
        node = SimpleNamespace(id=123)
        cache.ongoing_write_through = {node.id: node}

        class DoneEvent:
            def __init__(self):
                self.synchronized = False

            def query(self):
                return True

            def synchronize(self):
                self.synchronized = True

        finish_event = DoneEvent()
        cache.cache_controller = SimpleNamespace(
            ack_write_queue=[(None, finish_event, [node.id])]
        )
        stored = []
        unlocked = []

        cache._all_reduce_attn_groups = Mock(side_effect=AssertionError)
        cache._record_store_event = lambda node, medium=None: stored.append(
            (node, medium)
        )
        cache.dec_lock_ref = lambda node: unlocked.append(node)

        cache.writing_check()

        cache._all_reduce_attn_groups.assert_not_called()
        self.assertTrue(finish_event.synchronized)
        self.assertEqual(cache.cache_controller.ack_write_queue, [])
        self.assertEqual(cache.ongoing_write_through, {})
        self.assertEqual(stored, [(node, StorageMedium.CPU)])
        self.assertEqual(unlocked, [node])

    def test_metrics_use_reason_code_labels(self):
        collector = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        collector.labels = {"model_name": "dummy"}
        collector.shared_hicache_requests_total = FakePrometheusMetric()
        collector.shared_hicache_tokens_total = FakePrometheusMetric()
        collector.shared_hicache_wait_seconds = FakePrometheusMetric()
        collector.shared_hicache_insert_seconds = FakePrometheusMetric()
        collector.shared_hicache_transfer_bytes_total = FakePrometheusMetric()

        collector.observe_shared_hicache(
            backend="nixl",
            outcome="error",
            reason="direct_transfer_failed:detail_that_must_not_be_a_label",
            tokens=4,
            wait_ms=12.5,
            insert_ms=0.25,
            transfer_bytes=1024,
        )

        labels = {
            "model_name": "dummy",
            "backend": "nixl",
            "outcome": "error",
            "reason_code": "direct_transfer_failed",
        }
        self.assertEqual(
            collector.shared_hicache_requests_total.calls, [("inc", labels, 1)]
        )
        self.assertEqual(
            collector.shared_hicache_tokens_total.calls, [("inc", labels, 4)]
        )

    def test_scheduler_batches_tp_coordination_and_bounds_candidates(self):
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
        self.assertEqual(all_reduce_shapes, [(3,), (2,), (2,), (1,)])
        self.assertEqual(manager.prepared, ["rid-1", "rid-2"])
        self.assertEqual(reqs[0].shared_hicache_max_prefix_len, 6)

        with patch(
            "torch.distributed.all_reduce",
            side_effect=AssertionError("final init should not all-reduce"),
        ):
            scheduler._init_next_round_input_with_shared_hicache_tp_sync(reqs[0])
        self.assertEqual(reqs[0].init_calls[-1]["max_prefix_len"], 6)

        bounded_manager = FakeSharedHiCacheScheduleManager(
            plans={f"rid-{i}": True for i in range(5)},
            results={
                f"rid-{i}": SimpleNamespace(pending=False, prefix_len=8)
                for i in range(5)
            },
        )
        bounded_scheduler = FakeSharedHiCacheScheduler(
            bounded_manager, tp_size=1, allocatable_reqs=2, prefill_max_requests=2
        )
        bounded_reqs = [FakeSharedHiCacheReq(f"rid-{i}") for i in range(5)]

        candidates = bounded_scheduler._shared_hicache_schedule_candidates(
            bounded_reqs, running_bs=0
        )
        bounded_scheduler._prepare_shared_hicache_for_schedule_batch(candidates)

        self.assertEqual([req.rid for req in candidates], ["rid-0", "rid-1", "rid-2"])
        self.assertEqual(bounded_manager.prepared, ["rid-0", "rid-1", "rid-2"])
        self.assertEqual(bounded_reqs[3].init_calls, [])

    def test_scheduler_keeps_local_prefix_when_shared_hicache_prefix_is_shorter(self):
        manager = FakeSharedHiCacheScheduleManager(
            plans={"rid-1": True},
            results={"rid-1": SimpleNamespace(pending=False, prefix_len=8)},
        )
        scheduler = FakeSharedHiCacheScheduler(manager, tp_size=1)
        req = FakeSharedHiCacheReq("rid-1", local_prefix_len=24)

        pending_rids = scheduler._prepare_shared_hicache_for_schedule_batch([req])

        self.assertEqual(pending_rids, set())
        self.assertEqual(manager.prepared, ["rid-1"])
        self.assertEqual(req.init_calls[0]["max_prefix_len"], None)
        self.assertEqual(req.shared_hicache_max_prefix_len, 24)

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
