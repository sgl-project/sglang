import argparse
import base64
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.entrypoints.openai.utils import cached_tokens_details_from_dict
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.mem_cache.g2plus_transfer import (
    MooncakeG2plusTransferBackend,
    NixlG2plusTransferBackend,
    _apply_nixl_backend_thread_params,
    _default_nixl_num_threads,
    default_g2plus_transfer_parallelism,
    make_g2plus_transfer_backend,
)
from sglang.srt.mem_cache.router_kv_reuse import (
    RemoteG2ReuseHandler,
    RemoteKvReusePlan,
    ResolvedHostPage,
    RouterKVReuseResult,
    RouterKVReuseManager,
    resolve_host_pages,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.utils import (
    block_hash_aliases,
    compute_node_hash_values,
    hash_str_to_int64,
)


class FakeHostPool:
    dtype = torch.uint8

    def __init__(self, flat_page_numel=4):
        self.flat_page_numel = flat_page_numel
        self.pages = {}
        self.freed = []
        self.next_index = 100

    def get_data_page(self, index, flat=True):
        return self.pages[int(index)]

    def get_dummy_flat_data_page(self):
        return torch.zeros(self.flat_page_numel, dtype=self.dtype)

    def set_from_flat_data_page(self, index, data_page):
        self.pages[int(index)] = data_page.clone()

    def alloc(self, need_size):
        indices = torch.arange(self.next_index, self.next_index + need_size)
        self.next_index += need_size
        return indices

    def free(self, indices):
        self.freed.extend(int(idx) for idx in indices)
        return len(indices)


class CountingHostValue:
    def __init__(self, values):
        self.values = torch.tensor(values, dtype=torch.int64)
        self.reads = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        self.reads.append(list(index) if isinstance(index, list) else int(index))
        return self.values[index]


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


class FakeTree:
    def __init__(self, page_size=2):
        self.page_size = page_size
        self.is_eagle = False
        self.root_node = TreeNode()
        self.root_node.key = RadixKey([])
        self.device_allocator = FakeDeviceAllocator()
        self.cache_controller = SimpleNamespace(
            mem_pool_host=FakeHostPool(),
            mem_pool_device_allocator=self.device_allocator,
        )
        self.insert_calls = []
        self.insert_device_calls = []
        self.insert_prefix_len = 0
        self.locked_nodes = []
        self.unlocked_nodes = []

    def _insert_helper_host(self, node, key, host_value, hash_value):
        self.insert_calls.append((node, key, host_value.clone(), list(hash_value)))
        return 0

    def insert(self, params):
        self.insert_device_calls.append((params.key, params.value.clone()))
        return SimpleNamespace(prefix_len=self.insert_prefix_len)

    def evict(self, params):
        return SimpleNamespace(num_tokens_evicted=0)

    def evict_host(self, num_tokens):
        return 0

    def inc_lock_ref(self, node):
        self.locked_nodes.append(node)
        node.lock_ref += 1
        return SimpleNamespace(delta=0)

    def dec_lock_ref(self, node):
        self.unlocked_nodes.append(node)
        node.lock_ref -= 1
        return SimpleNamespace(delta=0)


def _make_plan(block_hashes, **overrides):
    plan = {
        "plan_id": "plan-1",
        "request_id": "request-1",
        "target_worker_id": 42,
        "target_dp_rank": 0,
        "source_worker_id": 7,
        "source_dp_rank": 0,
        "source_tier": "host_pinned",
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


def _wait_until_no_pending(handler, timeout_secs=1.0):
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        if not handler.has_pending():
            return True
        time.sleep(0.01)
    return not handler.has_pending()


class FakeMooncakeEngine:
    def __init__(self, ib_device=None):
        self.ib_device = ib_device
        self.transfers = []
        self.registered = []
        self.deregistered = []

    def get_session_id(self):
        return "target-session"

    def get_ib_device(self):
        return self.ib_device

    def batch_register(self, ptrs, lens):
        self.registered.append((ptrs, lens))
        return 0

    def batch_deregister(self, ptrs):
        self.deregistered.append(list(ptrs))
        return 0

    def batch_transfer_sync(self, session_id, src_addrs, dst_addrs, lengths):
        self.transfers.append((session_id, src_addrs, dst_addrs, lengths))
        return 0


class FakeNixlAgent:
    name = "source-agent"

    def __init__(self):
        self.remote_agents = []
        self.desc_calls = []
        self.registered = []
        self.xfers = []
        self.released_handles = []
        self.unregistered = []
        self.next_handle = 1

    def get_agent_metadata(self):
        return b"source-metadata"

    def add_remote_agent(self, metadata):
        self.remote_agents.append(metadata)

    def register_memory(self, addrs, memory_type):
        self.registered.append((memory_type, addrs))
        return [(memory_type, addrs)]

    def get_xfer_descs(self, reqs, memory_type):
        self.desc_calls.append((memory_type, reqs.copy()))
        return (memory_type, reqs.copy())

    def initialize_xfer(self, op, src_descs, dst_descs, peer_name, notif):
        self.xfers.append((op, src_descs, dst_descs, peer_name, notif))
        handle = f"handle-{self.next_handle}"
        self.next_handle += 1
        return handle

    def transfer(self, handle):
        return "PROC"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released_handles.append(handle)

    def unregister_memory(self, descs):
        self.unregistered.append(descs)


class FakeUrlopenResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeEvent:
    def query(self):
        return True

    def synchronize(self):
        return None


class FakeRunningFuture:
    def __init__(self):
        self.callbacks = []

    def cancel(self):
        return False

    def done(self):
        return False

    def add_done_callback(self, callback):
        self.callbacks.append(callback)


class FakeExecutor:
    def __init__(self):
        self.shutdown_calls = []

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


class FakeRouterMetricsCollector:
    def __init__(self):
        self.events = []
        self.quarantines = []

    def observe_router_kv_reuse(self, **kwargs):
        self.events.append(kwargs)

    def observe_router_kv_reuse_quarantine(self, **kwargs):
        self.quarantines.append(kwargs)


class FakePrometheusMetric:
    def __init__(self):
        self.calls = []

    def labels(self, **labels):
        metric = self

        class LabeledMetric:
            def inc(self, value=1):
                metric.calls.append(("inc", labels, value))

            def observe(self, value):
                metric.calls.append(("observe", labels, value))

            def set(self, value):
                metric.calls.append(("set", labels, value))

        return LabeledMetric()


class TestRouterKVReuse(unittest.TestCase):
    def test_block_hash_aliases_cover_signed_unsigned_int64_forms(self):
        signed = -(2**63) + 7
        unsigned = signed + 2**64

        self.assertEqual(block_hash_aliases(signed), {signed, unsigned})
        self.assertEqual(block_hash_aliases(unsigned), {signed, unsigned})
        self.assertEqual(block_hash_aliases(123), {123})

    def test_remote_kv_reuse_plan_parses_dynamo_shape(self):
        plan = RemoteKvReusePlan.from_dict(
            _make_plan(
                ["11", 22, {"value": 33}],
                start_block_index=4,
                kv_block_hashes=[111, 222, 333],
            )
        )

        self.assertEqual(plan.source_tier, "host_pinned")
        self.assertEqual(plan.block_hashes, (11, 22, 33))
        self.assertEqual(plan.kv_block_hashes, (111, 222, 333))
        self.assertEqual(plan.start_block_index, 4)
        self.assertTrue(plan.is_remote_g2())
        self.assertFalse(plan.is_expired())

    def test_remote_kv_reuse_plan_rejects_non_integer_fields(self):
        cases = [
            ("block_hashes", [1.25]),
            ("kv_block_hashes", [True]),
            ("target_worker_id", True),
            ("target_dp_rank", 0.5),
            ("source_worker_id", 7.9),
            ("source_dp_rank", False),
            ("planned_prefix_blocks", 1.5),
            ("start_block_index", 0.25),
            ("block_size_tokens", 2.1),
            ("created_at_ms", 1.1),
            ("expires_at_ms", 2.2),
            ("plan_version", 1.2),
        ]
        for field_name, value in cases:
            with self.subTest(field_name=field_name):
                if field_name == "block_hashes":
                    plan_data = _make_plan(value)
                else:
                    plan_data = _make_plan([11], **{field_name: value})
                with self.assertRaisesRegex(ValueError, "integer"):
                    RemoteKvReusePlan.from_dict(plan_data)

    def test_remote_kv_reuse_plan_rejects_non_array_hash_fields(self):
        cases = [
            ("block_hashes", 11),
            ("block_hashes", "11"),
            ("kv_block_hashes", 11),
            ("kv_block_hashes", "11"),
        ]
        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                plan_data = _make_plan([11])
                plan_data[field_name] = value
                with self.assertRaisesRegex(ValueError, "array"):
                    RemoteKvReusePlan.from_dict(plan_data)

    def test_remote_kv_reuse_plan_rejects_missing_required_fields(self):
        plan_data = _make_plan([11])
        del plan_data["target_worker_id"]

        with self.assertRaisesRegex(ValueError, "missing target_worker_id"):
            RemoteKvReusePlan.from_dict(plan_data)

    def test_remote_g2_handler_detects_valid_reuse_plan(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.tree_cache = FakeTree(page_size=2)
        req = SimpleNamespace(
            remote_kv_reuse_plan=_make_plan([11], target_worker_id=42)
        )

        self.assertTrue(handler.has_reuse_plan(req))

        req.remote_kv_reuse_plan = _make_plan([11], target_worker_id=99)
        self.assertFalse(handler.has_reuse_plan(req))

    def test_remote_g2_handler_requires_worker_identity(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.worker_id = None
        handler.dp_rank = 0
        handler.tree_cache = FakeTree(page_size=2)
        handler.direct_transfer = SimpleNamespace(enabled=True, name="mooncake")
        handler.metrics_collector = None
        req = SimpleNamespace(
            rid="r-missing-worker",
            remote_kv_reuse_plan=_make_plan([11], target_worker_id=42),
        )

        self.assertFalse(handler.has_reuse_plan(req))
        self.assertEqual(handler.check_remote_prefix(req), RouterKVReuseResult())

    def test_remote_g2_handler_does_not_reserve_prefetch_for_unexecutable_plan(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.tree_cache = FakeTree(page_size=2)
        handler.direct_transfer = None
        handler.allow_http_staging = False
        req = SimpleNamespace(
            remote_kv_reuse_plan=_make_plan([11], target_worker_id=42)
        )

        self.assertFalse(handler.has_reuse_plan(req))

    def test_remote_g2_handler_treats_disabled_direct_backend_as_unavailable(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.tree_cache = FakeTree(page_size=2)
        handler.direct_transfer = SimpleNamespace(name="mooncake", enabled=False)
        handler.allow_http_staging = False
        req = SimpleNamespace(
            remote_kv_reuse_plan=_make_plan([11], target_worker_id=42)
        )

        self.assertFalse(handler.has_reuse_plan(req))
        self.assertEqual(handler._current_backend_label(), "none")

    def test_source_resolver_rejects_oversized_control_payload(self):
        handler = RemoteG2ReuseHandler(
            server_args=SimpleNamespace(
                g2plus_timeout_secs=1,
                g2plus_transfer_backend="auto",
                g2plus_endpoint="127.0.0.1:0",
                g2plus_peer_endpoints=None,
                g2plus_fetch_workers=1,
            ),
            tree_cache=FakeTree(page_size=2),
            worker_id=7,
            dp_rank=0,
            direct_transfer=None,
        )
        handler.max_control_body_bytes = 8
        host, port = handler._source_server.server_address
        url = f"http://{host}:{port}/resolve"

        try:
            request = urllib.request.Request(
                url,
                data=b'{"oversized": true}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(request, timeout=1)
            self.assertEqual(cm.exception.code, 413)
            payload = json.loads(cm.exception.read().decode("utf-8"))
            self.assertEqual(payload["reason"], "control_payload_too_large")
            self.assertEqual(payload["pages"], [])
            self.assertTrue(_wait_until_no_pending(handler))
        finally:
            handler.shutdown()

    def test_source_resolver_rejects_http_staging_when_not_explicit_http(self):
        handler = RemoteG2ReuseHandler(
            server_args=SimpleNamespace(
                g2plus_timeout_secs=1,
                g2plus_transfer_backend="auto",
                g2plus_endpoint="127.0.0.1:0",
                g2plus_peer_endpoints=None,
                g2plus_fetch_workers=1,
            ),
            tree_cache=FakeTree(page_size=2),
            worker_id=7,
            dp_rank=0,
            direct_transfer=None,
        )
        host, port = handler._source_server.server_address
        request_body = json.dumps(
            {
                "plan": _make_plan([11]),
                "start_block": 0,
                "max_blocks": 1,
            }
        ).encode("utf-8")

        try:
            for path in ("/resolve", "/resolve_binary"):
                with self.subTest(path=path):
                    request = urllib.request.Request(
                        f"http://{host}:{port}{path}",
                        data=request_body,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with self.assertRaises(urllib.error.HTTPError) as cm:
                        urllib.request.urlopen(request, timeout=1)
                    self.assertEqual(cm.exception.code, 501)
                    payload = json.loads(cm.exception.read().decode("utf-8"))
                    self.assertEqual(payload["reason"], "http_staging_disabled")
                    self.assertEqual(payload["pages"], [])
                    self.assertTrue(_wait_until_no_pending(handler))
        finally:
            handler.shutdown()

    def test_source_resolver_rejects_malformed_json_control_payload(self):
        handler = RemoteG2ReuseHandler(
            server_args=SimpleNamespace(
                g2plus_timeout_secs=1,
                g2plus_transfer_backend="auto",
                g2plus_endpoint="127.0.0.1:0",
                g2plus_peer_endpoints=None,
                g2plus_fetch_workers=1,
            ),
            tree_cache=FakeTree(page_size=2),
            worker_id=7,
            dp_rank=0,
            direct_transfer=None,
        )
        host, port = handler._source_server.server_address

        try:
            request = urllib.request.Request(
                f"http://{host}:{port}/resolve",
                data=b"{",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(request, timeout=1)
            self.assertEqual(cm.exception.code, 400)
            payload = json.loads(cm.exception.read().decode("utf-8"))
            self.assertTrue(
                payload["reason"].startswith("malformed_control_payload:json:")
            )
            self.assertEqual(payload["pages"], [])
            self.assertTrue(_wait_until_no_pending(handler))
        finally:
            handler.shutdown()

    def test_source_resolver_allows_http_staging_when_explicit_http(self):
        handler = RemoteG2ReuseHandler(
            server_args=SimpleNamespace(
                g2plus_timeout_secs=1,
                g2plus_transfer_backend="http",
                g2plus_endpoint="127.0.0.1:0",
                g2plus_peer_endpoints=None,
                g2plus_fetch_workers=1,
            ),
            tree_cache=FakeTree(page_size=2),
            worker_id=7,
            dp_rank=0,
            direct_transfer=None,
        )
        host, port = handler._source_server.server_address
        request_body = json.dumps(
            {
                "plan": _make_plan([11]),
                "start_block": 0,
                "max_blocks": 1,
            }
        ).encode("utf-8")

        try:
            request = urllib.request.Request(
                f"http://{host}:{port}/resolve",
                data=request_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=1) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertNotEqual(payload["reason"], "http_staging_disabled")
            self.assertEqual(payload["pages"], [])
            self.assertTrue(_wait_until_no_pending(handler))
        finally:
            handler.shutdown()

    def test_source_resolver_rejects_malformed_plan_without_500(self):
        handler = RemoteG2ReuseHandler(
            server_args=SimpleNamespace(
                g2plus_timeout_secs=1,
                g2plus_transfer_backend="http",
                g2plus_endpoint="127.0.0.1:0",
                g2plus_peer_endpoints=None,
                g2plus_fetch_workers=1,
            ),
            tree_cache=FakeTree(page_size=2),
            worker_id=7,
            dp_rank=0,
            direct_transfer=None,
        )
        host, port = handler._source_server.server_address
        plan_data = _make_plan([11])
        plan_data["block_hashes"] = 11
        request_body = json.dumps(
            {
                "plan": plan_data,
                "start_block": 0,
                "max_blocks": 1,
            }
        ).encode("utf-8")

        try:
            request = urllib.request.Request(
                f"http://{host}:{port}/resolve",
                data=request_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(request, timeout=1)
            self.assertEqual(cm.exception.code, 400)
            payload = json.loads(cm.exception.read().decode("utf-8"))
            self.assertIn("block_hashes must be an array", payload["reason"])
            self.assertEqual(payload["pages"], [])
            self.assertTrue(_wait_until_no_pending(handler))
        finally:
            handler.shutdown()

    def test_source_resolver_returns_ordered_host_pages_and_releases_refs(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([1, 2, 3, 4])
        node.host_value = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        tree.cache_controller.mem_pool_host.pages[0] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8
        )
        tree.cache_controller.mem_pool_host.pages[2] = torch.tensor(
            [5, 6, 7, 8], dtype=torch.uint8
        )
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]
        plan = RemoteKvReusePlan.from_dict(_make_plan(block_hashes))

        pages, reason = resolve_host_pages(
            tree,
            plan,
            start_block=0,
            max_blocks=2,
            worker_id=7,
            dp_rank=0,
        )

        self.assertEqual(reason, "ok")
        self.assertEqual([page.block_hash for page in pages], block_hashes)
        self.assertEqual(pages[0].data, bytes([1, 2, 3, 4]))
        self.assertEqual(pages[1].data, bytes([5, 6, 7, 8]))
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_resolver_requires_worker_identity(self):
        tree = FakeTree(page_size=2)
        plan = RemoteKvReusePlan.from_dict(_make_plan([11]))

        pages, reason = resolve_host_pages(
            tree,
            plan,
            start_block=0,
            max_blocks=1,
            worker_id=None,
            dp_rank=0,
        )

        self.assertEqual(pages, [])
        self.assertEqual(reason, "missing_source_worker_id")

    def test_source_resolver_uses_tree_hash_index_when_available(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([1, 2, 3, 4])
        node.host_value = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.cache_controller.mem_pool_host.pages[0] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8
        )
        tree.cache_controller.mem_pool_host.pages[2] = torch.tensor(
            [5, 6, 7, 8], dtype=torch.uint8
        )
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]
        lookup_calls = []

        def lookup_router_kv_host_blocks(wanted_hashes):
            lookup_calls.append(set(wanted_hashes))
            return {
                block_hashes[0]: (node, 0, node.hash_value[0]),
                block_hashes[1]: (node, 1, node.hash_value[1]),
            }

        tree.lookup_router_kv_host_blocks = lookup_router_kv_host_blocks
        plan = RemoteKvReusePlan.from_dict(_make_plan(block_hashes))

        pages, reason = resolve_host_pages(
            tree,
            plan,
            start_block=0,
            max_blocks=2,
            worker_id=7,
            dp_rank=0,
        )

        self.assertEqual(reason, "ok")
        self.assertEqual([page.block_hash for page in pages], block_hashes)
        self.assertEqual(len(lookup_calls), 1)
        self.assertEqual(lookup_calls[0], set(block_hashes))

    def test_source_resolver_does_not_scan_tree_when_router_kv_index_misses(self):
        tree = FakeTree(page_size=2)
        tree.router_kv_block_index = {}
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([1, 2])
        node.host_value = torch.tensor([0, 1], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        tree.cache_controller.mem_pool_host.pages[0] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8
        )
        block_hash = hash_str_to_int64(node.hash_value[0])
        lookup_calls = []

        def lookup_router_kv_host_blocks(wanted_hashes):
            lookup_calls.append(set(wanted_hashes))
            return {}

        tree.lookup_router_kv_host_blocks = lookup_router_kv_host_blocks
        plan = RemoteKvReusePlan.from_dict(_make_plan([block_hash]))

        pages, reason = resolve_host_pages(
            tree,
            plan,
            start_block=0,
            max_blocks=1,
            worker_id=7,
            dp_rank=0,
        )

        self.assertEqual(reason, "missing_first_block")
        self.assertEqual(pages, [])
        self.assertEqual(len(lookup_calls), 1)
        self.assertEqual(lookup_calls[0], {block_hash})

    def test_source_resolver_does_not_drive_hicache_async_acks(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([1, 2])
        node.host_value = torch.tensor([0, 1], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        tree.cache_controller.mem_pool_host.pages[0] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8
        )
        tree.flush_write_through_acks = lambda: self.fail(
            "source resolver must not mutate HiCache async state"
        )
        block_hash = hash_str_to_int64(node.hash_value[0])
        plan = RemoteKvReusePlan.from_dict(_make_plan([block_hash]))

        pages, reason = resolve_host_pages(
            tree,
            plan,
            start_block=0,
            max_blocks=1,
            worker_id=7,
            dp_rank=0,
        )

        self.assertEqual(reason, "ok")
        self.assertEqual([page.block_hash for page in pages], [block_hash])

    def test_source_resolver_concurrency_guard_is_bounded(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler._source_resolver_semaphore = threading.BoundedSemaphore(1)

        self.assertTrue(handler._try_enter_source_resolver())
        self.assertFalse(handler._try_enter_source_resolver())

        handler._exit_source_resolver()
        self.assertTrue(handler._try_enter_source_resolver())
        handler._exit_source_resolver()

    def test_hiradix_router_kv_index_tracks_host_pages_and_aliases(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.router_kv_block_index = {}

        node = TreeNode()
        node.key = RadixKey([0, 1, 2, 3])
        node.host_value = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, cache.page_size)
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        cache._index_router_kv_host_node(node)

        wanted_hashes = set()
        for block_hash in block_hashes:
            wanted_hashes.update(block_hash_aliases(block_hash))
        matches = cache.lookup_router_kv_host_blocks(wanted_hashes)

        self.assertEqual(set(matches), wanted_hashes)
        for block_hash in block_hashes:
            for alias in block_hash_aliases(block_hash):
                self.assertIs(matches[alias][0], node)

        node.hash_value[0] = "00" * 32
        stale_aliases = set(block_hash_aliases(block_hashes[0]))
        self.assertEqual(cache.lookup_router_kv_host_blocks(stale_aliases), {})
        for alias in stale_aliases:
            self.assertNotIn(alias, cache.router_kv_block_index)

        cache._drop_router_kv_host_node(node)
        for alias in block_hash_aliases(block_hashes[1]):
            self.assertNotIn(alias, cache.router_kv_block_index)

    def test_hiradix_reset_clears_router_kv_index(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.router_kv_lock = threading.RLock()
        cache.router_kv_block_index = {123: ("stale-node", 0, "aa" * 32)}
        cache.cache_controller = SimpleNamespace(reset=lambda: None)
        cache.token_to_kv_pool_host = SimpleNamespace(clear=lambda: None)
        cache.prefetch_loaded_tokens_by_reqid = {"rid": 2}
        cache.evictable_host_leaves = {object()}
        cache.evictable_leaves = set()
        cache.device = torch.device("cpu")
        cache.enable_kv_cache_events = False
        cache.kv_event_queue = []

        cache.reset()

        self.assertEqual(cache.router_kv_block_index, {})
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid, {})
        self.assertEqual(cache.evictable_host_leaves, set())

    def test_hiradix_router_kv_index_publishes_after_write_ack(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.enable_storage = False
        cache.enable_router_kv_reuse = True
        cache.router_kv_block_index = {}
        cache.ongoing_write_through = {}
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])
        cache._record_store_event = lambda *args, **kwargs: None
        cache._all_reduce_attn_groups = lambda *args, **kwargs: None

        class FakeController:
            def __init__(self):
                self.ack_write_queue = []

            def write(self, *, device_indices, node_id, **kwargs):
                self.ack_write_queue.append(
                    (
                        None,
                        FakeEvent(),
                        [node_id],
                    )
                )
                return torch.tensor([100, 101, 102, 103], dtype=torch.int64)

        cache.cache_controller = FakeController()
        node = TreeNode()
        node.parent = cache.root_node
        node.key = RadixKey([0, 1, 2, 3])
        node.value = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        block_hashes = [
            hash_str_to_int64(hash_value)
            for hash_value in compute_node_hash_values(node, cache.page_size)
        ]

        cache.write_backup(node, write_back=True)

        wanted_hashes = set()
        for block_hash in block_hashes:
            wanted_hashes.update(block_hash_aliases(block_hash))
        self.assertEqual(cache.lookup_router_kv_host_blocks(wanted_hashes), {})

        cache.writing_check(write_back=True)

        matches = cache.lookup_router_kv_host_blocks(wanted_hashes)
        self.assertEqual(set(matches), wanted_hashes)
        for block_hash in block_hashes:
            for alias in block_hash_aliases(block_hash):
                self.assertIs(matches[alias][0], node)

    def test_hiradix_router_kv_index_rewrites_aliases_on_host_node_split(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.router_kv_block_index = {}
        cache.root_node = TreeNode()
        cache.root_node.key = RadixKey([])

        child = TreeNode()
        child.parent = cache.root_node
        child.key = RadixKey([0, 1, 2, 3])
        child.value = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
        child.host_value = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        child.hash_value = compute_node_hash_values(child, cache.page_size)
        cache.root_node.children[child.key.child_key(cache.page_size)] = child
        block_hashes = [
            hash_str_to_int64(hash_value) for hash_value in child.hash_value
        ]
        wanted_hashes = set()
        for block_hash in block_hashes:
            wanted_hashes.update(block_hash_aliases(block_hash))

        cache._index_router_kv_host_node(child)
        new_node = cache._split_node(child.key, child, 2)

        matches = cache.lookup_router_kv_host_blocks(wanted_hashes)

        for alias in block_hash_aliases(block_hashes[0]):
            self.assertIs(matches[alias][0], new_node)
            self.assertEqual(matches[alias][1], 0)
        for alias in block_hash_aliases(block_hashes[1]):
            self.assertIs(matches[alias][0], child)
            self.assertEqual(matches[alias][1], 0)

    def test_source_resolver_uses_kv_block_hashes_for_lookup(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([1, 2, 3, 4])
        node.host_value = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        tree.cache_controller.mem_pool_host.pages[0] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8
        )
        tree.cache_controller.mem_pool_host.pages[2] = torch.tensor(
            [5, 6, 7, 8], dtype=torch.uint8
        )
        kv_block_hashes = [
            hash_str_to_int64(hash_value) for hash_value in node.hash_value
        ]
        plan = RemoteKvReusePlan.from_dict(
            _make_plan([101, 102], kv_block_hashes=kv_block_hashes)
        )

        pages, reason = resolve_host_pages(
            tree,
            plan,
            start_block=0,
            max_blocks=2,
            worker_id=7,
            dp_rank=0,
        )

        self.assertEqual(reason, "ok")
        self.assertEqual([page.block_hash for page in pages], [101, 102])
        self.assertEqual(pages[0].data, bytes([1, 2, 3, 4]))
        self.assertEqual(pages[1].data, bytes([5, 6, 7, 8]))

    def test_source_resolver_accepts_unsigned_u64_hash_aliases(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node

        signed_hash = 0
        token = 0
        while signed_hash >= 0:
            node.key = RadixKey([token, token + 1])
            node.hash_value = compute_node_hash_values(node, tree.page_size)
            signed_hash = hash_str_to_int64(node.hash_value[0])
            token += 1

        node.host_value = torch.tensor([0, 1], dtype=torch.int64)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        tree.cache_controller.mem_pool_host.pages[0] = torch.tensor(
            [9, 8, 7, 6], dtype=torch.uint8
        )
        unsigned_hash = signed_hash + (1 << 64)
        plan = RemoteKvReusePlan.from_dict(_make_plan([unsigned_hash]))

        pages, reason = resolve_host_pages(
            tree,
            plan,
            start_block=0,
            max_blocks=1,
            worker_id=7,
            dp_rank=0,
        )

        self.assertEqual(reason, "ok")
        self.assertEqual([page.block_hash for page in pages], [unsigned_hash])
        self.assertEqual(pages[0].data, bytes([9, 8, 7, 6]))

    def test_target_insert_stages_pages_into_host_pool(self):
        tree = FakeTree(page_size=2)
        manager = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        manager.tree_cache = tree
        req = SimpleNamespace(
            fill_ids=[10, 11, 12, 13],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            host_hit_length=0,
        )
        pages = [
            ResolvedHostPage(11, "aa" * 32, bytes([1, 2, 3, 4])),
            ResolvedHostPage(22, "bb" * 32, bytes([5, 6, 7, 8])),
        ]

        staged = manager._insert_pages(req, pages, start_block=0)

        self.assertEqual(staged, 4)
        self.assertEqual(len(tree.insert_calls), 1)
        _, key, host_value, hash_values = tree.insert_calls[0]
        self.assertEqual(key.token_ids, [10, 11, 12, 13])
        self.assertEqual(hash_values, ["aa" * 32, "bb" * 32])
        self.assertEqual(host_value.tolist(), [100, 101, 102, 103])
        self.assertTrue(
            torch.equal(
                tree.cache_controller.mem_pool_host.pages[100],
                torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                tree.cache_controller.mem_pool_host.pages[102],
                torch.tensor([5, 6, 7, 8], dtype=torch.uint8),
            )
        )

    def test_remote_g2_handler_respects_plan_start_block(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()
        captured = {}

        def submit_fetch(plan, *, start_block, max_blocks):
            captured["start_block"] = start_block
            captured["max_blocks"] = max_blocks
            return _completed_future(
                (
                    [
                        ResolvedHostPage(11, "aa" * 32, bytes([1, 2, 3, 4])),
                        ResolvedHostPage(22, "bb" * 32, bytes([5, 6, 7, 8])),
                    ],
                    "ok",
                )
            )

        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r0",
            fill_ids=[10, 11, 12, 13, 14, 15, 16, 17],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.tensor([0, 1], dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22], start_block_index=1),
        )

        first = handler.check_remote_prefix(req)
        self.assertTrue(first.pending)
        second = handler.check_remote_prefix(req)

        self.assertEqual(second.staged_tokens, 4)
        self.assertEqual(req.remote_g2_hit_length, 4)
        self.assertEqual(captured, {"start_block": 0, "max_blocks": 2})
        _, key, _, _ = tree.insert_calls[0]
        self.assertEqual(key.token_ids, [12, 13, 14, 15])

    def test_remote_g2_direct_transfer_inserts_device_node(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            self.assertEqual(start_block, 0)
            self.assertEqual(max_blocks, 2)
            self.assertEqual(token_count, 4)
            return (
                _completed_future(
                    (
                        [
                            ResolvedHostPage(11, "aa" * 32, b""),
                            ResolvedHostPage(22, "bb" * 32, b""),
                        ],
                        "ok",
                    )
                ),
                device_indices,
            )

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("HTTP fallback should not run for direct transfer")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        first = handler.check_remote_prefix(req)
        self.assertTrue(first.pending)
        second = handler.check_remote_prefix(req)

        self.assertEqual(second.staged_tokens, 4)
        self.assertEqual(req.remote_g2_hit_length, 4)
        self.assertEqual(len(tree.insert_device_calls), 1)
        key, value = tree.insert_device_calls[0]
        self.assertEqual(key.token_ids, [10, 11, 12, 13])
        self.assertEqual(value.tolist(), [200, 201, 202, 203])
        self.assertEqual(tree.device_allocator.freed, [])

    def test_remote_g2_direct_transfer_accepts_real_req(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return (
                _completed_future(
                    (
                        [
                            ResolvedHostPage(11, "aa" * 32, b""),
                            ResolvedHostPage(22, "bb" * 32, b""),
                        ],
                        "ok",
                    )
                ),
                device_indices,
            )

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("HTTP fallback should not run for direct transfer")
        )
        req = Req(
            rid="real-req",
            origin_input_text="",
            origin_input_ids=[10, 11, 12, 13, 14, 15],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )
        req.fill_ids = req.origin_input_ids
        req.last_node = tree.root_node

        self.assertEqual(req.remote_g2_hit_length, 0)
        first = handler.check_remote_prefix(req)
        self.assertTrue(first.pending)
        second = handler.check_remote_prefix(req)

        self.assertEqual(second.staged_tokens, 4)
        self.assertEqual(req.remote_g2_hit_length, 4)
        self.assertEqual(len(tree.insert_device_calls), 1)

    def test_remote_g2_direct_transfer_attaches_after_device_prefix(self):
        tree = FakeTree(page_size=2)
        tree.insert_prefix_len = 2
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            self.assertEqual(start_block, 1)
            self.assertEqual(max_blocks, 2)
            self.assertEqual(token_count, 4)
            return (
                _completed_future(
                    (
                        [
                            ResolvedHostPage(22, "bb" * 32, b""),
                            ResolvedHostPage(33, "cc" * 32, b""),
                        ],
                        "ok",
                    )
                ),
                device_indices,
            )

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("HTTP fallback should not run for direct transfer")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-suffix",
            fill_ids=[10, 11, 12, 13, 14, 15, 16, 17],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.tensor([50, 51], dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22, 33]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 4)
        self.assertEqual(req.remote_g2_hit_length, 4)
        self.assertEqual(len(tree.insert_device_calls), 1)
        key, value = tree.insert_device_calls[0]
        self.assertEqual(key.token_ids, [10, 11, 12, 13, 14, 15])
        self.assertEqual(value.tolist(), [50, 51, 200, 201, 202, 203])
        self.assertEqual(tree.device_allocator.freed, [])

    def test_remote_g2_direct_transfer_partial_result_frees_unused_device_pages(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 206)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            self.assertEqual(start_block, 0)
            self.assertEqual(max_blocks, 3)
            self.assertEqual(token_count, 6)
            return (
                _completed_future(
                    (
                        [
                            ResolvedHostPage(11, "aa" * 32, b""),
                            ResolvedHostPage(22, "bb" * 32, b""),
                        ],
                        "partial",
                    )
                ),
                device_indices,
            )

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("HTTP fallback should not run for direct transfer")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-partial",
            fill_ids=[10, 11, 12, 13, 14, 15, 16, 17],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22, 33]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 4)
        self.assertEqual(req.remote_g2_hit_length, 4)
        self.assertEqual(tree.device_allocator.freed, [204, 205])
        self.assertEqual(len(tree.insert_device_calls), 1)
        key, value = tree.insert_device_calls[0]
        self.assertEqual(key.token_ids, [10, 11, 12, 13])
        self.assertEqual(value.tolist(), [200, 201, 202, 203])

    def test_remote_g2_direct_transfer_timeout_quarantines_target_pages(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()
        handler._quarantined_device_indices = []
        handler._quarantined_tokens_by_backend = {}
        handler.metrics_collector = FakeRouterMetricsCollector()

        device_indices = torch.arange(200, 204)
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22]))
        pending = SimpleNamespace(
            plan=plan,
            future=_completed_future(
                ([], "source_transfer_timeout_maybe_inflight:timed out")
            ),
            device_indices=device_indices,
            locked_node=None,
            backend="mooncake",
            submitted_at=0.0,
        )
        req = SimpleNamespace(rid="r-direct-timeout")

        result = handler._finish_pending_fetch(req, pending)

        self.assertEqual(result, RouterKVReuseResult())
        self.assertEqual(tree.device_allocator.freed, [])
        self.assertEqual(len(handler._quarantined_device_indices), 1)
        self.assertEqual(
            handler._quarantined_device_indices[0].tolist(),
            [200, 201, 202, 203],
        )
        self.assertEqual(handler._finished_plan_keys, {("r-direct-timeout", "plan-1")})
        self.assertEqual(
            handler.metrics_collector.quarantines,
            [
                {
                    "backend": "mooncake",
                    "reason": "source_transfer_timeout_maybe_inflight",
                    "tokens": 4,
                    "current_tokens": 4,
                }
            ],
        )

        handler._release_quarantined_device_indices()

        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])
        self.assertEqual(handler._quarantined_device_indices, [])
        self.assertEqual(
            handler.metrics_collector.quarantines[-1],
            {
                "backend": "mooncake",
                "reason": "released",
                "tokens": 0,
                "current_tokens": 0,
            },
        )

    def test_remote_g2_direct_transfer_frees_insert_matched_device_prefix(self):
        tree = FakeTree(page_size=2)
        tree.insert_prefix_len = 2
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return (
                _completed_future(
                    (
                        [
                            ResolvedHostPage(11, "aa" * 32, b""),
                            ResolvedHostPage(22, "bb" * 32, b""),
                        ],
                        "ok",
                    )
                ),
                device_indices,
            )

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("HTTP fallback should not run for direct transfer")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-matched-prefix",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 2)
        self.assertEqual(req.remote_g2_hit_length, 2)
        self.assertEqual(tree.device_allocator.freed, [200, 201])
        self.assertEqual(len(tree.insert_device_calls), 1)
        key, value = tree.insert_device_calls[0]
        self.assertEqual(key.token_ids, [10, 11, 12, 13])
        self.assertEqual(value.tolist(), [200, 201, 202, 203])

    def test_remote_g2_direct_transfer_protects_local_prefix_while_pending(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        prefix_node = TreeNode()
        prefix_node.parent = tree.root_node
        prefix_node.key = RadixKey([10, 11])
        prefix_node.value = torch.tensor([50, 51], dtype=torch.int64)
        tree.root_node.children[prefix_node.key.child_key(tree.page_size)] = prefix_node

        pending_future = Future()
        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            self.assertEqual(start_block, 1)
            self.assertEqual(max_blocks, 2)
            self.assertEqual(token_count, 4)
            self.assertEqual(prefix_node.lock_ref, 1)
            return pending_future, device_indices

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("HTTP fallback should not run for direct transfer")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-protect-prefix",
            fill_ids=[10, 11, 12, 13, 14, 15, 16, 17],
            extra_key=None,
            last_node=prefix_node,
            last_host_node=None,
            prefix_indices=torch.tensor([50, 51], dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22, 33]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        self.assertEqual(tree.locked_nodes, [prefix_node])
        self.assertEqual(prefix_node.lock_ref, 1)

        pending_future.set_result(
            (
                [
                    ResolvedHostPage(22, "bb" * 32, b""),
                    ResolvedHostPage(33, "cc" * 32, b""),
                ],
                "ok",
            )
        )
        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 4)
        self.assertEqual(req.remote_g2_hit_length, 4)
        self.assertEqual(tree.unlocked_nodes, [prefix_node])
        self.assertEqual(prefix_node.lock_ref, 0)
        key, value = tree.insert_device_calls[0]
        self.assertEqual(key.token_ids, [10, 11, 12, 13, 14, 15])
        self.assertEqual(value.tolist(), [50, 51, 200, 201, 202, 203])

    def test_remote_g2_direct_transfer_failure_becomes_cache_miss(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return (
                _completed_future(([], "mooncake_direct_unavailable")),
                device_indices,
            )

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("direct transfer failures should not use HTTP staging")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-fallback",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 0)
        self.assertEqual(req.remote_g2_hit_length, 0)
        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])
        self.assertEqual(len(tree.insert_calls), 0)
        self.assertEqual(handler._finished_plan_keys, {("r-direct-fallback", "plan-1")})

    def test_remote_g2_direct_insert_failure_becomes_cache_miss(self):
        tree = FakeTree(page_size=2)
        metrics = FakeRouterMetricsCollector()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(
            enabled=True,
            name="mooncake",
            target_kv_item_lens=[10, 20],
        )
        handler.metrics_collector = metrics
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return (
                _completed_future(
                    (
                        [
                            ResolvedHostPage(11, "aa" * 32, b""),
                            ResolvedHostPage(22, "bb" * 32, b""),
                        ],
                        "ok",
                    )
                ),
                device_indices,
            )

        def fail_insert(params):
            raise RuntimeError("insert failed")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("HTTP fallback should not run for direct transfer")
        )
        tree.insert = fail_insert
        req = SimpleNamespace(
            rid="r-direct-insert-failure",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 0)
        self.assertEqual(req.remote_g2_hit_length, 0)
        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])
        self.assertEqual(
            handler._finished_plan_keys, {("r-direct-insert-failure", "plan-1")}
        )
        self.assertEqual(len(metrics.events), 1)
        event = metrics.events[0]
        self.assertEqual(event["backend"], "mooncake")
        self.assertEqual(event["outcome"], "error")
        self.assertEqual(event["reason"], "insert_exception")
        self.assertEqual(event["tokens"], 0)
        self.assertEqual(event["transfer_bytes"], 60)
        self.assertGreaterEqual(event["insert_ms"], 0)

    def test_remote_g2_direct_transfer_records_hit_metrics(self):
        tree = FakeTree(page_size=2)
        metrics = FakeRouterMetricsCollector()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(
            enabled=True,
            name="mooncake",
            target_kv_item_lens=[10, 20],
        )
        handler.metrics_collector = metrics
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return (
                _completed_future(
                    (
                        [
                            ResolvedHostPage(11, "aa" * 32, b""),
                            ResolvedHostPage(22, "bb" * 32, b""),
                        ],
                        "ok",
                    )
                ),
                device_indices,
            )

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("HTTP fallback should not run for direct transfer")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-metrics-hit",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 4)
        self.assertEqual(len(metrics.events), 1)
        event = metrics.events[0]
        self.assertEqual(event["backend"], "mooncake")
        self.assertEqual(event["outcome"], "hit")
        self.assertEqual(event["reason"], "ok")
        self.assertEqual(event["tokens"], 4)
        self.assertEqual(event["transfer_bytes"], 60)
        self.assertGreaterEqual(event["wait_ms"], 0)
        self.assertGreaterEqual(event["insert_ms"], 0)

    def test_remote_g2_direct_transfer_records_miss_reason_metrics(self):
        tree = FakeTree(page_size=2)
        metrics = FakeRouterMetricsCollector()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(
            enabled=True,
            name="mooncake",
            target_kv_item_lens=[10, 20],
        )
        handler.metrics_collector = metrics
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return (
                _completed_future(([], "direct_transfer_failed:ucx timeout")),
                device_indices,
            )

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("direct transfer failures should not use HTTP staging")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-metrics-miss",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 0)
        self.assertEqual(len(metrics.events), 1)
        event = metrics.events[0]
        self.assertEqual(event["backend"], "mooncake")
        self.assertEqual(event["outcome"], "miss")
        self.assertEqual(event["reason"], "direct_transfer_failed")
        self.assertEqual(event["tokens"], 0)
        self.assertIsNone(event["transfer_bytes"])
        self.assertGreaterEqual(event["wait_ms"], 0)

    def test_remote_g2_direct_transfer_rejects_too_many_pages(self):
        tree = FakeTree(page_size=2)
        metrics = FakeRouterMetricsCollector()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(
            enabled=True,
            name="mooncake",
            target_kv_item_lens=[10, 20],
        )
        handler.metrics_collector = metrics
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return (
                _completed_future(
                    (
                        [
                            ResolvedHostPage(11, "aa" * 32, b""),
                            ResolvedHostPage(22, "bb" * 32, b""),
                            ResolvedHostPage(33, "cc" * 32, b""),
                        ],
                        "ok",
                    )
                ),
                device_indices,
            )

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("HTTP fallback should not run for direct transfer")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-too-many-pages",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 0)
        self.assertEqual(req.remote_g2_hit_length, 0)
        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])
        self.assertEqual(len(tree.insert_device_calls), 0)
        self.assertEqual(len(metrics.events), 1)
        event = metrics.events[0]
        self.assertEqual(event["outcome"], "error")
        self.assertEqual(event["reason"], "too_many_pages")
        self.assertEqual(event["transfer_bytes"], 90)

    def test_remote_g2_direct_allocation_failure_does_not_use_http_staging(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return None, None

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError(
                "direct allocation failures should not use HTTP staging"
            )

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-alloc-fail",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 0)
        self.assertFalse(result.pending)
        self.assertEqual(req.remote_g2_hit_length, 0)
        self.assertEqual(handler._pending_fetches, {})
        self.assertEqual(
            handler._finished_plan_keys, {("r-direct-alloc-fail", "plan-1")}
        )

    def test_remote_g2_direct_fetch_worker_backpressure_skips_hbm_allocation(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.endpoint = None
        handler.static_peer_endpoints = {"7:0": "http://127.0.0.1:39007"}
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._fetch_semaphore = threading.BoundedSemaphore(1)
        handler._fetch_semaphore.acquire()
        handler._fetch_executor = SimpleNamespace(
            submit=lambda *args, **kwargs: self.fail(
                "busy fetch pool must not submit direct transfer"
            )
        )
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22]))

        future, device_indices = handler._submit_direct_transfer(
            plan, start_block=0, max_blocks=2, token_count=4
        )

        self.assertIsNone(future)
        self.assertIsNone(device_indices)
        self.assertEqual(tree.device_allocator.next_index, 200)
        self.assertEqual(tree.device_allocator.freed, [])

    def test_remote_g2_direct_rejects_unaligned_device_allocation(self):
        tree = FakeTree(page_size=2)
        tree.device_allocator.next_index = 201
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.endpoint = None
        handler.static_peer_endpoints = {"7:0": "http://127.0.0.1:39007"}
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._fetch_executor = SimpleNamespace(
            submit=lambda *args, **kwargs: self.fail(
                "unaligned target allocation must not start direct transfer"
            )
        )
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22]))

        future, device_indices = handler._submit_direct_transfer(
            plan, start_block=0, max_blocks=2, token_count=4
        )

        self.assertIsNone(future)
        self.assertIsNone(device_indices)
        self.assertEqual(tree.device_allocator.freed, [201, 202, 203, 204])

    def test_remote_g2_auto_without_direct_backend_does_not_use_http_staging(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = None
        handler.allow_http_staging = False
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("auto without direct TE should not use HTTP staging")

        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-auto-no-direct",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 0)
        self.assertFalse(result.pending)
        self.assertEqual(handler._pending_fetches, {})
        self.assertEqual(
            handler._finished_plan_keys, {("r-auto-no-direct", "plan-1")}
        )

    def test_remote_g2_http_fetch_worker_backpressure_becomes_cache_miss(self):
        tree = FakeTree(page_size=2)
        metrics = FakeRouterMetricsCollector()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = None
        handler.allow_http_staging = True
        handler.metrics_collector = metrics
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()
        handler._fetch_semaphore = threading.BoundedSemaphore(1)
        handler._fetch_semaphore.acquire()
        handler._fetch_executor = SimpleNamespace(
            submit=lambda *args, **kwargs: self.fail(
                "busy fetch pool must not submit HTTP staging fetch"
            )
        )
        req = SimpleNamespace(
            rid="r-http-busy",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        result = handler.check_remote_prefix(req)

        self.assertFalse(result.pending)
        self.assertEqual(result.staged_tokens, 0)
        self.assertEqual(handler._pending_fetches, {})
        self.assertEqual(handler._finished_plan_keys, {("r-http-busy", "plan-1")})
        self.assertEqual(len(metrics.events), 1)
        self.assertEqual(metrics.events[0]["outcome"], "miss")
        self.assertEqual(metrics.events[0]["reason"], "fetch_workers_busy")

    def test_remote_g2_direct_skips_host_hit_http_staging(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            raise AssertionError(
                "direct path cannot attach while host hits are pending"
            )

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("direct-enabled path should not use HTTP staging")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-host-hit",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=tree.root_node,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=2,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        result = handler.check_remote_prefix(req)

        self.assertEqual(result.staged_tokens, 0)
        self.assertFalse(result.pending)
        self.assertEqual(handler._pending_fetches, {})
        self.assertEqual(handler._finished_plan_keys, {("r-direct-host-hit", "plan-1")})

    def test_remote_g2_release_request_cancels_pending_direct_transfer(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler._pending_fetches = {}
        handler._finished_plan_keys = {("r-direct-cancel", "old"), ("other", "plan")}

        pending_future = Future()
        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return pending_future, device_indices

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("HTTP fallback should not run for direct transfer")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-cancel",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        self.assertTrue(handler.prefetch_remote_prefix(req).pending)
        handler.release_request("r-direct-cancel")

        self.assertTrue(pending_future.cancelled())
        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])
        self.assertEqual(handler._pending_fetches, {})
        self.assertEqual(handler._finished_plan_keys, {("other", "plan")})

    def test_remote_g2_release_request_clears_finished_plan_without_pending(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler._pending_fetches = {}
        handler._finished_plan_keys = {
            ("r-finished", "old-plan"),
            ("other", "plan"),
        }

        handler.release_request("r-finished")

        self.assertEqual(handler._pending_fetches, {})
        self.assertEqual(handler._finished_plan_keys, {("other", "plan")})

    def test_remote_g2_release_request_unlocks_pending_prefix(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        prefix_node = TreeNode()
        prefix_node.parent = tree.root_node
        prefix_node.key = RadixKey([10, 11])
        prefix_node.value = torch.tensor([50, 51], dtype=torch.int64)
        tree.root_node.children[prefix_node.key.child_key(tree.page_size)] = prefix_node

        pending_future = Future()
        device_indices = torch.arange(200, 204)

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return pending_future, device_indices

        handler._submit_direct_transfer = submit_direct
        req = SimpleNamespace(
            rid="r-direct-release-prefix",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=prefix_node,
            last_host_node=None,
            prefix_indices=torch.tensor([50, 51], dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22, 33]),
        )

        self.assertTrue(handler.check_remote_prefix(req).pending)
        self.assertEqual(prefix_node.lock_ref, 1)

        handler.release_request("r-direct-release-prefix")

        self.assertTrue(pending_future.cancelled())
        self.assertEqual(tree.unlocked_nodes, [prefix_node])
        self.assertEqual(prefix_node.lock_ref, 0)
        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])

    def test_remote_g2_released_running_direct_transfer_remains_pending(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()
        handler._detached_fetches = set()

        running_future = Future()
        self.assertTrue(running_future.set_running_or_notify_cancel())
        device_indices = torch.arange(200, 204)
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22]))
        handler._pending_fetches = {
            "r-running-release": SimpleNamespace(
                plan=plan,
                future=running_future,
                device_indices=device_indices,
                locked_node=None,
            )
        }

        handler.release_request("r-running-release")

        self.assertFalse(running_future.cancelled())
        self.assertEqual(handler._pending_fetches, {})
        self.assertTrue(handler.has_pending())
        self.assertEqual(tree.device_allocator.freed, [])

        running_future.set_result(([], "ok"))

        self.assertFalse(handler.has_pending())
        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])

    def test_remote_g2_released_running_direct_transfer_frees_after_error(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()
        handler._detached_fetches = set()

        running_future = Future()
        self.assertTrue(running_future.set_running_or_notify_cancel())
        device_indices = torch.arange(200, 204)
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22]))
        handler._pending_fetches = {
            "r-running-error": SimpleNamespace(
                plan=plan,
                future=running_future,
                device_indices=device_indices,
                locked_node=None,
            )
        }

        handler.release_request("r-running-error")

        self.assertFalse(running_future.cancelled())
        self.assertEqual(handler._pending_fetches, {})
        self.assertTrue(handler.has_pending())
        self.assertEqual(tree.device_allocator.freed, [])

        running_future.set_exception(RuntimeError("transfer failed"))

        self.assertFalse(handler.has_pending())
        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])

    def test_remote_g2_release_request_quarantines_completed_direct_timeout(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()
        handler._quarantined_device_indices = []

        device_indices = torch.arange(200, 204)
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22]))
        handler._pending_fetches = {
            "r-completed-timeout-release": SimpleNamespace(
                plan=plan,
                future=_completed_future(
                    ([], "source_transfer_timeout_maybe_inflight:timed out")
                ),
                device_indices=device_indices,
                locked_node=None,
            )
        }

        handler.release_request("r-completed-timeout-release")

        self.assertEqual(handler._pending_fetches, {})
        self.assertEqual(tree.device_allocator.freed, [])
        self.assertEqual(len(handler._quarantined_device_indices), 1)
        self.assertEqual(
            handler._quarantined_device_indices[0].tolist(),
            [200, 201, 202, 203],
        )

    def test_remote_g2_released_running_direct_timeout_quarantines_when_done(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()
        handler._detached_fetches = set()
        handler._quarantined_device_indices = []

        running_future = Future()
        self.assertTrue(running_future.set_running_or_notify_cancel())
        device_indices = torch.arange(200, 204)
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22]))
        handler._pending_fetches = {
            "r-running-timeout-release": SimpleNamespace(
                plan=plan,
                future=running_future,
                device_indices=device_indices,
                locked_node=None,
            )
        }

        handler.release_request("r-running-timeout-release")

        self.assertFalse(running_future.cancelled())
        self.assertEqual(handler._pending_fetches, {})
        self.assertTrue(handler.has_pending())
        self.assertEqual(tree.device_allocator.freed, [])

        running_future.set_result(
            ([], "source_transfer_timeout_maybe_inflight:timed out")
        )

        self.assertFalse(handler.has_pending())
        self.assertEqual(tree.device_allocator.freed, [])
        self.assertEqual(len(handler._quarantined_device_indices), 1)
        self.assertEqual(
            handler._quarantined_device_indices[0].tolist(),
            [200, 201, 202, 203],
        )

    def test_remote_g2_active_source_resolver_counts_as_pending(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler._pending_fetches = {}
        handler._detached_fetches = set()
        handler._source_activity_lock = threading.Lock()
        handler._active_source_resolver_ops = 0
        handler._source_resolver_semaphore = threading.BoundedSemaphore(1)

        self.assertFalse(handler.has_pending())
        self.assertTrue(handler._try_enter_source_resolver())
        self.assertTrue(handler.has_pending())

        handler._exit_source_resolver()

        self.assertFalse(handler.has_pending())

    def test_remote_g2_plan_change_releases_old_pending_direct_transfer(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        old_future = Future()
        old_indices = torch.arange(200, 204)
        old_plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22], plan_id="old-plan"))
        locked_node = TreeNode()
        locked_node.parent = tree.root_node
        locked_node.key = RadixKey([10, 11])
        locked_node.value = torch.tensor([50, 51], dtype=torch.int64)
        tree.root_node.children[locked_node.key.child_key(tree.page_size)] = locked_node
        tree.inc_lock_ref(locked_node)
        handler._pending_fetches = {
            "r-plan-change": SimpleNamespace(
                plan=old_plan,
                future=old_future,
                device_indices=old_indices,
                locked_node=locked_node,
            )
        }
        handler._finished_plan_keys = set()

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            return None, None

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError("direct-enabled path should not use HTTP staging")

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-plan-change",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22], plan_id="new-plan"),
        )

        result = handler.check_remote_prefix(req)

        self.assertFalse(result.pending)
        self.assertTrue(old_future.cancelled())
        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])
        self.assertEqual(tree.unlocked_nodes, [locked_node])
        self.assertEqual(locked_node.lock_ref, 0)
        self.assertEqual(handler._pending_fetches, {})
        self.assertEqual(handler._finished_plan_keys, {("r-plan-change", "new-plan")})

    def test_remote_g2_shutdown_frees_running_direct_transfer_when_done(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler._shutdown = False
        handler._source_server = None
        handler._source_thread = None
        handler._fetch_executor = FakeExecutor()
        future = FakeRunningFuture()
        device_indices = torch.arange(200, 204)
        handler._pending_fetches = {
            "r-shutdown": SimpleNamespace(
                future=future,
                device_indices=device_indices,
            )
        }

        handler.shutdown()

        self.assertEqual(tree.device_allocator.freed, [])
        self.assertEqual(len(future.callbacks), 1)
        self.assertEqual(handler._pending_fetches, {})
        self.assertEqual(
            handler._fetch_executor.shutdown_calls,
            [{"wait": False, "cancel_futures": True}],
        )

        future.callbacks[0](future)

        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])

    def test_remote_g2_shutdown_defers_backend_close_until_direct_transfer_done(self):
        tree = FakeTree(page_size=2)
        shutdown_calls = []
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler._shutdown = False
        handler._source_server = None
        handler._source_thread = None
        handler._fetch_executor = FakeExecutor()
        handler._detached_fetches = set()
        handler._active_source_resolver_ops = 0
        handler.timeout_secs = 0
        handler.direct_transfer = SimpleNamespace(
            shutdown=lambda: shutdown_calls.append("shutdown")
        )
        running_future = Future()
        self.assertTrue(running_future.set_running_or_notify_cancel())
        device_indices = torch.arange(200, 204)
        handler._pending_fetches = {
            "r-shutdown-deferred": SimpleNamespace(
                future=running_future,
                device_indices=device_indices,
            )
        }

        handler.shutdown()

        self.assertEqual(shutdown_calls, [])
        self.assertTrue(handler.has_pending())

        running_future.set_result(([], "ok"))

        deadline = time.monotonic() + 1.0
        while not shutdown_calls and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(shutdown_calls, ["shutdown"])
        self.assertFalse(handler.has_pending())
        self.assertEqual(tree.device_allocator.freed, [200, 201, 202, 203])

    def test_remote_g2_shutdown_closes_idle_direct_transfer_backend(self):
        shutdown_calls = []
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler._shutdown = False
        handler._source_server = None
        handler._source_thread = None
        handler._fetch_executor = FakeExecutor()
        handler._pending_fetches = {}
        handler._detached_fetches = set()
        handler._active_source_resolver_ops = 0
        handler.timeout_secs = 1
        handler.direct_transfer = SimpleNamespace(
            shutdown=lambda: shutdown_calls.append("shutdown")
        )

        handler.shutdown()

        self.assertEqual(shutdown_calls, ["shutdown"])

    def test_remote_g2_queue_prefetch_skips_direct_backend(self):
        tree = FakeTree(page_size=2)
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.tree_cache = tree
        handler.worker_id = 42
        handler.dp_rank = 0
        handler.direct_transfer = SimpleNamespace(enabled=True)
        handler._pending_fetches = {}
        handler._finished_plan_keys = set()

        def submit_direct(plan, *, start_block, max_blocks, token_count):
            raise AssertionError("queue prefetch must not allocate target GPU slots")

        def submit_fetch(plan, *, start_block, max_blocks):
            raise AssertionError(
                "queue prefetch must not fall back when direct is enabled"
            )

        handler._submit_direct_transfer = submit_direct
        handler._submit_fetch = submit_fetch
        req = SimpleNamespace(
            rid="r-direct-prefetch",
            fill_ids=[10, 11, 12, 13, 14, 15],
            extra_key=None,
            last_node=tree.root_node,
            last_host_node=None,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            remote_g2_hit_length=0,
            remote_kv_reuse_plan=_make_plan([11, 22]),
        )

        result = handler.prefetch_remote_prefix(req)

        self.assertEqual(result, RouterKVReuseResult())
        self.assertEqual(handler._pending_fetches, {})

    def test_direct_transfer_control_plane_uses_backend_neutral_route(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.timeout_secs = 1
        requests = []
        transfer_backend = SimpleNamespace(
            name="nixl",
            target_session_id="target-agent",
            target_kv_ptrs=[1000, 2000],
            target_kv_item_lens=[64, 64],
            target_descriptor=lambda: {
                "backend": "nixl",
                "agent_name": "target-agent",
                "agent_metadata_b64": "dGFyZ2V0",
                "gpu_id": 1,
            },
        )
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22]))

        def fake_urlopen(request, timeout):
            requests.append((request.full_url, json.loads(request.data.decode())))
            self.assertEqual(timeout, 1)
            return FakeUrlopenResponse(
                {
                    "ok": True,
                    "reason": "ok",
                    "pages": [
                        {"block_hash": 11, "hash_value": "aa" * 32},
                        {"block_hash": 22, "hash_value": "bb" * 32},
                    ],
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            pages, reason = handler._request_source_transfer(
                transfer_backend=transfer_backend,
                endpoints=["http://127.0.0.1:39000"],
                plan=plan,
                start_block=0,
                max_blocks=2,
                target_page_indices=[3, 4],
            )

        self.assertEqual(reason, "ok")
        self.assertEqual([page.block_hash for page in pages], [11, 22])
        self.assertEqual(requests[0][0], "http://127.0.0.1:39000/transfer_direct")
        payload = requests[0][1]
        self.assertEqual(payload["transfer_backend"], "nixl")
        self.assertEqual(payload["target_metadata"]["agent_name"], "target-agent")
        self.assertEqual(payload["target_page_indices"], [3, 4])

    def test_direct_transfer_control_plane_timeout_is_indeterminate(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.timeout_secs = 1
        transfer_backend = SimpleNamespace(
            name="nixl",
            target_session_id="target-agent",
            target_kv_ptrs=[1000],
            target_kv_item_lens=[64],
            target_descriptor=lambda: {"backend": "nixl"},
        )
        plan = RemoteKvReusePlan.from_dict(_make_plan([11]))

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 1)
            raise TimeoutError("timed out")

        with patch("urllib.request.urlopen", fake_urlopen):
            pages, reason = handler._request_source_transfer(
                transfer_backend=transfer_backend,
                endpoints=["http://127.0.0.1:39000"],
                plan=plan,
                start_block=0,
                max_blocks=1,
                target_page_indices=[3],
            )

        self.assertEqual(pages, [])
        self.assertTrue(reason.startswith("source_transfer_timeout_maybe_inflight:"))

    def test_direct_transfer_control_plane_stops_on_indeterminate_source_rejection(
        self,
    ):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.timeout_secs = 1
        requests = []
        transfer_backend = SimpleNamespace(
            name="nixl",
            target_session_id="target-agent",
            target_kv_ptrs=[1000],
            target_kv_item_lens=[64],
            target_descriptor=lambda: {"backend": "nixl"},
        )
        plan = RemoteKvReusePlan.from_dict(_make_plan([11]))

        def fake_urlopen(request, timeout):
            requests.append(request.full_url)
            return FakeUrlopenResponse(
                {
                    "ok": False,
                    "reason": "source_transfer_timeout_maybe_inflight:source:timed out",
                    "pages": [],
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            pages, reason = handler._request_source_transfer(
                transfer_backend=transfer_backend,
                endpoints=["http://127.0.0.1:39000", "http://127.0.0.1:39001"],
                plan=plan,
                start_block=0,
                max_blocks=1,
                target_page_indices=[3],
            )

        self.assertEqual(pages, [])
        self.assertEqual(
            reason, "source_transfer_timeout_maybe_inflight:source:timed out"
        )
        self.assertEqual(requests, ["http://127.0.0.1:39000/transfer_direct"])

    def test_direct_transfer_control_plane_accepts_compact_block_count(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.timeout_secs = 1
        transfer_backend = SimpleNamespace(
            name="nixl",
            target_session_id="target-agent",
            target_kv_ptrs=[1000],
            target_kv_item_lens=[64],
            target_descriptor=lambda: {"backend": "nixl"},
        )
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22, 33]))

        def fake_urlopen(request, timeout):
            return FakeUrlopenResponse(
                {
                    "ok": True,
                    "reason": "ok",
                    "transferred_blocks": 2,
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            pages, reason = handler._request_source_transfer(
                transfer_backend=transfer_backend,
                endpoints=["http://127.0.0.1:39000"],
                plan=plan,
                start_block=1,
                max_blocks=2,
                target_page_indices=[3, 4],
            )

        self.assertEqual(reason, "ok")
        self.assertEqual([page.block_hash for page in pages], [22, 33])
        self.assertEqual([page.hash_value for page in pages], ["", ""])

    def test_direct_transfer_control_plane_returns_miss_on_source_rejection(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.timeout_secs = 1
        transfer_backend = SimpleNamespace(
            name="nixl",
            target_session_id="target-agent",
            target_kv_ptrs=[1000],
            target_kv_item_lens=[64],
            target_descriptor=lambda: {"backend": "nixl"},
        )
        plan = RemoteKvReusePlan.from_dict(_make_plan([11, 22]))

        def fake_urlopen(request, timeout):
            return FakeUrlopenResponse(
                {
                    "ok": False,
                    "reason": "direct_transfer_failed:ucx_error",
                    "pages": [],
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            pages, reason = handler._request_source_transfer(
                transfer_backend=transfer_backend,
                endpoints=["http://127.0.0.1:39000"],
                plan=plan,
                start_block=0,
                max_blocks=2,
                target_page_indices=[3, 4],
            )

        self.assertEqual(pages, [])
        self.assertEqual(reason, "direct_transfer_failed:ucx_error")

    def test_direct_transfer_control_plane_skips_malformed_source_response(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.timeout_secs = 1
        requests = []
        transfer_backend = SimpleNamespace(
            name="nixl",
            target_session_id="target-agent",
            target_kv_ptrs=[1000],
            target_kv_item_lens=[64],
            target_descriptor=lambda: {"backend": "nixl"},
        )
        plan = RemoteKvReusePlan.from_dict(_make_plan([11]))

        def fake_urlopen(request, timeout):
            requests.append(request.full_url)
            if len(requests) == 1:
                return FakeUrlopenResponse(["not-an-object"])
            return FakeUrlopenResponse(
                {
                    "ok": True,
                    "reason": "ok",
                    "pages": [{"block_hash": 11, "hash_value": "aa" * 32}],
                }
            )

        with patch("urllib.request.urlopen", fake_urlopen):
            pages, reason = handler._request_source_transfer(
                transfer_backend=transfer_backend,
                endpoints=["http://127.0.0.1:39000", "http://127.0.0.1:39001"],
                plan=plan,
                start_block=0,
                max_blocks=1,
                target_page_indices=[3],
            )

        self.assertEqual(requests[0], "http://127.0.0.1:39000/transfer_direct")
        self.assertEqual(requests[1], "http://127.0.0.1:39001/transfer_direct")
        self.assertEqual(reason, "ok")
        self.assertEqual([page.block_hash for page in pages], [11])

    def test_source_direct_transfer_returns_timing_fields(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11, 12, 13])
        node.host_value = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        class RecordingTransfer:
            name = "mooncake"
            enabled = True

            def __init__(self):
                self.calls = []

            def transfer_pages(self, **kwargs):
                self.calls.append(kwargs)

        transfer = RecordingTransfer()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = transfer
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 2,
                "target_session_id": "target-session",
                "target_page_indices": [3, 4],
                "target_kv_ptrs": [1000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertTrue(response["ok"])
        self.assertEqual(response["reason"], "ok")
        self.assertEqual(response["block_size_tokens"], 2)
        self.assertEqual(response["transfer_bytes"], 128)
        self.assertGreaterEqual(response["resolve_ms"], 0)
        self.assertGreaterEqual(response["transfer_ms"], 0)
        self.assertGreaterEqual(response["total_ms"], 0)
        self.assertEqual(response["transferred_blocks"], 2)
        self.assertNotIn("pages", response)
        self.assertEqual(len(transfer.calls), 1)
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_timeout_returns_indeterminate_reason(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11, 12, 13])
        node.host_value = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        class TimeoutTransfer:
            name = "mooncake"
            enabled = True

            def transfer_pages(self, **kwargs):
                raise TimeoutError("transport timed out")

        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = TimeoutTransfer()
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 2,
                "target_session_id": "target-session",
                "target_page_indices": [3, 4],
                "target_kv_ptrs": [1000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertFalse(response["ok"])
        self.assertTrue(
            response["reason"].startswith(
                "source_transfer_timeout_maybe_inflight:source:"
            )
        )
        self.assertEqual(response["pages"], [])
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_batches_host_value_reads_by_node(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11, 12, 13])
        node.host_value = CountingHostValue([100, 101, 102, 103])
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        class RecordingTransfer:
            name = "mooncake"
            enabled = True

            def __init__(self):
                self.source_page_indices = None

            def transfer_pages(self, **kwargs):
                self.source_page_indices = kwargs["source_page_indices"].tolist()

        transfer = RecordingTransfer()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = transfer
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 2,
                "target_session_id": "target-session",
                "target_page_indices": [3, 4],
                "target_kv_ptrs": [1000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertTrue(response["ok"])
        self.assertEqual(node.host_value.reads, [[0, 2]])
        self.assertEqual(transfer.source_page_indices, [50, 51])
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_failure_is_structured_and_releases_host_refs(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11, 12, 13])
        node.host_value = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        class FailingTransfer:
            name = "mooncake"
            enabled = True

            def transfer_pages(self, **kwargs):
                raise RuntimeError("rdma_write_failed")

        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = FailingTransfer()
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 2,
                "target_session_id": "target-session",
                "target_page_indices": [3, 4],
                "target_kv_ptrs": [1000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertEqual(response["reason"], "direct_transfer_failed:rdma_write_failed")
        self.assertEqual(response["transfer_bytes"], 128)
        self.assertGreaterEqual(response["resolve_ms"], 0)
        self.assertGreaterEqual(response["transfer_ms"], 0)
        self.assertGreaterEqual(response["total_ms"], 0)
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_rejects_disabled_backend(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = SimpleNamespace(name="mooncake", enabled=False)

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan([11]),
                "start_block": 0,
                "max_blocks": 1,
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertEqual(response["reason"], "direct_transfer_unavailable")

    def test_source_direct_transfer_rejects_malformed_plan_before_lookup(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = SimpleNamespace(name="mooncake", enabled=True)
        handler.tree_cache = FakeTree(page_size=2)
        handler.worker_id = 7
        handler.dp_rank = 0
        plan_data = _make_plan([11])
        plan_data["block_hashes"] = "11"

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": plan_data,
                "start_block": 0,
                "max_blocks": 1,
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertIn("block_hashes must be an array", response["reason"])

    def test_source_direct_transfer_rejects_non_integer_bounds_before_lookup(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = SimpleNamespace(name="mooncake", enabled=True)
        handler.tree_cache = FakeTree(page_size=2)
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan([11]),
                "start_block": 0.25,
                "max_blocks": 1,
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertIn("start_block must be an integer", response["reason"])

    def test_source_direct_transfer_rejects_short_target_indices_before_te(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11, 12, 13])
        node.host_value = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        class RecordingTransfer:
            name = "mooncake"
            enabled = True

            def __init__(self):
                self.calls = 0

            def transfer_pages(self, **kwargs):
                self.calls += 1

        transfer = RecordingTransfer()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = transfer
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 2,
                "target_session_id": "target-session",
                "target_page_indices": [3],
                "target_kv_ptrs": [1000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertEqual(
            response["reason"],
            "malformed_transfer_request:target_page_indices_too_short:1<2",
        )
        self.assertEqual(transfer.calls, 0)
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_rejects_scalar_target_indices_before_te(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11])
        node.host_value = torch.tensor([100, 101], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        def fail_if_host_protected():
            raise AssertionError("target indices should be validated before host lookup")

        node.protect_host = fail_if_host_protected

        class RecordingTransfer:
            name = "mooncake"
            enabled = True

            def __init__(self):
                self.calls = 0

            def transfer_pages(self, **kwargs):
                self.calls += 1

        transfer = RecordingTransfer()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = transfer
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 1,
                "target_session_id": "target-session",
                "target_page_indices": "3",
                "target_kv_ptrs": [1000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertEqual(
            response["reason"],
            "malformed_transfer_request:target_page_indices_must_be_array",
        )
        self.assertEqual(transfer.calls, 0)
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_rejects_unaligned_source_host_page_before_te(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11])
        node.host_value = torch.tensor([101, 102], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        class RecordingTransfer:
            name = "mooncake"
            enabled = True

            def __init__(self):
                self.calls = 0

            def transfer_pages(self, **kwargs):
                self.calls += 1

        transfer = RecordingTransfer()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = transfer
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 1,
                "target_session_id": "target-session",
                "target_page_indices": [3],
                "target_kv_ptrs": [1000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertEqual(response["reason"], "source_host_page_index_unaligned")
        self.assertEqual(transfer.calls, 0)
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_rejects_out_of_range_target_index_before_te(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11])
        node.host_value = torch.tensor([100, 101], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        def fail_if_host_protected():
            raise AssertionError("target indices should be validated before host lookup")

        node.protect_host = fail_if_host_protected

        class RecordingTransfer:
            name = "mooncake"
            enabled = True

            def __init__(self):
                self.calls = 0

            def transfer_pages(self, **kwargs):
                self.calls += 1

        transfer = RecordingTransfer()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = transfer
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 1,
                "target_session_id": "target-session",
                "target_page_indices": [-1],
                "target_kv_ptrs": [1000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertEqual(
            response["reason"],
            "malformed_transfer_request:target_page_index_out_of_range",
        )
        self.assertEqual(transfer.calls, 0)
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_rejects_target_kv_shape_before_te(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11])
        node.host_value = torch.tensor([100, 101], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        def fail_if_host_protected():
            raise AssertionError("target KV metadata should be validated before host lookup")

        node.protect_host = fail_if_host_protected

        class RecordingTransfer:
            name = "mooncake"
            enabled = True
            target_kv_item_lens = [64, 64]

            def __init__(self):
                self.calls = 0

            def transfer_pages(self, **kwargs):
                self.calls += 1

        transfer = RecordingTransfer()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = transfer
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 1,
                "target_session_id": "target-session",
                "target_page_indices": [3],
                "target_kv_ptrs": [1000, 2000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertIn("target_kv_ptrs_len_2", response["reason"])
        self.assertEqual(transfer.calls, 0)
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_rejects_target_kv_item_len_mismatch_before_te(
        self,
    ):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11])
        node.host_value = torch.tensor([100, 101], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        class RecordingTransfer:
            name = "mooncake"
            enabled = True
            target_kv_item_lens = [64, 64]

            def __init__(self):
                self.calls = 0

            def transfer_pages(self, **kwargs):
                self.calls += 1

        transfer = RecordingTransfer()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = transfer
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 1,
                "target_session_id": "target-session",
                "target_page_indices": [3],
                "target_kv_ptrs": [1000, 2000],
                "target_kv_item_lens": [64, 128],
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertIn("target_kv_item_lens_mismatch", response["reason"])
        self.assertEqual(transfer.calls, 0)
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_direct_transfer_rejects_target_session_metadata_mismatch(self):
        tree = FakeTree(page_size=2)
        node = TreeNode()
        node.parent = tree.root_node
        node.key = RadixKey([10, 11])
        node.host_value = torch.tensor([100, 101], dtype=torch.int64)
        node.hash_value = compute_node_hash_values(node, tree.page_size)
        tree.root_node.children[node.key.child_key(tree.page_size)] = node
        block_hashes = [hash_str_to_int64(hash_value) for hash_value in node.hash_value]

        class RecordingTransfer:
            name = "mooncake"
            enabled = True

            def __init__(self):
                self.calls = 0

            def transfer_pages(self, **kwargs):
                self.calls += 1

        transfer = RecordingTransfer()
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.direct_transfer = transfer
        handler.tree_cache = tree
        handler.worker_id = 7
        handler.dp_rank = 0

        response = handler._handle_source_transfer(
            {
                "transfer_backend": "mooncake",
                "plan": _make_plan(block_hashes),
                "start_block": 0,
                "max_blocks": 1,
                "target_session_id": "target-session",
                "target_metadata": {
                    "backend": "mooncake",
                    "session_id": "other-session",
                },
                "target_page_indices": [3],
                "target_kv_ptrs": [1000],
                "target_kv_item_lens": [64],
            }
        )

        self.assertFalse(response["ok"])
        self.assertEqual(response["pages"], [])
        self.assertEqual(
            response["reason"],
            "malformed_transfer_request:target_session_id_mismatch",
        )
        self.assertEqual(transfer.calls, 0)
        self.assertEqual(node.host_ref_counter, 0)

    def test_mooncake_transfer_backend_groups_source_and_target_pages(self):
        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        source_item_len = source_k[0].nbytes * page_size
        target_k_ptr = 1_000_000
        target_v_ptr = 2_000_000
        engine = FakeMooncakeEngine()
        tree = SimpleNamespace(
            page_size=page_size,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )
        backend = MooncakeG2plusTransferBackend(
            engine=engine,
            tree_cache=tree,
            target_kv_ptrs=[target_k_ptr, target_v_ptr],
            target_kv_item_lens=[source_item_len, source_item_len],
            transfer_parallelism=1,
        )

        backend.transfer_pages(
            target_session_id="peer-session",
            source_page_indices=torch.tensor([3, 4, 7], dtype=torch.int32).numpy(),
            target_page_indices=torch.tensor([10, 11, 12], dtype=torch.int32).numpy(),
            target_kv_ptrs=[target_k_ptr, target_v_ptr],
            target_kv_item_lens=[source_item_len, source_item_len],
        )

        self.assertEqual(len(engine.transfers), 1)
        session_id, src_addrs, dst_addrs, lengths = engine.transfers[0]
        self.assertEqual(session_id, "peer-session")
        self.assertEqual(
            src_addrs,
            [
                source_k.data_ptr() + 3 * source_item_len,
                source_k.data_ptr() + 7 * source_item_len,
                source_v.data_ptr() + 3 * source_item_len,
                source_v.data_ptr() + 7 * source_item_len,
            ],
        )
        self.assertEqual(
            dst_addrs,
            [
                target_k_ptr + 10 * source_item_len,
                target_k_ptr + 12 * source_item_len,
                target_v_ptr + 10 * source_item_len,
                target_v_ptr + 12 * source_item_len,
            ],
        )
        self.assertEqual(
            lengths,
            [
                source_item_len * 2,
                source_item_len,
                source_item_len * 2,
                source_item_len,
            ],
        )

    def test_mooncake_transfer_backend_can_split_transfer_batches(self):
        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        source_item_len = source_k[0].nbytes * page_size
        target_k_ptr = 1_000_000
        target_v_ptr = 2_000_000
        engine = FakeMooncakeEngine()
        tree = SimpleNamespace(
            page_size=page_size,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )
        with envs.SGLANG_G2PLUS_MOONCAKE_TRANSFER_PARALLELISM.override(2):
            backend = MooncakeG2plusTransferBackend(
                engine=engine,
                tree_cache=tree,
                target_kv_ptrs=[target_k_ptr, target_v_ptr],
                target_kv_item_lens=[source_item_len, source_item_len],
            )

        backend.transfer_pages(
            target_session_id="peer-session",
            source_page_indices=torch.tensor([3, 4, 7], dtype=torch.int32).numpy(),
            target_page_indices=torch.tensor([10, 11, 12], dtype=torch.int32).numpy(),
            target_kv_ptrs=[target_k_ptr, target_v_ptr],
            target_kv_item_lens=[source_item_len, source_item_len],
        )
        backend.shutdown()

        self.assertEqual(len(engine.transfers), 2)
        self.assertTrue(
            all(session_id == "peer-session" for session_id, _, _, _ in engine.transfers)
        )
        src_addrs = [addr for _, addrs, _, _ in engine.transfers for addr in addrs]
        dst_addrs = [addr for _, _, addrs, _ in engine.transfers for addr in addrs]
        lengths = [length for _, _, _, lens in engine.transfers for length in lens]
        self.assertCountEqual(
            src_addrs,
            [
                source_k.data_ptr() + 3 * source_item_len,
                source_k.data_ptr() + 7 * source_item_len,
                source_v.data_ptr() + 3 * source_item_len,
                source_v.data_ptr() + 7 * source_item_len,
            ],
        )
        self.assertCountEqual(
            dst_addrs,
            [
                target_k_ptr + 10 * source_item_len,
                target_k_ptr + 12 * source_item_len,
                target_v_ptr + 10 * source_item_len,
                target_v_ptr + 12 * source_item_len,
            ],
        )
        self.assertCountEqual(
            lengths,
            [
                source_item_len * 2,
                source_item_len,
                source_item_len * 2,
                source_item_len,
            ],
        )

    def test_mooncake_transfer_backend_drains_parallel_chunks_before_failure(self):
        class PartiallyFailingMooncakeEngine(FakeMooncakeEngine):
            def __init__(self):
                super().__init__()
                self._lock = threading.Lock()
                self._next_index = 0
                self.second_started = threading.Event()
                self.completed = []

            def batch_transfer_sync(self, session_id, src_addrs, dst_addrs, lengths):
                with self._lock:
                    call_index = self._next_index
                    self._next_index += 1
                    self.transfers.append((session_id, src_addrs, dst_addrs, lengths))
                if call_index == 0:
                    self.second_started.wait(timeout=1)
                    self.completed.append(call_index)
                    return 7
                self.second_started.set()
                time.sleep(0.05)
                self.completed.append(call_index)
                return 0

        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        source_item_len = source_k[0].nbytes * page_size
        engine = PartiallyFailingMooncakeEngine()
        tree = SimpleNamespace(
            page_size=page_size,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )
        backend = MooncakeG2plusTransferBackend(
            engine=engine,
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_item_lens=[source_item_len, source_item_len],
            transfer_parallelism=2,
        )

        with self.assertRaisesRegex(RuntimeError, "ret=7"):
            backend.transfer_pages(
                target_session_id="peer-session",
                source_page_indices=torch.tensor([3, 4, 7], dtype=torch.int32).numpy(),
                target_page_indices=torch.tensor(
                    [10, 11, 12], dtype=torch.int32
                ).numpy(),
                target_kv_ptrs=[1_000_000, 2_000_000],
                target_kv_item_lens=[source_item_len, source_item_len],
            )
        backend.shutdown()

        self.assertEqual(len(engine.transfers), 2)
        self.assertCountEqual(engine.completed, [0, 1])

    def test_mooncake_transfer_backend_reports_transport_status(self):
        source_item_len = 8
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                    v_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                )
            ),
        )
        with envs.SGLANG_G2PLUS_MOONCAKE_TRANSFER_PARALLELISM.override(3):
            backend = MooncakeG2plusTransferBackend(
                engine=FakeMooncakeEngine(ib_device="mlx5_0"),
                tree_cache=tree,
                target_kv_ptrs=[1, 2],
                target_kv_item_lens=[source_item_len, source_item_len],
            )

        self.assertEqual(
            backend.target_descriptor(),
            {
                "backend": "mooncake",
                "session_id": "target-session",
                "transport": {
                    "protocol": "rdma",
                    "ib_device": "mlx5_0",
                    "path_hint": "explicit_ib_device",
                    "transfer_parallelism": 3,
                },
            },
        )

        backend = MooncakeG2plusTransferBackend(
            engine=FakeMooncakeEngine(),
            tree_cache=tree,
            target_kv_ptrs=[1, 2],
            target_kv_item_lens=[source_item_len, source_item_len],
        )

        descriptor = backend.target_descriptor()
        self.assertIsNone(descriptor["transport"]["ib_device"])
        self.assertEqual(
            descriptor["transport"]["path_hint"], "no_explicit_ib_device"
        )

    def test_mooncake_transfer_backend_registers_source_host_pool(self):
        host_buffer = torch.zeros((32,), dtype=torch.uint8)
        engine = FakeMooncakeEngine()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    kv_buffer=host_buffer,
                    k_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                    v_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                )
            ),
        )
        backend = MooncakeG2plusTransferBackend(
            engine=engine,
            tree_cache=tree,
            target_kv_ptrs=[1, 2],
            target_kv_item_lens=[8, 8],
        )

        backend._register_source_host_pool()

        self.assertTrue(backend.enabled)
        self.assertEqual(engine.registered, [([host_buffer.data_ptr()], [32])])

    def test_mooncake_transfer_backend_rejects_host_pool_without_refs(self):
        host_buffer = torch.zeros((32,), dtype=torch.uint8)
        engine = FakeMooncakeEngine()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    kv_buffer=host_buffer,
                )
            ),
        )
        backend = MooncakeG2plusTransferBackend(
            engine=engine,
            tree_cache=tree,
            target_kv_ptrs=[1],
            target_kv_item_lens=[8],
        )

        backend._register_source_host_pool()

        self.assertFalse(backend.enabled)
        self.assertEqual(engine.registered, [])

    def test_mooncake_transfer_backend_rejects_host_target_shape_mismatch(self):
        host_buffer = torch.zeros((32,), dtype=torch.uint8)
        engine = FakeMooncakeEngine()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    kv_buffer=host_buffer,
                    k_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                    v_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                )
            ),
        )
        backend = MooncakeG2plusTransferBackend(
            engine=engine,
            tree_cache=tree,
            target_kv_ptrs=[1, 2],
            target_kv_item_lens=[8, 16],
        )

        backend._register_source_host_pool()

        self.assertFalse(backend.enabled)
        self.assertEqual(engine.registered, [])

    def test_mooncake_transfer_backend_shutdown_keeps_process_lifetime_registration(
        self,
    ):
        host_buffer = torch.zeros((32,), dtype=torch.uint8)
        engine = FakeMooncakeEngine()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    kv_buffer=host_buffer,
                    k_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                    v_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                )
            ),
        )
        backend = MooncakeG2plusTransferBackend(
            engine=engine,
            tree_cache=tree,
            target_kv_ptrs=[1, 2],
            target_kv_item_lens=[8, 8],
            target_registered=True,
        )
        backend._register_source_host_pool()

        backend.shutdown()
        backend.shutdown()

        self.assertFalse(backend.enabled)
        self.assertEqual(engine.deregistered, [])

    def test_nixl_transfer_backend_groups_source_host_to_target_gpu_pages(self):
        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        source_item_len = source_k[0].nbytes * page_size
        target_k_ptr = 1_000_000
        target_v_ptr = 2_000_000
        agent = FakeNixlAgent()
        tree = SimpleNamespace(
            page_size=page_size,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[target_k_ptr, target_v_ptr],
            target_kv_lens=[10_000, 10_000],
            target_kv_item_lens=[source_item_len, source_item_len],
            gpu_id=3,
            timeout_secs=1,
            transfer_parallelism=1,
        )
        target_metadata = {
            "agent_name": "target-agent",
            "agent_metadata_b64": base64.b64encode(b"target-metadata").decode("ascii"),
            "gpu_id": 3,
        }

        backend.transfer_pages(
            target_session_id="unused",
            source_page_indices=torch.tensor([3, 4, 7], dtype=torch.int32).numpy(),
            target_page_indices=torch.tensor([10, 11, 12], dtype=torch.int32).numpy(),
            target_kv_ptrs=[target_k_ptr, target_v_ptr],
            target_kv_item_lens=[source_item_len, source_item_len],
            target_metadata=target_metadata,
        )

        self.assertEqual(agent.remote_agents, [b"target-metadata"])
        self.assertEqual(len(agent.desc_calls), 2)
        src_type, src_reqs = agent.desc_calls[0]
        dst_type, dst_reqs = agent.desc_calls[1]
        self.assertEqual(src_type, "DRAM")
        self.assertEqual(dst_type, "VRAM")
        self.assertEqual(
            src_reqs.tolist(),
            [
                [source_k.data_ptr() + 3 * source_item_len, source_item_len * 2, 0],
                [source_k.data_ptr() + 7 * source_item_len, source_item_len, 0],
                [source_v.data_ptr() + 3 * source_item_len, source_item_len * 2, 0],
                [source_v.data_ptr() + 7 * source_item_len, source_item_len, 0],
            ],
        )
        self.assertEqual(
            dst_reqs.tolist(),
            [
                [target_k_ptr + 10 * source_item_len, source_item_len * 2, 3],
                [target_k_ptr + 12 * source_item_len, source_item_len, 3],
                [target_v_ptr + 10 * source_item_len, source_item_len * 2, 3],
                [target_v_ptr + 12 * source_item_len, source_item_len, 3],
            ],
        )
        self.assertEqual(agent.xfers[0][0], "WRITE")
        self.assertEqual(agent.xfers[0][3], "target-agent")
        self.assertEqual(agent.released_handles, ["handle-1"])

    def test_nixl_transfer_backend_can_post_parallel_transfers(self):
        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        source_item_len = source_k[0].nbytes * page_size
        target_k_ptr = 1_000_000
        target_v_ptr = 2_000_000
        agent = FakeNixlAgent()
        tree = SimpleNamespace(
            page_size=page_size,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[target_k_ptr, target_v_ptr],
            target_kv_lens=[10_000, 10_000],
            target_kv_item_lens=[source_item_len, source_item_len],
            gpu_id=3,
            timeout_secs=1,
            transfer_parallelism=2,
        )
        target_metadata = {
            "agent_name": "target-agent",
            "agent_metadata_b64": base64.b64encode(b"target-metadata").decode("ascii"),
            "gpu_id": 3,
        }

        backend.transfer_pages(
            target_session_id="unused",
            source_page_indices=torch.tensor([3, 4, 7], dtype=torch.int32).numpy(),
            target_page_indices=torch.tensor([10, 11, 12], dtype=torch.int32).numpy(),
            target_kv_ptrs=[target_k_ptr, target_v_ptr],
            target_kv_item_lens=[source_item_len, source_item_len],
            target_metadata=target_metadata,
        )

        self.assertEqual(len(agent.xfers), 2)
        self.assertEqual(len(agent.desc_calls), 4)
        self.assertEqual(agent.released_handles, ["handle-1", "handle-2"])
        self.assertTrue(all(xfer[0] == "WRITE" for xfer in agent.xfers))
        self.assertTrue(all(xfer[3] == "target-agent" for xfer in agent.xfers))

    def test_nixl_transfer_backend_drains_handles_after_error(self):
        class ErrorThenSlowDoneAgent(FakeNixlAgent):
            def __init__(self):
                super().__init__()
                self.checks = []
                self.handle_2_checks = 0

            def check_xfer_state(self, handle):
                self.checks.append(handle)
                if handle == "handle-1":
                    return "ERR"
                if handle == "handle-2":
                    self.handle_2_checks += 1
                    return "DONE" if self.handle_2_checks >= 2 else "PROC"
                return "DONE"

        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        source_item_len = source_k[0].nbytes * page_size
        agent = ErrorThenSlowDoneAgent()
        tree = SimpleNamespace(
            page_size=page_size,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_lens=[10_000, 10_000],
            target_kv_item_lens=[source_item_len, source_item_len],
            gpu_id=3,
            timeout_secs=1,
            transfer_parallelism=2,
        )
        target_metadata = {
            "agent_name": "target-agent",
            "agent_metadata_b64": base64.b64encode(b"target-metadata").decode("ascii"),
            "gpu_id": 3,
        }

        with self.assertRaisesRegex(RuntimeError, "encountered ERR"):
            backend.transfer_pages(
                target_session_id="unused",
                source_page_indices=torch.tensor([3, 4, 7], dtype=torch.int32).numpy(),
                target_page_indices=torch.tensor(
                    [10, 11, 12], dtype=torch.int32
                ).numpy(),
                target_kv_ptrs=[1_000_000, 2_000_000],
                target_kv_item_lens=[source_item_len, source_item_len],
                target_metadata=target_metadata,
            )

        self.assertEqual(agent.released_handles, ["handle-1", "handle-2"])
        self.assertGreaterEqual(agent.handle_2_checks, 2)

    def test_nixl_transfer_backend_reports_timeout_if_error_drain_does_not_finish(self):
        class ErrorThenStuckAgent(FakeNixlAgent):
            def __init__(self):
                super().__init__()
                self.checks = []

            def check_xfer_state(self, handle):
                self.checks.append(handle)
                if handle == "handle-1":
                    return "ERR"
                return "PROC"

        page_size = 2
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        source_item_len = source_k[0].nbytes * page_size
        agent = ErrorThenStuckAgent()
        tree = SimpleNamespace(
            page_size=page_size,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_lens=[10_000, 10_000],
            target_kv_item_lens=[source_item_len, source_item_len],
            gpu_id=3,
            timeout_secs=0,
            transfer_parallelism=2,
        )
        target_metadata = {
            "agent_name": "target-agent",
            "agent_metadata_b64": base64.b64encode(b"target-metadata").decode("ascii"),
            "gpu_id": 3,
        }

        with self.assertRaisesRegex(TimeoutError, "1 handles to drain") as cm:
            backend.transfer_pages(
                target_session_id="unused",
                source_page_indices=torch.tensor([3, 4, 7], dtype=torch.int32).numpy(),
                target_page_indices=torch.tensor(
                    [10, 11, 12], dtype=torch.int32
                ).numpy(),
                target_kv_ptrs=[1_000_000, 2_000_000],
                target_kv_item_lens=[source_item_len, source_item_len],
                target_metadata=target_metadata,
            )

        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        self.assertEqual(agent.released_handles, ["handle-1", "handle-2"])

    def test_nixl_transfer_backend_reports_transport_status(self):
        agent = FakeNixlAgent()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                    v_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[1, 2],
            target_kv_lens=[128, 256],
            target_kv_item_lens=[8, 8],
            gpu_id=3,
            timeout_secs=1,
            backend_name="UCX",
            transfer_parallelism=4,
        )

        descriptor = backend.target_descriptor()

        self.assertEqual(descriptor["backend"], "nixl")
        self.assertEqual(descriptor["agent_name"], "source-agent")
        self.assertEqual(descriptor["gpu_id"], 3)
        self.assertEqual(
            descriptor["transport"],
            {
                "backend": "UCX",
                "gpu_id": 3,
                "transfer_parallelism": 4,
            },
        )
        self.assertEqual(
            base64.b64decode(descriptor["agent_metadata_b64"]),
            b"source-metadata",
        )

    def test_g2plus_transfer_parallelism_prefers_backend_neutral_env(self):
        with envs.SGLANG_G2PLUS_TRANSFER_PARALLELISM.override(
            None
        ), envs.SGLANG_G2PLUS_MOONCAKE_TRANSFER_PARALLELISM.override(None):
            self.assertEqual(default_g2plus_transfer_parallelism(), 4)

        with envs.SGLANG_G2PLUS_TRANSFER_PARALLELISM.override(
            5
        ), envs.SGLANG_G2PLUS_MOONCAKE_TRANSFER_PARALLELISM.override(2):
            self.assertEqual(default_g2plus_transfer_parallelism(), 5)

        with envs.SGLANG_G2PLUS_TRANSFER_PARALLELISM.override(
            None
        ), envs.SGLANG_G2PLUS_MOONCAKE_TRANSFER_PARALLELISM.override(3):
            self.assertEqual(default_g2plus_transfer_parallelism(), 3)

    def test_nixl_thread_params_mirror_disagg_defaults(self):
        with envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.override(None):
            self.assertEqual(_default_nixl_num_threads(), 8)
        with envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.override(6):
            self.assertEqual(_default_nixl_num_threads(), 6)

        params = {}
        _apply_nixl_backend_thread_params("UCX", params, 6)
        self.assertEqual(params, {"num_threads": "6"})

        params = {"num_threads": "9"}
        _apply_nixl_backend_thread_params("UCX", params, 6)
        self.assertEqual(params, {"num_threads": "9"})

        params = {}
        _apply_nixl_backend_thread_params("GDS_MT", params, 6)
        self.assertEqual(params, {"thread_count": "6"})

        params = {}
        _apply_nixl_backend_thread_params("UCCL", params, 6)
        self.assertEqual(params, {"num_cpus": "6"})

    def test_nixl_transfer_backend_rejects_invalid_target_metadata_before_te(self):
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        source_item_len = source_k[0].nbytes * 2
        agent = FakeNixlAgent()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_lens=[10_000, 10_000],
            target_kv_item_lens=[source_item_len, source_item_len],
            gpu_id=3,
            timeout_secs=1,
        )

        with self.assertRaisesRegex(RuntimeError, "invalid agent_metadata_b64"):
            backend.transfer_pages(
                target_session_id="unused",
                source_page_indices=torch.tensor([3], dtype=torch.int32).numpy(),
                target_page_indices=torch.tensor([10], dtype=torch.int32).numpy(),
                target_kv_ptrs=[1_000_000, 2_000_000],
                target_kv_item_lens=[source_item_len, source_item_len],
                target_metadata={
                    "agent_name": "target-agent",
                    "agent_metadata_b64": "not-base64!!",
                    "gpu_id": 3,
                },
            )

        self.assertEqual(agent.remote_agents, [])
        self.assertEqual(agent.desc_calls, [])
        self.assertEqual(agent.xfers, [])

    def test_nixl_transfer_backend_rejects_invalid_target_gpu_before_te(self):
        source_k = torch.zeros((20, 4), dtype=torch.uint8)
        source_v = torch.zeros((20, 4), dtype=torch.uint8)
        source_item_len = source_k[0].nbytes * 2
        agent = FakeNixlAgent()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    k_data_refs=[source_k],
                    v_data_refs=[source_v],
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[1_000_000, 2_000_000],
            target_kv_lens=[10_000, 10_000],
            target_kv_item_lens=[source_item_len, source_item_len],
            gpu_id=3,
            timeout_secs=1,
        )

        with self.assertRaisesRegex(RuntimeError, "gpu_id out of range"):
            backend.transfer_pages(
                target_session_id="unused",
                source_page_indices=torch.tensor([3], dtype=torch.int32).numpy(),
                target_page_indices=torch.tensor([10], dtype=torch.int32).numpy(),
                target_kv_ptrs=[1_000_000, 2_000_000],
                target_kv_item_lens=[source_item_len, source_item_len],
                target_metadata={
                    "agent_name": "target-agent",
                    "agent_metadata_b64": base64.b64encode(
                        b"target-metadata"
                    ).decode("ascii"),
                    "gpu_id": -1,
                },
            )

        self.assertEqual(agent.remote_agents, [])
        self.assertEqual(agent.desc_calls, [])
        self.assertEqual(agent.xfers, [])

    def test_nixl_transfer_backend_shutdown_unregisters_memory(self):
        host_buffer = torch.zeros((32,), dtype=torch.uint8)
        agent = FakeNixlAgent()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    kv_buffer=host_buffer,
                    k_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                    v_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[1, 2],
            target_kv_lens=[128, 256],
            target_kv_item_lens=[8, 8],
            gpu_id=3,
            timeout_secs=1,
        )
        backend._register_target_device_pool()
        backend._register_source_host_pool()

        backend.shutdown()
        backend.shutdown()

        self.assertFalse(backend.enabled)
        self.assertEqual(
            agent.unregistered,
            [
                [("DRAM", [(host_buffer.data_ptr(), host_buffer.nbytes, 0, "")])],
                [("VRAM", [(1, 128, 3, ""), (2, 256, 3, "")])],
            ],
        )

    def test_nixl_transfer_backend_rejects_host_pool_without_refs(self):
        host_buffer = torch.zeros((32,), dtype=torch.uint8)
        agent = FakeNixlAgent()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    kv_buffer=host_buffer,
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[1],
            target_kv_lens=[128],
            target_kv_item_lens=[8],
            gpu_id=3,
            timeout_secs=1,
        )

        backend._register_source_host_pool()

        self.assertFalse(backend.enabled)
        self.assertEqual(agent.registered, [])

    def test_nixl_transfer_backend_rejects_host_target_shape_mismatch(self):
        host_buffer = torch.zeros((32,), dtype=torch.uint8)
        agent = FakeNixlAgent()
        tree = SimpleNamespace(
            page_size=2,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(
                    layout="layer_first",
                    kv_buffer=host_buffer,
                    k_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                    v_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                )
            ),
        )
        backend = NixlG2plusTransferBackend(
            agent=agent,
            tree_cache=tree,
            target_kv_ptrs=[1, 2],
            target_kv_lens=[128, 256],
            target_kv_item_lens=[8, 16],
            gpu_id=3,
            timeout_secs=1,
        )

        backend._register_source_host_pool()

        self.assertFalse(backend.enabled)
        self.assertEqual(agent.registered, [])

    def test_explicit_g2plus_direct_backend_fails_fast_when_unavailable(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(g2plus_transfer_backend="mooncake")
        )
        with patch.object(
            MooncakeG2plusTransferBackend, "from_scheduler", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "Mooncake"):
                make_g2plus_transfer_backend(scheduler)

        scheduler.server_args.g2plus_transfer_backend = "nixl"
        with patch.object(
            NixlG2plusTransferBackend, "from_scheduler", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "NIXL"):
                make_g2plus_transfer_backend(scheduler)

    def test_explicit_g2plus_direct_backend_error_includes_diagnostics(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(g2plus_transfer_backend="mooncake")
        )

        def unavailable(_scheduler, diagnostics=None):
            diagnostics.append("mooncake target KV registration failed")
            return None

        with patch.object(
            MooncakeG2plusTransferBackend, "from_scheduler", side_effect=unavailable
        ):
            with self.assertRaisesRegex(
                RuntimeError, "mooncake target KV registration failed"
            ):
                make_g2plus_transfer_backend(scheduler, diagnostics=[])

    def test_explicit_g2plus_direct_backend_rejects_unsupported_topology(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                g2plus_transfer_backend="mooncake",
                tp_size=2,
                pp_size=1,
                attn_cp_size=1,
            )
        )
        with patch.object(
            MooncakeG2plusTransferBackend,
            "from_scheduler",
            side_effect=AssertionError("must reject before initializing backend"),
        ):
            with self.assertRaisesRegex(RuntimeError, "tp_size=2"):
                make_g2plus_transfer_backend(scheduler)

    def test_auto_g2plus_direct_backend_rejects_unsupported_topology_without_init(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                g2plus_transfer_backend="auto",
                tp_size=1,
                pp_size=2,
                attn_cp_size=1,
            )
        )
        with patch.object(
            MooncakeG2plusTransferBackend, "from_scheduler", return_value=None
        ) as mooncake_from_scheduler, patch.object(
            NixlG2plusTransferBackend, "from_scheduler", return_value=None
        ) as nixl_from_scheduler:
            self.assertIsNone(make_g2plus_transfer_backend(scheduler))

        mooncake_from_scheduler.assert_not_called()
        nixl_from_scheduler.assert_not_called()

    def test_auto_g2plus_backend_without_direct_te_returns_none(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(g2plus_transfer_backend="auto")
        )
        with patch.object(
            MooncakeG2plusTransferBackend, "from_scheduler", return_value=None
        ), patch.object(NixlG2plusTransferBackend, "from_scheduler", return_value=None):
            self.assertIsNone(make_g2plus_transfer_backend(scheduler))

    def test_auto_g2plus_backend_records_unavailable_diagnostics(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(g2plus_transfer_backend="auto")
        )
        diagnostics = []

        with patch.object(
            MooncakeG2plusTransferBackend, "from_scheduler", return_value=None
        ) as mooncake_from_scheduler, patch.object(
            NixlG2plusTransferBackend, "from_scheduler", return_value=None
        ) as nixl_from_scheduler:
            self.assertIsNone(
                make_g2plus_transfer_backend(scheduler, diagnostics=diagnostics)
            )

        mooncake_from_scheduler.assert_called_once_with(
            scheduler, diagnostics=diagnostics
        )
        nixl_from_scheduler.assert_called_once_with(scheduler, diagnostics=diagnostics)
        self.assertEqual(diagnostics, ["mooncake unavailable", "nixl unavailable"])

    def test_remote_g2_from_scheduler_logs_transfer_diagnostics(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                enable_router_kv_reuse=True,
                enable_g2plus=False,
                g2plus_worker_id=42,
                g2plus_timeout_secs=1,
                g2plus_transfer_backend="auto",
                g2plus_endpoint=None,
                g2plus_peer_endpoints=None,
                g2plus_fetch_workers=1,
            ),
            enable_hierarchical_cache=True,
            tree_cache=FakeTree(page_size=2),
            dp_rank=0,
            enable_metrics=False,
        )

        def make_backend(_scheduler, diagnostics=None):
            diagnostics.append("mooncake initialization failed: missing transfer engine")
            diagnostics.append("nixl initialization failed: missing nixl")
            return None

        with patch(
            "sglang.srt.mem_cache.router_kv_reuse.make_g2plus_transfer_backend",
            side_effect=make_backend,
        ), self.assertLogs(
            "sglang.srt.mem_cache.router_kv_reuse", level="WARNING"
        ) as logs:
            handler = RemoteG2ReuseHandler.from_scheduler(scheduler)
            self.assertIsNotNone(handler)
            handler.shutdown()

        self.assertIn(
            "Diagnostics: mooncake initialization failed: missing transfer engine; "
            "nixl initialization failed: missing nixl",
            "\n".join(logs.output),
        )

    def test_remote_g2_from_scheduler_requires_worker_identity(self):
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                enable_router_kv_reuse=True,
                enable_g2plus=False,
                g2plus_worker_id=None,
            ),
            enable_hierarchical_cache=True,
            tree_cache=FakeTree(page_size=2),
        )

        with patch(
            "sglang.srt.mem_cache.router_kv_reuse.make_g2plus_transfer_backend",
            side_effect=AssertionError("must not initialize transfer backend"),
        ):
            self.assertIsNone(RemoteG2ReuseHandler.from_scheduler(scheduler))

    def test_server_args_cli_wires_g2plus_config(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--g2plus-config",
                json.dumps(
                    {
                        "worker_id": 7,
                        "control": {"backend": "http"},
                        "http_control": {
                            "endpoint": "127.0.0.1:39007",
                            "static_peer_endpoints": {"8": "127.0.0.1:39008"},
                        },
                        "transfer": {
                            "backend": "nixl",
                            "timeout_secs": 2.5,
                        },
                    }
                ),
            ]
        )
        server_args = ServerArgs.from_cli_args(args)

        self.assertTrue(server_args.enable_router_kv_reuse)
        self.assertTrue(server_args.enable_hierarchical_cache)
        self.assertEqual(server_args.g2plus_config["worker_id"], 7)
        self.assertEqual(server_args.g2plus_config["control_backend"], "http")
        self.assertEqual(
            server_args.g2plus_config["http_control"]["endpoint"],
            "127.0.0.1:39007",
        )
        self.assertEqual(
            server_args.g2plus_config["http_control"]["static_peer_endpoints"],
            {"8": "127.0.0.1:39008"},
        )
        self.assertEqual(server_args.g2plus_config["timeout_secs"], 2.5)
        self.assertEqual(server_args.g2plus_config["transfer_backend"], "nixl")

    def test_server_args_rejects_g2plus_config_without_worker_id(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(["--model-path", "dummy", "--g2plus-config", "{}"])

        with self.assertRaisesRegex(
            ValueError, "--g2plus-config requires worker_id"
        ):
            ServerArgs.from_cli_args(args)

    def test_server_args_allows_generic_router_reuse_without_g2plus_worker_id(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(["--model-path", "dummy"])
        args.enable_router_kv_reuse = True
        server_args = ServerArgs.from_cli_args(args)

        self.assertTrue(server_args.enable_router_kv_reuse)
        self.assertIsNone(server_args.g2plus_config)

    def test_server_args_rejects_static_peer_endpoints_outside_http_control(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--g2plus-config",
                '{"worker_id": 7, "peer_endpoints": {"8": "127.0.0.1:39008"}}',
            ]
        )

        with self.assertRaisesRegex(
            ValueError, "g2plus_config.peer_endpoints is not supported"
        ):
            ServerArgs.from_cli_args(args)

    def test_server_args_rejects_invalid_g2plus_http_endpoint_values(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--g2plus-config",
                '{"worker_id": 7, "http_control": {"endpoint": {"0": null}}}',
            ]
        )

        with self.assertRaisesRegex(
            ValueError, "g2plus_config.http_control.endpoint values must be non-empty strings"
        ):
            ServerArgs.from_cli_args(args)

    def test_remote_g2_handler_rejects_invalid_dp_endpoint_values(self):
        server_args = SimpleNamespace(
            g2plus_config={
                "timeout_secs": 1,
                "transfer_backend": "auto",
                "http_control": {"endpoint": {"0": 123}},
            },
        )

        with self.assertRaisesRegex(
            ValueError, "g2plus_config.endpoint values must be non-empty strings"
        ):
            RemoteG2ReuseHandler(
                server_args=server_args,
                tree_cache=FakeTree(page_size=2),
                worker_id=42,
                dp_rank=0,
                direct_transfer=None,
            )

    def test_mooncake_transfer_parallelism_can_come_from_server_args(self):
        engine = FakeMooncakeEngine()
        source_item_len = 8
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                g2plus_transfer_backend="mooncake",
                mooncake_ib_device=None,
                g2plus_transfer_parallelism=4,
                tp_size=1,
                pp_size=1,
                attn_cp_size=1,
            ),
            gpu_id=0,
            tp_rank=0,
            tree_cache=SimpleNamespace(
                page_size=2,
                cache_controller=SimpleNamespace(
                    mem_pool_host=SimpleNamespace(
                        layout="layer_first",
                        kv_buffer=torch.zeros((32,), dtype=torch.uint8),
                        k_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                        v_data_refs=[torch.zeros((2, 4), dtype=torch.uint8)],
                    )
                ),
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(
                    get_contiguous_buf_infos=lambda: ([1, 2], [128, 128], [source_item_len, source_item_len])
                )
            ),
        )

        with patch(
            "sglang.srt.distributed.parallel_state.get_mooncake_transfer_engine",
            return_value=engine,
        ):
            backend = MooncakeG2plusTransferBackend.from_scheduler(scheduler)

        self.assertIsNotNone(backend)
        self.assertEqual(
            backend.target_descriptor()["transport"]["transfer_parallelism"], 4
        )
        backend.shutdown()

    def test_scheduler_initializes_router_reuse_manager(self):
        manager = object()
        scheduler = SimpleNamespace()

        with patch(
            "sglang.srt.managers.scheduler.RouterKVReuseManager.from_scheduler",
            return_value=manager,
        ) as from_scheduler:
            Scheduler.init_router_kv_reuse(scheduler)

        from_scheduler.assert_called_once_with(scheduler)
        self.assertIs(scheduler.router_kv_reuse_manager, manager)

    def test_router_manager_dispatches_through_generic_interface(self):
        req = object()
        handler = SimpleNamespace(maybe_stage_remote_prefix=lambda request: 2)
        manager = RouterKVReuseManager([handler])

        self.assertEqual(manager.maybe_stage_reuse_plan(req), 2)

    def test_router_manager_reports_pending_reuse(self):
        req = object()
        handler = SimpleNamespace(
            check_remote_prefix=lambda request: RouterKVReuseResult(pending=True)
        )
        manager = RouterKVReuseManager([handler])

        result = manager.check_reuse_plan_progress(req)

        self.assertTrue(result.pending)

    def test_router_manager_prefetches_through_generic_interface(self):
        req = object()
        handler = SimpleNamespace(
            prefetch_remote_prefix=lambda request: RouterKVReuseResult(pending=True)
        )
        manager = RouterKVReuseManager([handler])

        result = manager.prefetch_reuse_plan(req)

        self.assertTrue(result.pending)

    def test_router_manager_reports_pending_handlers(self):
        manager = RouterKVReuseManager(
            [
                SimpleNamespace(has_pending=lambda: False),
                SimpleNamespace(has_pending=lambda: True),
            ]
        )

        self.assertTrue(manager.has_pending())

    def test_router_manager_treats_pending_check_error_as_pending(self):
        def raise_has_pending():
            raise RuntimeError("pending check failed")

        manager = RouterKVReuseManager([SimpleNamespace(has_pending=raise_has_pending)])

        self.assertTrue(manager.has_pending())

    def test_router_manager_shutdown_continues_after_handler_error(self):
        calls = []

        def raise_shutdown():
            calls.append("bad")
            raise RuntimeError("shutdown failed")

        manager = RouterKVReuseManager(
            [
                SimpleNamespace(shutdown=raise_shutdown),
                SimpleNamespace(shutdown=lambda: calls.append("good")),
            ]
        )

        manager.shutdown()

        self.assertEqual(calls, ["bad", "good"])

    def test_router_manager_release_request_continues_after_handler_error(self):
        calls = []

        def raise_release(rid):
            calls.append(("bad", rid))
            raise RuntimeError("release failed")

        manager = RouterKVReuseManager(
            [
                SimpleNamespace(release_request=raise_release),
                SimpleNamespace(release_request=lambda rid: calls.append(("good", rid))),
            ]
        )

        manager.release_request("rid-1")

        self.assertEqual(calls, [("bad", "rid-1"), ("good", "rid-1")])

    def test_scheduler_idle_waits_for_pending_router_reuse(self):
        empty_batch = SimpleNamespace(is_empty=lambda: True)
        scheduler = SimpleNamespace(
            running_batch=empty_batch,
            chunked_req=None,
            dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
            last_batch=None,
            cur_batch=None,
            enable_overlap=False,
            result_queue=[],
            pp_size=1,
            waiting_queue=[],
            grammar_manager=SimpleNamespace(grammar_queue=[]),
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hisparse=False,
            enable_hierarchical_cache=True,
            tree_cache=SimpleNamespace(
                ongoing_write_through={},
                ongoing_load_back={},
                enable_storage=False,
            ),
            router_kv_reuse_manager=SimpleNamespace(has_pending=lambda: True),
        )

        self.assertFalse(Scheduler.is_fully_idle(scheduler))

    def test_scheduler_shutdown_releases_router_reuse_manager(self):
        shutdown_calls = []
        scheduler = SimpleNamespace(
            router_kv_reuse_manager=SimpleNamespace(
                shutdown=lambda: shutdown_calls.append("shutdown")
            )
        )

        Scheduler.shutdown(scheduler)
        Scheduler.shutdown(scheduler)

        self.assertEqual(shutdown_calls, ["shutdown"])
        self.assertTrue(scheduler._shutdown_called)

    def test_scheduler_prepares_router_reuse_before_prefill_schedule(self):
        calls = []
        req = SimpleNamespace(
            rid="r-prepare",
            init_next_round_input=lambda tree_cache, cow_mamba=None: calls.append(
                (tree_cache, cow_mamba)
            )
        )
        scheduler = SimpleNamespace(
            tree_cache="tree",
            router_kv_reuse_manager=SimpleNamespace(
                has_reuse_plan=lambda request: True,
                check_reuse_plan_progress=lambda request: RouterKVReuseResult(
                    pending=True
                ),
            ),
        )

        self.assertFalse(Scheduler._prepare_router_kv_reuse_for_schedule(scheduler, req))
        self.assertEqual(calls, [("tree", False)])

    def test_scheduler_router_reuse_error_falls_back_to_local_prefill(self):
        calls = []
        release_calls = []
        req = SimpleNamespace(
            rid="r-router-error",
            remote_kv_reuse_plan={"plan_id": "plan-1"},
            init_next_round_input=lambda tree_cache, cow_mamba=None: calls.append(
                (tree_cache, cow_mamba)
            ),
        )

        def raise_progress(request):
            raise RuntimeError("router reuse failed")

        scheduler = SimpleNamespace(
            tree_cache="tree",
            router_kv_reuse_manager=SimpleNamespace(
                has_reuse_plan=lambda request: True,
                check_reuse_plan_progress=raise_progress,
                release_request=lambda rid: release_calls.append(rid),
            ),
        )

        self.assertTrue(Scheduler._prepare_router_kv_reuse_for_schedule(scheduler, req))
        self.assertEqual(calls, [("tree", False)])
        self.assertIsNone(req.remote_kv_reuse_plan)
        self.assertEqual(release_calls, ["r-router-error"])

    def test_scheduler_router_reuse_plan_probe_error_falls_back_to_local_prefill(self):
        release_calls = []
        req = SimpleNamespace(
            rid="r-router-probe-error",
            remote_kv_reuse_plan={"plan_id": "plan-1"},
            init_next_round_input=lambda tree_cache, cow_mamba=None: (_ for _ in ()).throw(
                AssertionError("prefix probe should not run after has_reuse_plan failure")
            ),
        )

        def raise_has_reuse_plan(request):
            raise RuntimeError("router plan probe failed")

        scheduler = SimpleNamespace(
            tree_cache="tree",
            router_kv_reuse_manager=SimpleNamespace(
                has_reuse_plan=raise_has_reuse_plan,
                release_request=lambda rid: release_calls.append(rid),
            ),
        )

        self.assertTrue(Scheduler._prepare_router_kv_reuse_for_schedule(scheduler, req))
        self.assertIsNone(req.remote_kv_reuse_plan)
        self.assertEqual(release_calls, ["r-router-probe-error"])

    def test_scheduler_releases_router_reuse_for_aborted_request(self):
        release_calls = []
        scheduler = SimpleNamespace(
            router_kv_reuse_manager=SimpleNamespace(
                release_request=lambda rid: release_calls.append(rid)
            )
        )

        Scheduler._release_router_kv_reuse_request(scheduler, "r-abort")

        self.assertEqual(release_calls, ["r-abort"])

    def test_finished_request_releases_router_reuse_state(self):
        release_calls = []
        release_kv_calls = []
        req = SimpleNamespace(
            rid="r-finished",
            finished=lambda: True,
            multimodal_inputs=None,
            session=None,
            time_stats=SimpleNamespace(set_completion_time=lambda: None),
        )
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            ),
            maybe_collect_routed_experts=lambda request: None,
            maybe_collect_indexer_topk=lambda request: None,
            maybe_collect_customized_info=lambda i, request, logits_output: None,
            enable_hisparse=False,
            tree_cache=object(),
            router_kv_reuse_manager=SimpleNamespace(
                release_request=lambda rid: release_calls.append(rid)
            ),
        )

        with patch(
            "sglang.srt.managers.scheduler_output_processor_mixin.release_kv_cache",
            lambda request, tree_cache: release_kv_calls.append(request.rid),
        ):
            SchedulerOutputProcessorMixin._handle_finished_req(scheduler, req, 0, None)

        self.assertEqual(release_kv_calls, ["r-finished"])
        self.assertEqual(release_calls, ["r-finished"])

    def test_cached_tokens_details_separates_remote_g2_from_device(self):
        scheduler = SimpleNamespace(enable_hicache_storage=False)
        req = SimpleNamespace(
            cached_tokens=12,
            cached_tokens_device=8,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            cached_tokens_remote_g2=4,
        )

        details = SchedulerOutputProcessorMixin._get_cached_tokens_details(
            scheduler, req
        )

        self.assertEqual(details, {"device": 8, "host": 0, "remote_g2": 4})

    def test_openai_cached_tokens_details_preserves_remote_g2(self):
        details = cached_tokens_details_from_dict(
            {"device": 8, "host": 0, "remote_g2": 4}
        )

        self.assertEqual(
            details.model_dump(exclude_none=True),
            {"device": 8, "host": 0, "remote_g2": 4},
        )

    def test_scheduler_metrics_collector_observes_router_kv_reuse(self):
        collector = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        collector.labels = {"model_name": "dummy"}
        collector.router_kv_reuse_events_total = FakePrometheusMetric()
        collector.router_kv_reuse_tokens_total = FakePrometheusMetric()
        collector.router_kv_reuse_wait_ms = FakePrometheusMetric()
        collector.router_kv_reuse_insert_ms = FakePrometheusMetric()
        collector.router_kv_reuse_transfer_mb = FakePrometheusMetric()

        collector.observe_router_kv_reuse(
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
        self.assertEqual(
            collector.router_kv_reuse_events_total.calls,
            [("inc", labels, 1)],
        )
        self.assertEqual(
            collector.router_kv_reuse_tokens_total.calls,
            [("inc", labels, 4)],
        )
        self.assertEqual(
            collector.router_kv_reuse_wait_ms.calls,
            [("observe", labels, 12.5)],
        )
        self.assertEqual(
            collector.router_kv_reuse_insert_ms.calls,
            [("observe", labels, 0.25)],
        )
        self.assertEqual(
            collector.router_kv_reuse_transfer_mb.calls,
            [("observe", labels, 2.0)],
        )

    def test_scheduler_metrics_collector_observes_router_kv_reuse_quarantine(self):
        collector = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        collector.labels = {"model_name": "dummy"}
        collector.router_kv_reuse_quarantine_events_total = FakePrometheusMetric()
        collector.router_kv_reuse_quarantined_tokens_total = FakePrometheusMetric()
        collector.router_kv_reuse_quarantined_tokens = FakePrometheusMetric()

        collector.observe_router_kv_reuse_quarantine(
            backend="mooncake",
            reason="source_transfer_timeout_maybe_inflight",
            tokens=4,
            current_tokens=6,
        )

        event_labels = {
            "model_name": "dummy",
            "backend": "mooncake",
            "reason": "source_transfer_timeout_maybe_inflight",
        }
        gauge_labels = {
            "model_name": "dummy",
            "backend": "mooncake",
        }
        self.assertEqual(
            collector.router_kv_reuse_quarantine_events_total.calls,
            [("inc", event_labels, 1)],
        )
        self.assertEqual(
            collector.router_kv_reuse_quarantined_tokens_total.calls,
            [("inc", event_labels, 4)],
        )
        self.assertEqual(
            collector.router_kv_reuse_quarantined_tokens.calls,
            [("set", gauge_labels, 6)],
        )

        collector.observe_router_kv_reuse_quarantine(
            backend="mooncake",
            reason="released",
            tokens=0,
            current_tokens=0,
        )

        self.assertEqual(
            collector.router_kv_reuse_quarantine_events_total.calls,
            [("inc", event_labels, 1)],
        )
        self.assertEqual(
            collector.router_kv_reuse_quarantined_tokens.calls[-1],
            ("set", gauge_labels, 0),
        )

    def test_remote_g2_handler_falls_back_to_all_static_peer_endpoints(self):
        handler = RemoteG2ReuseHandler.__new__(RemoteG2ReuseHandler)
        handler.worker_id = None
        handler.dp_rank = 0
        handler.endpoint = None
        handler.static_peer_endpoints = {
            "11": "http://127.0.0.1:39011",
            "22": "http://127.0.0.1:39022",
        }
        plan = RemoteKvReusePlan.from_dict(_make_plan([1], source_worker_id=99))

        self.assertEqual(
            handler._candidate_endpoints_for_plan(plan),
            ["http://127.0.0.1:39011", "http://127.0.0.1:39022"],
        )

    def test_generate_req_batch_preserves_remote_plan_metadata(self):
        plan_0 = _make_plan([1])
        plan_1 = _make_plan([2], plan_id="plan-2")
        req = GenerateReqInput(
            text=["hello", "world"],
            sampling_params=[{}, {}],
            rid=["r0", "r1"],
            remote_kv_reuse_plan=[plan_0, plan_1],
        )

        req.normalize_batch_and_arguments()

        self.assertEqual(req[0].remote_kv_reuse_plan, plan_0)
        self.assertEqual(req[1].remote_kv_reuse_plan, plan_1)


if __name__ == "__main__":
    unittest.main()
