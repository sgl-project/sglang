import time
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.hicache_host_index import HiCacheHostBlockIndex
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.shared_hicache.manager import SharedHiCacheManager
from sglang.srt.mem_cache.shared_hicache.plan import SharedHiCachePlan
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
from sglang.srt.mem_cache.utils import block_hash_aliases, hash_str_to_int64


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


class FakeDeviceAllocator:
    def __init__(self):
        self.fail_alloc = False

    def alloc(self, need_size):
        if self.fail_alloc:
            return None
        return torch.arange(200, 200 + need_size)

    def free(self, indices):
        return len(indices)


class FakeHostPool:
    def __init__(self):
        self.pages = {}

    def get_data_page(self, index, flat=True):
        return self.pages[int(index)]


class FakeTree:
    def __init__(self, page_size=2):
        self.page_size = page_size
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(array("q"))
        self.device_allocator = FakeDeviceAllocator()
        self.cache_controller = SimpleNamespace(
            mem_pool_host=FakeHostPool(),
            mem_pool_device_allocator=self.device_allocator,
        )
        self.evict_count = 0

    def evict(self, params):
        self.evict_count += 1
        return SimpleNamespace(num_tokens_evicted=0)


class FakeTransferBackend:
    name = "nixl"
    enabled = True
    target_session_id = "target-session"
    target_kv_ptrs = [1]
    target_kv_item_lens = [64]


class FakeSharedHiCacheReq:
    def __init__(self, rid, *, local_prefix_len=0):
        self.rid = rid
        self.shared_hicache_plan = True
        self.shared_hicache_max_prefix_len = None
        self.local_prefix_len = local_prefix_len
        self.host_hit_length = 0
        self.prefix_indices = torch.arange(local_prefix_len, dtype=torch.int64)
        self.init_calls = []

    def init_next_round_input(self, tree_cache=None, cow_mamba=None):
        self.init_calls.append(self.shared_hicache_max_prefix_len)
        prefix_len = self.local_prefix_len
        if self.shared_hicache_max_prefix_len is not None:
            prefix_len = min(prefix_len, self.shared_hicache_max_prefix_len)
        self.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)


class FakeScheduleManager:
    def __init__(self, prefix_len):
        self.prefix_len = prefix_len
        self.prepared = []

    def has_reuse_plan(self, req):
        return True

    def prepare_reuse(self, req):
        self.prepared.append(req.rid)
        return SimpleNamespace(pending=False, prefix_len=self.prefix_len)

    def release_request(self, rid):
        pass


class FakeScheduler(SharedHiCacheSchedulerMixin):
    def __init__(self, manager):
        self.shared_hicache_manager = manager
        self.ps = SimpleNamespace(tp_size=1)
        self.tree_cache = object()
        self.server_args = SimpleNamespace(prefill_max_requests=None)
        self.chunked_req = None
        self.enable_priority_preemption = False

    def get_num_allocatable_reqs(self, running_bs):
        return 1


def _make_manager():
    manager = SharedHiCacheManager.__new__(SharedHiCacheManager)
    manager.worker_id = 42
    manager.tree_cache = FakeTree(page_size=2)
    manager.source_endpoints = {}
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
    return manager


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
        tree = FakeTree(page_size=2)
        tree.cache_controller.mem_pool_host.pages[100] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8
        )

        def lookup(wanted_hashes, *, protect=False):
            self.assertEqual(set(wanted_hashes), {kv_hash})
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
        self.assertEqual(
            pages, [ResolvedHostPage(identity_hash, "aa" * 32, bytes([1, 2, 3, 4]))]
        )
        self.assertEqual(node.host_ref_counter, 0)

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
            transfer_backend=FakeTransferBackend(),
            tree_cache=FakeTree(),
        )
        self.assertIsNone(error)

        response = execute_source_transfer_request(
            request=request,
            transfer_backend=FakeTransferBackend(),
            tree_cache=FakeTree(),
            worker_id=7,
            tp_rank=0,
            tp_size=2,
            attn_tp_size=2,
        )

        self.assertFalse(response["ok"])
        self.assertIn("wrong_source_tp_rank_for_target", response["reason"])

    def test_target_direct_transfer_allocation_does_not_evict(self):
        tree = FakeTree()
        tree.device_allocator.fail_alloc = True
        target = SharedHiCacheTarget(tree_cache=tree, metrics_collector=None)

        self.assertIsNone(target.alloc_device_indices(4))
        self.assertEqual(tree.evict_count, 0)

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

    def test_manager_validates_target_tp_rank(self):
        manager = _make_manager()
        wrong_rank = SharedHiCachePlan.from_dict(
            _make_plan(
                [11],
                source_tp_rank=0,
                source_tp_size=2,
                target_tp_rank=0,
                target_tp_size=2,
            )
        )
        rank_generic = SharedHiCachePlan.from_dict(
            _make_plan([11], source_tp_size=2, target_tp_size=2)
        )

        self.assertEqual(
            manager._validate_plan(wrong_rank),
            "wrong_target_tp_rank:plan=0:local=1",
        )
        self.assertIsNone(manager._validate_plan(rank_generic))

    def test_scheduler_keeps_longer_local_prefix(self):
        manager = FakeScheduleManager(prefix_len=8)
        scheduler = FakeScheduler(manager)
        req = FakeSharedHiCacheReq("rid-1", local_prefix_len=24)

        pending_rids = scheduler._prepare_shared_hicache_for_schedule_batch([req])

        self.assertEqual(pending_rids, set())
        self.assertEqual(manager.prepared, ["rid-1"])
        self.assertEqual(req.shared_hicache_max_prefix_len, 24)
        self.assertEqual(req.init_calls, [None, 24])


if __name__ == "__main__":
    unittest.main()
