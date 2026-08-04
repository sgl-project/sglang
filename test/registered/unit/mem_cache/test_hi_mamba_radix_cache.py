"""CPU unit tests for ``srt/mem_cache/hi_mamba_radix_cache.py``.

The production cache owns GPU/host pools and asynchronous I/O workers.  These
tests keep those boundaries as small recording fakes while exercising the real
radix nodes, LRU lists, reference accounting, and tier-transition logic.
"""

import json
import tempfile
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from numpy import float64

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.hi_mamba_radix_cache import (
    HiMambaRadixCache,
    HostLRUList,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PrefetchTimeoutConfig,
)
from sglang.srt.mem_cache.mamba_radix_cache import LRUList, TreeNode
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _indices(values):
    return torch.tensor(values, dtype=torch.int64)


def _reset_tree_node_counters():
    TreeNode.counter = 0
    TreeNode.last_access_time_counter_float = float64(1.0)


def _key(values, extra_key=None):
    return RadixKey(array("q", values), extra_key)


def _make_node(
    tokens,
    *,
    value=None,
    host_value=None,
    mamba_value=None,
    mamba_host_value=None,
    hash_value=None,
):
    node = TreeNode()
    node.key = _key(tokens)
    node.value = value
    node.host_value = host_value
    node.mamba_value = mamba_value
    node.mamba_host_value = mamba_host_value
    node.hash_value = hash_value
    return node


def _link(parent, child, page_size):
    child.parent = parent
    parent.children[child.key.child_key(page_size)] = child


class _RecordingPool:
    def __init__(self, alloc_results=(), available_size=0):
        self.alloc_results = list(alloc_results)
        self.available = available_size
        self.alloc_sizes = []
        self.freed = []

    def alloc(self, size):
        self.alloc_sizes.append(size)
        if not self.alloc_results:
            return None
        return self.alloc_results.pop(0)

    def free(self, indices):
        self.freed.append(indices.clone())

    def available_size(self):
        return self.available


class _RecordingEvictionController:
    def __init__(self, write_policy="write_through"):
        self.write_policy = write_policy
        self.device_evictions = []
        self.host_evictions = []

    def evict_device(self, indices):
        self.device_evictions.append(indices.clone())

    def evict_host(self, indices):
        self.host_evictions.append(indices.clone())
        return len(indices)


class _FinishedEvent:
    def __init__(self):
        self.synchronize_count = 0

    def query(self):
        return True

    def synchronize(self):
        self.synchronize_count += 1


def _make_cache(page_size=1):
    cache = object.__new__(HiMambaRadixCache)
    cache.disable = False
    cache.page_size = page_size
    cache.device = torch.device("cpu")
    cache.tp_group = None
    cache.tp_world_size = 1

    root = _make_node([], value=_indices([]), hash_value=[])
    root.full_lock_ref = 1
    root.mamba_lock_ref = 1
    cache.root_node = root

    cache.full_lru_list = LRUList(mamba=False)
    cache.mamba_lru_list = LRUList(mamba=True)
    cache.mamba_host_lru_list = HostLRUList()
    cache.evictable_full_device_leaves = set()
    cache.evictable_full_host_leaves = set()

    cache.full_evictable_size_ = 0
    cache.full_protected_size_ = 0
    cache.mamba_evictable_size_ = 0
    cache.mamba_protected_size_ = 0

    cache.ongoing_write_through = {}
    cache.ongoing_load_back = {}
    cache.ongoing_prefetch = {}
    cache.ongoing_backup = {}
    cache.prefetch_loaded_tokens_by_reqid = {}

    cache.write_through_threshold = 2
    cache.load_back_threshold = 10
    cache.prefetch_threshold = 4
    cache.prefetch_timeout_config = PrefetchTimeoutConfig()
    cache.prefetch_stop_policy = "best_effort"

    cache.enable_storage = False
    cache.enable_storage_metrics = False
    cache.enable_kv_cache_events = False
    cache.metrics_collector = None
    cache.kv_event_queue = []

    cache.req_to_token_pool = SimpleNamespace(mamba_allocator=_RecordingPool())
    cache.mamba_pool_host = _RecordingPool()
    cache._record_store_event = MagicMock()
    cache._record_remove_event = MagicMock()
    return cache


class TestHostMambaLRU(CustomTestCase):
    def setUp(self):
        super().setUp()
        _reset_tree_node_counters()

    def test_host_lru_order_is_independent_from_device_mamba_lru(self):
        device_lru = LRUList(mamba=True)
        host_lru = HostLRUList()
        first = _make_node(
            [1], mamba_value=_indices([11]), mamba_host_value=_indices([21])
        )
        second = _make_node([2], mamba_host_value=_indices([22]))

        device_lru.insert_mru(first)
        device_links = (first.mamba_prev, first.mamba_next)
        host_lru.insert_mru(first)
        host_lru.insert_mru(second)

        self.assertIs(host_lru.get_lru_no_lock(), first)
        host_lru.reset_node_mru(first)
        self.assertIs(host_lru.get_lru_no_lock(), second)
        self.assertEqual((first.mamba_prev, first.mamba_next), device_links)

        host_lru.remove_node(first)
        self.assertFalse(host_lru.in_list(first))
        self.assertTrue(device_lru.in_list(first))
        self.assertIsNone(first.host_mamba_prev)
        self.assertIsNone(first.host_mamba_next)

    def test_host_lru_rejects_tombstones_and_duplicate_inserts(self):
        host_lru = HostLRUList()
        tombstone = _make_node([1])
        resident = _make_node([2], mamba_host_value=_indices([20]))

        with self.assertRaisesRegex(AssertionError, "tombstone"):
            host_lru.insert_mru(tombstone)

        host_lru.insert_mru(resident)
        with self.assertRaisesRegex(AssertionError, "already"):
            host_lru.insert_mru(resident)


class TestReferenceAndHostProtection(CustomTestCase):
    def setUp(self):
        super().setUp()
        _reset_tree_node_counters()
        self.cache = _make_cache()

    def test_nested_host_protection_reinserts_only_after_final_release(self):
        node = _make_node(
            [1],
            host_value=_indices([10]),
            mamba_host_value=_indices([20]),
        )
        _link(self.cache.root_node, node, self.cache.page_size)
        self.cache.mamba_host_lru_list.insert_mru(node)
        self.cache._update_full_host_leaf_status(node)
        self.assertIn(node, self.cache.evictable_full_host_leaves)

        self.cache._protect_host_node(node)
        self.cache._protect_host_node(node)

        self.assertEqual(node.host_ref_counter, 2)
        self.assertEqual(node.host_mamba_ref_counter, 2)
        self.assertNotIn(node, self.cache.evictable_full_host_leaves)
        self.assertFalse(self.cache.mamba_host_lru_list.in_list(node))

        self.cache._release_host_node(node)
        self.assertEqual(node.host_ref_counter, 1)
        self.assertEqual(node.host_mamba_ref_counter, 1)
        self.assertFalse(self.cache.mamba_host_lru_list.in_list(node))

        self.cache._release_host_node(node)
        self.assertEqual(node.host_ref_counter, 0)
        self.assertEqual(node.host_mamba_ref_counter, 0)
        self.assertTrue(self.cache.mamba_host_lru_list.in_list(node))
        self.assertIn(node, self.cache.evictable_full_host_leaves)

    def test_lock_transitions_move_capacity_only_at_zero_one_boundaries(self):
        parent = _make_node([1, 2], value=_indices([10, 11]))
        leaf = _make_node(
            [3, 4, 5],
            value=_indices([12, 13, 14]),
            mamba_value=_indices([30]),
        )
        _link(self.cache.root_node, parent, self.cache.page_size)
        _link(parent, leaf, self.cache.page_size)
        self.cache.full_lru_list.insert_mru(parent)
        self.cache.full_lru_list.insert_mru(leaf)
        self.cache.mamba_lru_list.insert_mru(leaf)
        self.cache.full_evictable_size_ = 5
        self.cache.mamba_evictable_size_ = 1
        self.cache._update_full_device_leaf_status(leaf)

        first = self.cache.inc_lock_ref(leaf)
        second = self.cache.inc_lock_ref(leaf)

        self.assertEqual(first.delta, -5)
        self.assertEqual(second.delta, 0)
        self.assertEqual((parent.full_lock_ref, leaf.full_lock_ref), (2, 2))
        self.assertEqual(leaf.mamba_lock_ref, 2)
        self.assertEqual(self.cache.full_evictable_size_, 0)
        self.assertEqual(self.cache.full_protected_size_, 5)
        self.assertEqual(self.cache.mamba_evictable_size_, 0)
        self.assertEqual(self.cache.mamba_protected_size_, 1)
        self.assertNotIn(leaf, self.cache.evictable_full_device_leaves)

        first_release = self.cache.dec_lock_ref(leaf)
        final_release = self.cache.dec_lock_ref(leaf)

        self.assertEqual(first_release.delta, 0)
        self.assertEqual(final_release.delta, 5)
        self.assertEqual((parent.full_lock_ref, leaf.full_lock_ref), (0, 0))
        self.assertEqual(leaf.mamba_lock_ref, 0)
        self.assertEqual(self.cache.full_evictable_size_, 5)
        self.assertEqual(self.cache.full_protected_size_, 0)
        self.assertEqual(self.cache.mamba_evictable_size_, 1)
        self.assertEqual(self.cache.mamba_protected_size_, 0)
        self.assertIn(leaf, self.cache.evictable_full_device_leaves)


class TestBackupAndLoadBack(CustomTestCase):
    def setUp(self):
        super().setUp()
        _reset_tree_node_counters()
        self.cache = _make_cache()

    def test_write_backup_requires_a_contiguous_host_prefix(self):
        parent = _make_node([1], value=_indices([10]))
        child = _make_node([2], value=_indices([11]), mamba_value=_indices([30]))
        _link(self.cache.root_node, parent, self.cache.page_size)
        _link(parent, child, self.cache.page_size)
        self.cache.full_lru_list.insert_mru(parent)
        self.cache.full_lru_list.insert_mru(child)
        self.cache.mamba_lru_list.insert_mru(child)
        self.cache.full_evictable_size_ = 2
        self.cache.mamba_evictable_size_ = 1
        self.cache._update_full_device_leaf_status(child)
        self.cache.cache_controller = MagicMock()

        self.assertEqual(self.cache.write_backup(child), 0)
        self.cache.cache_controller.write.assert_not_called()
        self.assertIsNone(child.host_value)

        parent.host_value = _indices([20])

        def commit_write(*, device_indices, node_id, extra_pools):
            extra_pools[0].host_indices = _indices([40])
            return _indices([21])

        self.cache.cache_controller.write.side_effect = commit_write

        self.assertEqual(self.cache.write_backup(child), 1)

        write_call = self.cache.cache_controller.write.call_args
        self.assertTrue(
            torch.equal(write_call.kwargs["device_indices"], _indices([11]))
        )
        self.assertEqual(write_call.kwargs["node_id"], child.id)
        self.assertTrue(torch.equal(child.host_value, _indices([21])))
        self.assertTrue(torch.equal(child.mamba_host_value, _indices([40])))
        self.assertTrue(self.cache.mamba_host_lru_list.in_list(child))
        self.assertIs(self.cache.ongoing_write_through[child.id], child)

    def test_write_backup_evicts_host_then_commits_kv_and_mamba(self):
        node = _make_node([1, 2], value=_indices([10, 11]), mamba_value=_indices([30]))
        _link(self.cache.root_node, node, self.cache.page_size)
        self.cache.full_lru_list.insert_mru(node)
        self.cache.mamba_lru_list.insert_mru(node)
        self.cache.full_evictable_size_ = 2
        self.cache.mamba_evictable_size_ = 1
        self.cache._update_full_device_leaf_status(node)

        class RetryWriteController:
            def __init__(self):
                self.calls = []
                self.host_result = _indices([20, 21])
                self.mamba_host_result = _indices([40])

            def write(self, *, device_indices, node_id, extra_pools):
                self.calls.append((device_indices.clone(), node_id, extra_pools))
                if len(self.calls) == 1:
                    return None
                extra_pools[0].host_indices = self.mamba_host_result
                return self.host_result

        controller = RetryWriteController()
        self.cache.cache_controller = controller
        self.cache.evict_host = MagicMock()

        backed_up = self.cache.write_backup(node)

        self.assertEqual(backed_up, 2)
        self.assertEqual(len(controller.calls), 2)
        self.cache.evict_host.assert_called_once_with(2)
        self.assertTrue(torch.equal(node.host_value, _indices([20, 21])))
        self.assertTrue(torch.equal(node.mamba_host_value, _indices([40])))
        self.assertNotEqual(
            node.host_value.data_ptr(), controller.host_result.data_ptr()
        )
        self.assertNotEqual(
            node.mamba_host_value.data_ptr(), controller.mamba_host_result.data_ptr()
        )
        self.assertTrue(self.cache.mamba_host_lru_list.in_list(node))
        self.assertIs(self.cache.ongoing_write_through[node.id], node)
        self.assertEqual((node.full_lock_ref, node.mamba_lock_ref), (1, 1))
        self.assertEqual(self.cache.full_protected_size_, 2)
        self.assertEqual(self.cache.mamba_protected_size_, 1)

    def test_load_back_skips_short_kv_only_hits(self):
        node = _make_node([1, 2], host_value=_indices([20, 21]))
        _link(self.cache.root_node, node, self.cache.page_size)
        self.cache.cache_controller = MagicMock()

        result = self.cache.load_back(node)

        self.assertIsNone(result)
        self.cache.cache_controller.load.assert_not_called()
        self.assertTrue(node.evicted)
        self.assertEqual(self.cache.ongoing_load_back, {})

    def test_load_back_restores_contiguous_path_and_releases_on_ack(self):
        first = _make_node([1, 2], host_value=_indices([20, 21]))
        last = _make_node(
            [3], host_value=_indices([22]), mamba_host_value=_indices([40])
        )
        _link(self.cache.root_node, first, self.cache.page_size)
        _link(first, last, self.cache.page_size)
        self.cache.mamba_host_lru_list.insert_mru(last)
        self.cache._update_full_host_leaf_status(first)
        self.cache._update_full_host_leaf_status(last)
        self.assertIn(last, self.cache.evictable_full_host_leaves)

        finish_event = _FinishedEvent()

        class LoadController:
            def __init__(self):
                self.calls = []
                self.ack_load_queue = []

            def load(self, *, host_indices, node_id, extra_pools):
                self.calls.append((host_indices.clone(), node_id, extra_pools))
                extra_pools[0].device_indices = _indices([50])
                self.ack_load_queue.append(
                    SimpleNamespace(
                        finish_event=finish_event,
                        node_ids=[node_id],
                        num_tokens=len(host_indices),
                        timing_enabled=False,
                    )
                )
                return _indices([100, 101, 102])

        controller = LoadController()
        self.cache.cache_controller = controller
        self.cache.metrics_collector = MagicMock()

        loaded = self.cache.load_back(last)

        self.assertTrue(torch.equal(loaded, _indices([100, 101, 102])))
        self.assertTrue(torch.equal(controller.calls[0][0], _indices([20, 21, 22])))
        self.assertTrue(torch.equal(first.value, _indices([100, 101])))
        self.assertTrue(torch.equal(last.value, _indices([102])))
        self.assertTrue(torch.equal(last.mamba_value, _indices([50])))
        self.assertTrue(self.cache.full_lru_list.in_list(first))
        self.assertTrue(self.cache.full_lru_list.in_list(last))
        self.assertTrue(self.cache.mamba_lru_list.in_list(last))
        self.assertTrue(self.cache.mamba_host_lru_list.in_list(last))
        self.assertNotIn(last, self.cache.evictable_full_host_leaves)
        self.assertTrue(torch.equal(first.host_value, _indices([20, 21])))
        self.assertTrue(torch.equal(last.host_value, _indices([22])))
        self.assertTrue(torch.equal(last.mamba_host_value, _indices([40])))
        self.assertIs(self.cache.ongoing_load_back[last.id], last)
        self.assertEqual((first.full_lock_ref, last.full_lock_ref), (1, 1))
        self.assertEqual(last.mamba_lock_ref, 1)

        self.cache.loading_check()

        self.assertEqual(controller.ack_load_queue, [])
        self.assertEqual(self.cache.ongoing_load_back, {})
        self.assertEqual((first.full_lock_ref, last.full_lock_ref), (0, 0))
        self.assertEqual(last.mamba_lock_ref, 0)
        self.assertTrue(self.cache.mamba_host_lru_list.in_list(last))
        self.assertNotIn(last, self.cache.evictable_full_host_leaves)
        self.assertEqual(finish_event.synchronize_count, 1)
        self.cache.metrics_collector.increment_load_back_num_tokens.assert_called_once_with(
            3
        )
        self.cache.metrics_collector.observe_load_back_duration.assert_not_called()


class TestAsyncAckProcessing(CustomTestCase):
    def setUp(self):
        super().setUp()
        _reset_tree_node_counters()
        self.cache = _make_cache()

    def test_writing_check_processes_only_the_tp_common_completed_prefix(self):
        first = _make_node([1], value=_indices([10]), host_value=_indices([20]))
        second = _make_node([2], value=_indices([11]), host_value=_indices([21]))
        _link(self.cache.root_node, first, self.cache.page_size)
        _link(self.cache.root_node, second, self.cache.page_size)
        self.cache.full_lru_list.insert_mru(first)
        self.cache.full_lru_list.insert_mru(second)
        self.cache.full_evictable_size_ = 2
        self.cache._update_full_device_leaf_status(first)
        self.cache._update_full_device_leaf_status(second)
        self.cache.inc_lock_ref(first)
        self.cache.inc_lock_ref(second)
        first_event = _FinishedEvent()
        second_event = _FinishedEvent()
        first_ack = SimpleNamespace(finish_event=first_event, node_ids=[first.id])
        second_ack = SimpleNamespace(finish_event=second_event, node_ids=[second.id])
        self.cache.cache_controller = SimpleNamespace(
            ack_write_queue=[first_ack, second_ack]
        )
        self.cache.ongoing_write_through = {
            first.id: first,
            second.id: second,
        }
        self.cache.tp_world_size = 2
        self.cache.tp_group = object()

        def report_one_common_ack(queue_size, **_):
            queue_size.fill_(1)

        with patch(
            "torch.distributed.all_reduce", side_effect=report_one_common_ack
        ) as all_reduce:
            self.cache.writing_check()

        all_reduce.assert_called_once()
        self.assertEqual(
            all_reduce.call_args.kwargs["op"], torch.distributed.ReduceOp.MIN
        )
        self.assertIs(all_reduce.call_args.kwargs["group"], self.cache.tp_group)
        self.assertEqual(self.cache.cache_controller.ack_write_queue, [second_ack])
        self.assertNotIn(first.id, self.cache.ongoing_write_through)
        self.assertIs(self.cache.ongoing_write_through[second.id], second)
        self.assertEqual(first_event.synchronize_count, 1)
        self.assertEqual(second_event.synchronize_count, 0)
        self.assertEqual((first.full_lock_ref, second.full_lock_ref), (0, 1))
        self.assertEqual(self.cache.full_evictable_size_, 1)
        self.assertEqual(self.cache.full_protected_size_, 1)
        self.assertIn(first, self.cache.evictable_full_device_leaves)
        self.assertNotIn(second, self.cache.evictable_full_device_leaves)
        self.cache._record_store_event.assert_called_once_with(
            first, medium=StorageMedium.CPU
        )


class TestEvictionAndTreeTransitions(CustomTestCase):
    def setUp(self):
        super().setUp()
        _reset_tree_node_counters()
        self.cache = _make_cache()

    def test_evict_to_host_preserves_tree_and_host_backups(self):
        node = _make_node(
            [1, 2],
            value=_indices([10, 11]),
            host_value=_indices([20, 21]),
            mamba_value=_indices([30]),
            mamba_host_value=_indices([40]),
        )
        _link(self.cache.root_node, node, self.cache.page_size)
        self.cache.full_lru_list.insert_mru(node)
        self.cache.mamba_lru_list.insert_mru(node)
        self.cache.mamba_host_lru_list.insert_mru(node)
        self.cache.full_evictable_size_ = 2
        self.cache.mamba_evictable_size_ = 1
        self.cache._update_full_device_leaf_status(node)
        self.assertIn(node, self.cache.evictable_full_device_leaves)
        self.cache.cache_controller = _RecordingEvictionController()
        mamba_allocator = _RecordingPool()
        self.cache.req_to_token_pool = SimpleNamespace(mamba_allocator=mamba_allocator)

        full_count, mamba_count = self.cache._evict_to_host(node)

        self.assertEqual((full_count, mamba_count), (2, 1))
        self.assertTrue(node.evicted)
        self.assertTrue(node.mamba_evicted)
        self.assertTrue(torch.equal(node.host_value, _indices([20, 21])))
        self.assertTrue(torch.equal(node.mamba_host_value, _indices([40])))
        self.assertIs(
            self.cache.root_node.children[node.key.child_key(self.cache.page_size)],
            node,
        )
        self.assertFalse(self.cache.full_lru_list.in_list(node))
        self.assertFalse(self.cache.mamba_lru_list.in_list(node))
        self.assertTrue(self.cache.mamba_host_lru_list.in_list(node))
        self.assertNotIn(node, self.cache.evictable_full_device_leaves)
        self.assertIn(node, self.cache.evictable_full_host_leaves)
        self.assertEqual(self.cache.full_evictable_size_, 0)
        self.assertEqual(self.cache.mamba_evictable_size_, 0)
        self.assertEqual(self.cache.full_protected_size_, 0)
        self.assertEqual(self.cache.mamba_protected_size_, 0)
        self.assertTrue(
            torch.equal(
                self.cache.cache_controller.device_evictions[0], _indices([10, 11])
            )
        )
        self.assertTrue(torch.equal(mamba_allocator.freed[0], _indices([30])))
        self.cache._record_remove_event.assert_called_once_with(
            node, medium=StorageMedium.GPU
        )

    def test_evict_device_leaf_dispatches_by_backup_and_write_policy(self):
        node = _make_node([1], value=_indices([10]))
        self.cache.cache_controller = SimpleNamespace(write_policy="write_through")
        self.cache._evict_to_host = MagicMock(return_value=(1, 1))
        self.cache._evict_regular = MagicMock(return_value=(2, 2))
        self.cache.write_backup = MagicMock()
        self.cache.writing_check = MagicMock()

        node.host_value = _indices([20])
        self.assertEqual(self.cache._evict_device_leaf(node), (1, 1))
        self.cache._evict_to_host.assert_called_once_with(node)

        self.cache._evict_to_host.reset_mock()
        node.host_value = None
        self.assertEqual(self.cache._evict_device_leaf(node), (2, 2))
        self.cache._evict_regular.assert_called_once_with(node)

        self.cache._evict_regular.reset_mock()
        self.cache._evict_to_host.reset_mock()
        self.cache.cache_controller.write_policy = "write_back"
        ordered_calls = []
        self.cache.write_backup.side_effect = (
            lambda *args, **kwargs: ordered_calls.append("backup")
        )
        self.cache.writing_check.side_effect = (
            lambda *args, **kwargs: ordered_calls.append("wait")
        )
        self.cache._evict_to_host.side_effect = lambda *_: (
            ordered_calls.append("demote") or (1, 1)
        )
        self.assertEqual(self.cache._evict_device_leaf(node), (1, 1))
        self.assertEqual(ordered_calls, ["backup", "wait", "demote"])
        self.cache.write_backup.assert_called_once_with(node, write_back=True)
        self.cache.writing_check.assert_called_once_with(write_back=True)
        self.cache._evict_to_host.assert_called_once_with(node)

    def test_evict_regular_deletes_unbacked_leaf_and_frees_device_state(self):
        node = _make_node([1, 2], value=_indices([10, 11]), mamba_value=_indices([30]))
        _link(self.cache.root_node, node, self.cache.page_size)
        self.cache.full_lru_list.insert_mru(node)
        self.cache.mamba_lru_list.insert_mru(node)
        self.cache.full_evictable_size_ = 2
        self.cache.mamba_evictable_size_ = 1
        self.cache._update_full_device_leaf_status(node)
        self.cache.cache_controller = _RecordingEvictionController()
        mamba_allocator = _RecordingPool()
        self.cache.req_to_token_pool = SimpleNamespace(mamba_allocator=mamba_allocator)

        full_count, mamba_count = self.cache._evict_regular(node)

        self.assertEqual((full_count, mamba_count), (2, 1))
        self.assertIsNone(node.value)
        self.assertIsNone(node.mamba_value)
        self.assertNotIn(
            node.key.child_key(self.cache.page_size), self.cache.root_node.children
        )
        self.assertFalse(self.cache.full_lru_list.in_list(node))
        self.assertFalse(self.cache.mamba_lru_list.in_list(node))
        self.assertNotIn(node, self.cache.evictable_full_device_leaves)
        self.assertNotIn(node, self.cache.evictable_full_host_leaves)
        self.assertEqual(self.cache.full_evictable_size_, 0)
        self.assertEqual(self.cache.mamba_evictable_size_, 0)
        self.assertTrue(
            torch.equal(
                self.cache.cache_controller.device_evictions[0], _indices([10, 11])
            )
        )
        self.assertTrue(torch.equal(mamba_allocator.freed[0], _indices([30])))
        self.cache._record_remove_event.assert_called_once_with(
            node, medium=StorageMedium.GPU
        )

    def test_internal_mamba_eviction_creates_tombstone_but_keeps_full_kv(self):
        internal = _make_node([1], value=_indices([10]), mamba_value=_indices([30]))
        child = _make_node([2], value=_indices([11]), mamba_value=_indices([31]))
        _link(self.cache.root_node, internal, self.cache.page_size)
        _link(internal, child, self.cache.page_size)
        self.cache.full_lru_list.insert_mru(internal)
        self.cache.full_lru_list.insert_mru(child)
        self.cache.mamba_lru_list.insert_mru(internal)
        self.cache.mamba_lru_list.insert_mru(child)
        self.cache.full_evictable_size_ = 2
        self.cache.mamba_evictable_size_ = 2
        self.cache._update_full_device_leaf_status(internal)
        self.cache._update_full_device_leaf_status(child)
        self.assertEqual(self.cache.evictable_full_device_leaves, {child})
        mamba_allocator = _RecordingPool()
        self.cache.req_to_token_pool = SimpleNamespace(mamba_allocator=mamba_allocator)

        evicted = self.cache.evict_mamba(1)

        self.assertEqual(evicted, 1)
        self.assertIsNone(internal.mamba_value)
        self.assertTrue(torch.equal(internal.value, _indices([10])))
        self.assertIs(
            internal.children[child.key.child_key(self.cache.page_size)], child
        )
        self.assertTrue(self.cache.full_lru_list.in_list(internal))
        self.assertTrue(self.cache.full_lru_list.in_list(child))
        self.assertFalse(self.cache.mamba_lru_list.in_list(internal))
        self.assertTrue(self.cache.mamba_lru_list.in_list(child))
        self.assertTrue(torch.equal(child.mamba_value, _indices([31])))
        self.assertEqual(self.cache.full_evictable_size_, 2)
        self.assertEqual(self.cache.mamba_evictable_size_, 1)
        self.assertEqual(self.cache.evictable_full_device_leaves, {child})
        self.assertTrue(torch.equal(mamba_allocator.freed[0], _indices([30])))

    def test_split_evicted_node_splits_host_data_hashes_and_topology(self):
        self.cache = _make_cache(page_size=2)
        child = _make_node(
            [1, 2, 3, 4],
            host_value=_indices([10, 11, 12, 13]),
            mamba_host_value=_indices([40]),
            hash_value=["h12", "h34"],
        )
        _link(self.cache.root_node, child, self.cache.page_size)
        self.cache.mamba_host_lru_list.insert_mru(child)
        self.cache._update_full_host_leaf_status(child)

        new_parent = self.cache._split_evicted_node(child.key, child, split_len=2)

        self.assertEqual(list(new_parent.key), [1, 2])
        self.assertEqual(list(child.key), [3, 4])
        self.assertTrue(torch.equal(new_parent.host_value, _indices([10, 11])))
        self.assertTrue(torch.equal(child.host_value, _indices([12, 13])))
        self.assertEqual(new_parent.hash_value, ["h12"])
        self.assertEqual(child.hash_value, ["h34"])
        self.assertIs(new_parent.parent, self.cache.root_node)
        self.assertIs(child.parent, new_parent)
        self.assertIs(
            self.cache.root_node.children[
                new_parent.key.child_key(self.cache.page_size)
            ],
            new_parent,
        )
        self.assertIs(
            new_parent.children[child.key.child_key(self.cache.page_size)], child
        )
        self.assertIsNone(new_parent.mamba_host_value)
        self.assertTrue(torch.equal(child.mamba_host_value, _indices([40])))
        self.assertTrue(self.cache.mamba_host_lru_list.in_list(child))
        self.assertFalse(self.cache.mamba_host_lru_list.in_list(new_parent))
        self.assertNotIn(new_parent, self.cache.evictable_full_host_leaves)
        self.assertIn(child, self.cache.evictable_full_host_leaves)


class TestTieredMatchingAndHostInsertion(CustomTestCase):
    def setUp(self):
        super().setUp()
        _reset_tree_node_counters()

    def test_match_prefix_reports_device_kv_and_host_mamba_boundary(self):
        cache = _make_cache(page_size=2)
        device_node = _make_node(
            [1, 2], value=_indices([10, 11]), mamba_value=_indices([30])
        )
        host_intermediate = _make_node(
            [3, 4],
            host_value=_indices([20, 21]),
        )
        host_checkpoint = _make_node(
            [5, 6],
            host_value=_indices([22, 23]),
            mamba_host_value=_indices([40]),
        )
        _link(cache.root_node, device_node, cache.page_size)
        _link(device_node, host_intermediate, cache.page_size)
        _link(host_intermediate, host_checkpoint, cache.page_size)
        cache.full_lru_list.insert_mru(device_node)
        cache.mamba_lru_list.insert_mru(device_node)
        cache.mamba_host_lru_list.insert_mru(host_checkpoint)
        cache.full_evictable_size_ = 2
        cache.mamba_evictable_size_ = 1
        cache._update_full_device_leaf_status(device_node)
        cache._update_full_host_leaf_status(host_intermediate)
        cache._update_full_host_leaf_status(host_checkpoint)
        self.assertIn(device_node, cache.evictable_full_device_leaves)
        self.assertIn(host_checkpoint, cache.evictable_full_host_leaves)

        result = cache.match_prefix(
            MatchPrefixParams(key=_key([1, 2, 3, 4, 5, 6, 99]), cow_mamba=False)
        )

        self.assertTrue(torch.equal(result.device_indices, _indices([10, 11])))
        self.assertIs(result.last_device_node, device_node)
        self.assertIs(result.last_host_node, host_checkpoint)
        self.assertIs(result.best_match_node, host_checkpoint)
        self.assertEqual(result.host_hit_length, 4)
        self.assertEqual(result.mamba_host_hit_length, 1)
        self.assertTrue(cache.mamba_host_lru_list.in_list(host_checkpoint))
        self.assertIn(host_checkpoint, cache.evictable_full_host_leaves)

    def test_insert_helper_host_reuses_prefix_and_attaches_mamba_to_new_suffix(self):
        cache = _make_cache(page_size=2)
        prefix = _make_node([1, 2], host_value=_indices([10, 11]), hash_value=["h12"])
        _link(cache.root_node, prefix, cache.page_size)
        cache._update_full_host_leaf_status(prefix)

        matched = cache._insert_helper_host(
            cache.root_node,
            _key([1, 2, 3, 4]),
            _indices([20, 21, 22, 23]),
            ["new12", "new34"],
            _indices([40]),
            True,
        )

        self.assertEqual(matched, 2)
        suffix = next(iter(prefix.children.values()))
        self.assertEqual(list(suffix.key), [3, 4])
        self.assertTrue(suffix.evicted)
        self.assertTrue(torch.equal(prefix.host_value, _indices([10, 11])))
        self.assertTrue(torch.equal(suffix.host_value, _indices([22, 23])))
        self.assertEqual(prefix.hash_value, ["h12"])
        self.assertEqual(suffix.hash_value, ["new34"])
        self.assertTrue(torch.equal(suffix.mamba_host_value, _indices([40])))
        self.assertNotIn(prefix, cache.evictable_full_host_leaves)
        self.assertIn(suffix, cache.evictable_full_host_leaves)
        self.assertTrue(cache.mamba_host_lru_list.in_list(suffix))


class TestStorageConfigurationAndPrefetchPolicy(CustomTestCase):
    def setUp(self):
        super().setUp()
        _reset_tree_node_counters()
        self.cache = _make_cache(page_size=4)

    def test_parse_storage_config_extracts_hicache_options(self):
        extra, threshold, timeout, pass_prefix_keys = (
            self.cache._parse_storage_backend_extra_config(None)
        )
        self.assertEqual(extra, {})
        self.assertEqual(threshold, 256)
        self.assertEqual(timeout, PrefetchTimeoutConfig())
        self.assertFalse(pass_prefix_keys)

        config = json.dumps(
            {
                "prefetch_threshold": 128,
                "prefetch_timeout_base": 1.5,
                "prefetch_timeout_per_ki_token": 0.25,
                "prefetch_timeout_max": 9,
                "hicache_storage_pass_prefix_keys": True,
                "backend_option": "kept",
            }
        )
        extra, threshold, timeout, pass_prefix_keys = (
            self.cache._parse_storage_backend_extra_config(config)
        )
        self.assertEqual(extra, {"backend_option": "kept"})
        self.assertEqual(threshold, 128)
        self.assertEqual(
            timeout, PrefetchTimeoutConfig(base=1.5, per_ki_token=0.25, max=9.0)
        )
        self.assertTrue(pass_prefix_keys)

    def test_parse_storage_config_file_and_invalid_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump({"prefetch_threshold": 64, "backend": "file"}, config_file)
            config_file.flush()
            extra, threshold, _, _ = self.cache._parse_storage_backend_extra_config(
                f"@{config_file.name}"
            )
        self.assertEqual(extra, {"backend": "file"})
        self.assertEqual(threshold, 64)

        invalid_cases = [
            ({"prefetch_threshold": "64"}, "prefetch_threshold"),
            ({"prefetch_timeout_base": "fast"}, "prefetch_timeout_base"),
            (
                {"hicache_storage_pass_prefix_keys": 1},
                "hicache_storage_pass_prefix_keys",
            ),
        ]
        for config, expected_name in invalid_cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, expected_name):
                    self.cache._parse_storage_backend_extra_config(json.dumps(config))

    def test_prefetch_policy_waits_for_kv_and_mamba_sidecar(self):
        operation = SimpleNamespace(
            hash_value=["p0", "p1"],
            completed_tokens=4,
            pool_transfers=[PoolTransfer(name=PoolName.MAMBA)],
            pool_transfers_done=False,
            is_terminated=lambda: False,
        )
        self.cache.prefetch_stop_policy = "wait_complete"

        self.assertFalse(self.cache.can_terminate_prefetch(operation))
        operation.completed_tokens = 8
        self.assertFalse(self.cache.can_terminate_prefetch(operation))
        operation.pool_transfers_done = True
        self.assertTrue(self.cache.can_terminate_prefetch(operation))

    def test_prefetch_policy_timeout_and_external_termination(self):
        terminated = False
        operation = SimpleNamespace(
            hash_value=["p0"],
            completed_tokens=0,
            pool_transfers=None,
            pool_transfers_done=True,
            is_terminated=lambda: terminated,
        )
        self.cache.prefetch_stop_policy = "timeout"
        self.cache.is_prefetch_timeout = MagicMock(return_value=False)

        self.assertFalse(self.cache.can_terminate_prefetch(operation))
        self.cache.is_prefetch_timeout.return_value = True
        self.assertTrue(self.cache.can_terminate_prefetch(operation))

        self.cache.is_prefetch_timeout.return_value = False
        terminated = True
        self.assertTrue(self.cache.can_terminate_prefetch(operation))

    def test_prefetch_allocation_failure_releases_host_protection(self):
        anchor = _make_node([1, 2, 3, 4], host_value=_indices([20, 21, 22, 23]))
        _link(self.cache.root_node, anchor, self.cache.page_size)
        self.cache._update_full_host_leaf_status(anchor)
        self.assertIn(anchor, self.cache.evictable_full_host_leaves)

        self.cache.enable_storage = True
        self.cache.mamba_pool_host = _RecordingPool()
        self.cache.cache_controller = SimpleNamespace(
            prefetch_rate_limited=MagicMock(return_value=False),
            prefetch=MagicMock(),
            prefetch_tokens_occupied=0,
        )

        self.cache.prefetch_from_storage(
            "request-0", anchor, [5, 6, 7, 8], last_hash="h4"
        )

        self.assertEqual(self.cache.mamba_pool_host.alloc_sizes, [1, 1])
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertEqual(anchor.host_mamba_ref_counter, 0)
        self.assertIn(anchor, self.cache.evictable_full_host_leaves)
        self.assertEqual(self.cache.ongoing_prefetch, {})
        self.assertEqual(self.cache.cache_controller.prefetch_tokens_occupied, 0)
        self.cache.cache_controller.prefetch.assert_not_called()


class TestMambaPoolTransfers(CustomTestCase):
    def setUp(self):
        super().setUp()
        _reset_tree_node_counters()
        self.cache = _make_cache()

    def test_backup_commit_and_archive_use_the_mamba_boundary_hash(self):
        node = _make_node(
            [1, 2],
            mamba_value=_indices([30]),
            hash_value=["first", "last"],
        )
        self.assertIsNone(self.cache.mamba_backup_transfers(_make_node([0])))

        backup = self.cache.mamba_backup_transfers(node)
        self.assertEqual(len(backup), 1)
        self.assertEqual(backup[0].name, PoolName.MAMBA)
        self.assertIsNone(backup[0].host_indices)
        self.assertTrue(torch.equal(backup[0].device_indices, _indices([30])))

        backup[0].host_indices = _indices([40])
        self.cache.mamba_backup_commit(node, backup)
        self.assertTrue(torch.equal(node.mamba_host_value, _indices([40])))
        self.assertTrue(self.cache.mamba_host_lru_list.in_list(node))

        archive = self.cache.mamba_archive_transfers(node)
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive[0].keys, ["last"])
        self.assertEqual(archive[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)
        self.assertTrue(torch.equal(archive[0].host_indices, _indices([40])))

    def test_restore_builds_node_and_request_transfers_then_partitions_commit(self):
        first = _make_node([1], mamba_host_value=_indices([40]))
        middle = _make_node([2], mamba_host_value=_indices([41]))
        last = _make_node([3], mamba_host_value=_indices([42]))
        request_allocator = _RecordingPool(alloc_results=[_indices([70])])
        self.cache.req_to_token_pool = SimpleNamespace(
            mamba_allocator=request_allocator
        )
        request = SimpleNamespace(mamba_pool_idx=None)

        restored_nodes = [first, middle, last]
        transfers = self.cache.mamba_restore_transfers(last, restored_nodes, request)

        self.assertEqual(len(transfers), 2)
        self.assertTrue(torch.equal(transfers[0].host_indices, _indices([40, 41, 42])))
        self.assertIsNone(transfers[0].device_indices)
        self.assertEqual(request.mamba_pool_idx.item(), 70)
        self.assertTrue(torch.equal(transfers[1].host_indices, _indices([42])))
        self.assertTrue(torch.equal(transfers[1].device_indices, _indices([70])))

        transfers[0].device_indices = _indices([80, 81, 82])
        self.cache.mamba_restore_commit(restored_nodes, transfers)
        self.assertTrue(torch.equal(first.mamba_value, _indices([80])))
        self.assertTrue(torch.equal(middle.mamba_value, _indices([81])))
        self.assertTrue(torch.equal(last.mamba_value, _indices([82])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
