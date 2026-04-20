"""
Unit tests for HiRadixCache buffer_only host memory mode
(--hicache-host-memory-mode buffer_only).

In buffer_only mode host memory is a transient staging buffer for L3
storage writes, not a persistent cache tier.  Key differences from
the default "cache" mode:

- host_value is never set on tree nodes (buffer pages are tracked
  separately and freed after the storage write completes).
- Because backuped == (host_value is not None), buffer_only nodes
  always have backuped=False, so GPU eviction goes through
  _evict_regular which fully removes the node from the tree and
  emits BlockRemoved.  There are no "demoted to host" zombie nodes.
- Storage prefetch in buffer_only passes a device-tree node as the
  anchor.  That anchor must be protected from GPU eviction with
  lock_ref for the lifetime of the prefetch.

Usage:
    python -m pytest test/registered/unit/mem_cache/test_hiradix_buffer_only_unit.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

import collections
import sys
import unittest
import unittest.mock
from functools import partial
from queue import Queue

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.evict_policy import LRUStrategy
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import (
    RadixKey,
    TreeNode,
    _key_match_page_size1,
    get_child_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_hiradix_cache(host_memory_mode="buffer_only", page_size=1):
    """Build a minimal HiRadixCache for unit testing without GPU/distributed."""
    cache = object.__new__(HiRadixCache)

    # --- RadixCache base attributes ---
    cache.disable = False
    cache.req_to_token_pool = None
    cache.token_to_kv_pool_allocator = None
    cache.page_size = page_size
    cache.enable_kv_cache_events = False
    cache.is_eagle = False
    cache.disable_finished_insert = False
    cache.eviction_policy = "lru"
    cache.kv_event_queue = []
    cache.device = torch.device("cpu")
    cache.metrics_collector = None
    cache.eviction_strategy = LRUStrategy()
    cache.evictable_leaves = set()

    cache.key_match_fn = _key_match_page_size1
    cache.get_child_key_fn = partial(get_child_key, page_size=page_size)

    # Root node
    root = TreeNode(priority=-sys.maxsize)
    root.key = RadixKey(token_ids=[], extra_key=None)
    root.value = []
    root.host_value = []
    root.lock_ref = 1
    root.hash_value = []
    cache.root_node = root

    cache.evictable_size_ = 0
    cache.protected_size_ = 0
    cache.pinned_size_ = 0

    # --- HiRadixCache-specific attributes ---
    cache.host_memory_mode = host_memory_mode
    cache.evictable_host_leaves = set()
    cache.ongoing_write_through = {}
    cache.ongoing_load_back = {}
    cache.ongoing_prefetch = {}
    cache.ongoing_backup = {}
    cache.pending_write_queue = collections.deque()
    cache.pending_write_node_ids = set()
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.write_through_threshold = 1
    cache.load_back_threshold = 10
    cache.enable_storage = True
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.hicache_storage_pass_prefix_keys = False
    cache.prefetch_threshold = 0
    cache.prefetch_stop_policy = "best_effort"
    cache.tp_world_size = 1
    cache.tp_group = None

    # Mock cache controller
    cc = unittest.mock.Mock()
    cc.write_policy = "write_through"
    cc.evict_device = lambda indices: len(indices) if indices is not None else 0
    cc.mem_pool_device_allocator = unittest.mock.Mock()
    cc.mem_pool_host = unittest.mock.Mock()
    cc.prefetch_tokens_occupied = 0
    cc.prefetch_revoke_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.prefetch_rate_limited = unittest.mock.Mock(return_value=False)
    cc.prefetch_capacity_limit = 10000
    cache.cache_controller = cc

    return cache


def _make_node(cache, parent, token_ids, value_tensor=None):
    """Add a child node to the tree under parent."""
    node = TreeNode()
    node.parent = parent
    node.key = RadixKey(token_ids=token_ids)
    node.value = value_tensor
    node.hash_value = [f"h{t}" for t in token_ids]
    child_key = cache.get_child_key_fn(node.key)
    parent.children[child_key] = node
    if value_tensor is not None:
        cache.evictable_size_ += len(value_tensor)
        cache._update_leaf_status(parent)
        cache._update_leaf_status(node)
    return node


# ---------------------------------------------------------------------------
# Tier 1: TreeNode property tests
# ---------------------------------------------------------------------------


class TestTreeNodeBackuped(unittest.TestCase):
    """TreeNode.backuped is purely host_value-based."""

    def test_backuped_false_by_default(self):
        node = TreeNode()
        self.assertIsNone(node.host_value)
        self.assertFalse(node.backuped)

    def test_backuped_true_when_host_value_set(self):
        node = TreeNode()
        node.host_value = torch.tensor([0, 1])
        self.assertTrue(node.backuped)

    def test_backuped_false_when_host_value_none(self):
        node = TreeNode()
        node.host_value = None
        self.assertFalse(node.backuped)


class TestEvictedNodeNotInEvictableLeaves(unittest.TestCase):
    """_update_leaf_status removes evicted nodes from evictable_leaves."""

    def test_evicted_node_removed(self):
        cache = _create_test_hiradix_cache()
        node = _make_node(cache, cache.root_node, [1, 2], torch.tensor([10, 11]))
        self.assertIn(node, cache.evictable_leaves)

        node.value = None
        cache._update_leaf_status(node)
        self.assertNotIn(node, cache.evictable_leaves)


# ---------------------------------------------------------------------------
# Tier 2: buffer_only eviction = full tree deletion
# ---------------------------------------------------------------------------


class TestBufferOnlyEvictionDeletesFromTree(unittest.TestCase):
    """In buffer_only mode, eviction fully removes nodes (no zombies)."""

    def setUp(self):
        self.cache = _create_test_hiradix_cache(host_memory_mode="buffer_only")

    def test_eviction_deletes_node_from_tree(self):
        node = _make_node(self.cache, self.cache.root_node, [1], torch.tensor([10]))
        child_key = self.cache.get_child_key_fn(node.key)
        self.assertIn(child_key, self.cache.root_node.children)

        self.cache.evict(EvictParams(num_tokens=1))

        self.assertIsNone(node.value)
        self.assertNotIn(child_key, self.cache.root_node.children)
        self.assertNotIn(node, self.cache.evictable_leaves)

    def test_eviction_emits_block_removed(self):
        self.cache.enable_kv_cache_events = True
        node = _make_node(self.cache, self.cache.root_node, [1], torch.tensor([10]))

        with unittest.mock.patch.object(
            self.cache, "_record_remove_event"
        ) as mock_record:
            self.cache.evict(EvictParams(num_tokens=1))
            mock_record.assert_called_once_with(node)

    def test_backuped_always_false_in_buffer_only(self):
        """buffer_only nodes never have host_value set, so backuped=False."""
        node = _make_node(self.cache, self.cache.root_node, [1], torch.tensor([10]))
        self.assertFalse(node.backuped)
        self.assertIsNone(node.host_value)


class TestCacheModeEvictBackuped(unittest.TestCase):
    """In cache mode, _evict_backuped demotes to host (keeps in tree)."""

    def test_evict_backuped_keeps_node_in_tree(self):
        cache = _create_test_hiradix_cache(host_memory_mode="cache")
        node = _make_node(cache, cache.root_node, [1, 2], torch.tensor([10, 11]))
        node.host_value = torch.tensor([20, 21])
        child_key = cache.get_child_key_fn(node.key)

        result = cache._evict_backuped(node)

        self.assertEqual(result, 2)
        self.assertIsNone(node.value)
        # Node stays in tree (demoted to host)
        self.assertIn(child_key, cache.root_node.children)


# ---------------------------------------------------------------------------
# Tier 2: Prefetch anchor lock_ref protection
# ---------------------------------------------------------------------------


class TestPrefetchAnchorLockRef(unittest.TestCase):
    """Prefetch anchor is protected by lock_ref in buffer_only mode."""

    def setUp(self):
        self.cache = _create_test_hiradix_cache(host_memory_mode="buffer_only")
        cc = self.cache.cache_controller
        cc.mem_pool_host.alloc = unittest.mock.Mock(
            return_value=torch.tensor([100, 101, 102, 103])
        )
        self.mock_operation = unittest.mock.Mock()
        self.mock_operation.host_indices = torch.tensor([100, 101, 102, 103])
        cc.prefetch = unittest.mock.Mock(return_value=self.mock_operation)

    def _make_anchor(self):
        node = _make_node(
            self.cache,
            self.cache.root_node,
            [1, 2, 3, 4],
            torch.tensor([10, 11, 12, 13]),
        )
        node.hash_value = ["h1", "h2", "h3", "h4"]
        return node

    def test_prefetch_acquires_lock_ref_buffer_only(self):
        anchor = self._make_anchor()
        self.assertEqual(anchor.lock_ref, 0)

        self.cache.prefetch_from_storage("req1", anchor, [5, 6, 7, 8], last_hash="h4")

        self.assertGreater(anchor.lock_ref, 0)
        self.assertEqual(anchor.host_ref_counter, 1)
        self.assertIn("req1", self.cache.ongoing_prefetch)

    def test_prefetch_releases_lock_ref_on_host_alloc_failure(self):
        anchor = self._make_anchor()
        cc = self.cache.cache_controller
        cc.mem_pool_host.alloc = unittest.mock.Mock(return_value=None)

        self.cache.prefetch_from_storage("req1", anchor, [5, 6, 7, 8], last_hash="h4")

        self.assertEqual(anchor.lock_ref, 0)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertNotIn("req1", self.cache.ongoing_prefetch)

    def test_prefetch_no_lock_ref_in_cache_mode(self):
        cache = _create_test_hiradix_cache(host_memory_mode="cache")
        cc = cache.cache_controller
        cc.mem_pool_host.alloc = unittest.mock.Mock(
            return_value=torch.tensor([100, 101, 102, 103])
        )
        mock_op = unittest.mock.Mock()
        mock_op.host_indices = torch.tensor([100, 101, 102, 103])
        cc.prefetch = unittest.mock.Mock(return_value=mock_op)
        cc.prefetch_rate_limited = unittest.mock.Mock(return_value=False)

        anchor = _make_node(
            cache, cache.root_node, [1, 2, 3, 4], torch.tensor([10, 11, 12, 13])
        )
        anchor.hash_value = ["h1", "h2", "h3", "h4"]

        cache.prefetch_from_storage("req1", anchor, [5, 6, 7, 8], last_hash="h4")
        self.assertEqual(anchor.lock_ref, 0)
        self.assertEqual(anchor.host_ref_counter, 1)

    def test_drain_revoke_releases_lock_ref(self):
        anchor = self._make_anchor()
        self.cache.prefetch_from_storage("req1", anchor, [5, 6, 7, 8], last_hash="h4")
        self.assertGreater(anchor.lock_ref, 0)

        cc = self.cache.cache_controller
        cc.prefetch_revoke_queue.put("req1")

        self.cache._drain_storage_control_queues_impl(
            n_revoke=1, n_backup=0, n_release=0, log_metrics=False
        )

        self.assertEqual(anchor.lock_ref, 0)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertNotIn("req1", self.cache.ongoing_prefetch)

    def test_abort_releases_lock_ref(self):
        anchor = self._make_anchor()
        self.cache.prefetch_from_storage("req1", anchor, [5, 6, 7, 8], last_hash="h4")
        self.assertGreater(anchor.lock_ref, 0)

        cc = self.cache.cache_controller
        cc.terminate_prefetch = unittest.mock.Mock(return_value=(0, []))
        cc.append_host_mem_release = unittest.mock.Mock()

        self.cache.release_aborted_request("req1")

        self.assertEqual(anchor.lock_ref, 0)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertNotIn("req1", self.cache.ongoing_prefetch)

    def test_force_release_releases_lock_ref(self):
        anchor = self._make_anchor()
        self.cache.prefetch_from_storage("req1", anchor, [5, 6, 7, 8], last_hash="h4")
        self.assertGreater(anchor.lock_ref, 0)

        self.cache._force_release_pending_storage_ops()

        self.assertEqual(anchor.lock_ref, 0)
        self.assertEqual(anchor.host_ref_counter, 0)
        self.assertNotIn("req1", self.cache.ongoing_prefetch)


# ---------------------------------------------------------------------------
# Tier 2: Eviction respects lock_ref
# ---------------------------------------------------------------------------


class TestEvictionSkipsLockedAnchor(unittest.TestCase):
    """Eviction skips nodes with lock_ref > 0."""

    def test_evict_skips_node_with_lock_ref(self):
        cache = _create_test_hiradix_cache(host_memory_mode="buffer_only")
        node_a = _make_node(cache, cache.root_node, [1], torch.tensor([10]))
        node_b = _make_node(cache, cache.root_node, [2], torch.tensor([20]))
        cache.inc_lock_ref(node_b)
        self.assertGreater(node_b.lock_ref, 0)

        self.assertIn(node_a, cache.evictable_leaves)
        self.assertNotIn(node_b, cache.evictable_leaves)

        cache.evict(EvictParams(num_tokens=100))

        self.assertIsNone(node_a.value)
        self.assertIsNotNone(node_b.value)


if __name__ == "__main__":
    unittest.main()
