"""Unit tests for BoundaryHiRadixCache L1/L2 invariants."""

from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.memory_pool_host import HostKVCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.registry import create_tree_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _key(tokens):
    return RadixKey(array("q", tokens))


def _node(tokens, *, value=False, host=False, host_ref=0):
    node = TreeNode()
    node.key = _key(tokens)
    node.value = torch.tensor(tokens, dtype=torch.int64) if value else None
    node.host_value = torch.tensor(tokens, dtype=torch.int64) if host else None
    node.host_ref_counter = host_ref
    return node


def _attach(parent, child):
    child.parent = parent
    parent.children[child.key.child_key(1)] = child
    return child


class _Controller:
    write_policy = "write_through"

    def __init__(self):
        self.evicted_device = []
        self.evicted_host = []

    def evict_device(self, value):
        self.evicted_device.append(value.clone())
        return len(value)

    def evict_host(self, value):
        self.evicted_host.append(value.clone())
        return len(value)

    def write(self, device_indices, priority=None, node_id=-1):
        return device_indices.clone()


class _TinyHostKVCache(HostKVCache):
    def get_size_per_token(self):
        return 1

    def init_kv_buffer(self):
        return torch.empty((self.size,), dtype=self.dtype)

    def load_to_device_per_layer(self, *args, **kwargs):
        raise NotImplementedError()

    def backup_from_device_all_layer(self, *args, **kwargs):
        raise NotImplementedError()

    def get_data_page(self, index, flat=True):
        raise NotImplementedError()

    def get_dummy_flat_data_page(self):
        raise NotImplementedError()

    def set_from_flat_data_page(self, index, data_page):
        raise NotImplementedError()


class TestBoundaryHiRadixCachePolicy(CustomTestCase):
    def _cache(self):
        from sglang.srt.mem_cache.boundary_hiradix_cache import BoundaryHiRadixCache

        cache = BoundaryHiRadixCache.__new__(BoundaryHiRadixCache)
        cache.page_size = 1
        cache.root_node = _node([], value=True, host=True)
        cache.root_node.parent = None
        cache.root_node.lock_ref = 1
        cache.disable = False
        cache.evictable_size_ = 0
        cache.protected_size_ = 0
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.eviction_strategy = MagicMock()
        cache.eviction_strategy.get_priority.side_effect = lambda node: node.id
        cache.cache_controller = _Controller()
        cache.enable_storage = False
        cache.enable_kv_cache_events = False
        cache.kv_event_queue = []
        cache._record_remove_event = MagicMock()
        cache._record_store_event = MagicMock()
        cache.update_eviction_metrics = MagicMock()
        cache.writing_check = MagicMock()
        cache.ongoing_write_through = {}
        return cache

    def test_sanity_rejects_d_parent_to_h_child(self):
        cache = self._cache()
        parent = _attach(cache.root_node, _node([1], value=True, host=False))
        _attach(parent, _node([2], value=False, host=True))

        with self.assertRaisesRegex(AssertionError, "D-only parent"):
            cache.sanity_check_boundary_invariant()

    def test_l1_evicts_d_leaf_under_d_parent_by_boundarying_parent_first(self):
        cache = self._cache()
        parent = _attach(cache.root_node, _node([1], value=True, host=False))
        child = _attach(parent, _node([2], value=True, host=False))
        cache.evictable_size_ = len(parent.value) + len(child.value)
        cache.evictable_leaves.add(child)

        evicted = cache._evict_boundary_l1_node(child)

        self.assertEqual(evicted, len(child.host_value))
        self.assertIsNotNone(parent.value)
        self.assertIsNotNone(parent.host_value)
        self.assertIsNone(child.value)
        self.assertIsNotNone(child.host_value)
        cache.sanity_check_boundary_invariant()

    def test_l1_evicts_dh_leaf_to_h_without_deleting_children(self):
        cache = self._cache()
        leaf = _attach(cache.root_node, _node([1], value=True, host=True))
        child = _attach(leaf, _node([2], value=False, host=True))
        cache.evictable_size_ = len(leaf.value)

        evicted = cache._evict_boundary_l1_node(leaf)

        self.assertEqual(evicted, 1)
        self.assertIsNone(leaf.value)
        self.assertIsNotNone(leaf.host_value)
        self.assertIs(child.parent, leaf)
        self.assertIn(child.key.child_key(1), leaf.children)

    def test_l2_evicts_duplicate_dh_host_leaf_before_h_only_leaf(self):
        cache = self._cache()
        duplicate = _attach(cache.root_node, _node([1], value=True, host=True))
        boundary = _attach(cache.root_node, _node([2], value=True, host=True))
        h_leaf = _attach(boundary, _node([3], value=False, host=True))
        cache.evictable_host_leaves.update([duplicate, h_leaf])

        evicted = cache.evict_host(1)

        self.assertEqual(evicted, 1)
        self.assertIsNotNone(duplicate.value)
        self.assertIsNone(duplicate.host_value)
        self.assertIsNotNone(boundary.host_value)
        self.assertIn(h_leaf.key.child_key(1), boundary.children)

    def test_l2_evicts_duplicate_dh_root_first(self):
        cache = self._cache()
        parent = _attach(cache.root_node, _node([1], value=True, host=True))
        child = _attach(parent, _node([2], value=True, host=True))
        cache.evictable_host_leaves.update([parent, child])

        evicted = cache.evict_host(1)

        self.assertEqual(evicted, 1)
        self.assertIsNone(parent.host_value)
        self.assertIsNotNone(child.host_value)
        cache.sanity_check_boundary_invariant()

    def test_l2_preserves_boundary_host_value_when_h_descendant_exists(self):
        cache = self._cache()
        boundary = _attach(cache.root_node, _node([1], value=True, host=True))
        h_leaf = _attach(boundary, _node([2], value=False, host=True))
        cache.evictable_host_leaves.update([boundary, h_leaf])

        evicted = cache.evict_host(1)

        self.assertEqual(evicted, 1)
        self.assertIsNotNone(boundary.host_value)
        self.assertNotIn(h_leaf.key.child_key(1), boundary.children)

    def test_l2_deletes_h_leaves_after_duplicates_are_gone(self):
        cache = self._cache()
        boundary = _attach(cache.root_node, _node([1], value=True, host=True))
        h_leaf = _attach(boundary, _node([2], value=False, host=True))
        cache.evictable_host_leaves.add(h_leaf)

        evicted = cache.evict_host(1)

        self.assertEqual(evicted, 1)
        self.assertNotIn(h_leaf.key.child_key(1), boundary.children)
        cache.sanity_check_boundary_invariant()

    def test_boundary_becomes_duplicate_after_last_h_descendant_removed(self):
        cache = self._cache()
        boundary = _attach(cache.root_node, _node([1], value=True, host=True))
        h_leaf = _attach(boundary, _node([2], value=False, host=True))
        cache.evictable_host_leaves.add(h_leaf)

        self.assertEqual(cache.evict_host(1), 1)
        cache.evictable_host_leaves.add(boundary)
        self.assertEqual(cache.evict_host(1), 1)

        self.assertIsNotNone(boundary.value)
        self.assertIsNone(boundary.host_value)
        cache.sanity_check_boundary_invariant()

    def test_l2_respects_host_ref_counter(self):
        cache = self._cache()
        protected = _attach(
            cache.root_node, _node([1], value=True, host=True, host_ref=1)
        )
        evictable = _attach(cache.root_node, _node([2], value=True, host=True))
        cache.evictable_host_leaves.update([protected, evictable])

        self.assertEqual(cache.evict_host(1), 1)

        self.assertIsNotNone(protected.host_value)
        self.assertIsNone(evictable.host_value)

    def test_hit_count_write_through_backs_up_inserted_node(self):
        cache = self._cache()
        node = _attach(cache.root_node, _node([1], value=True, host=False))
        cache.write_through_threshold = 1

        cache._inc_hit_count(node)

        self.assertEqual(node.hit_count, 1)
        self.assertIsNotNone(node.host_value)
        self.assertEqual(node.lock_ref, 1)
        self.assertEqual(node.write_through_pending_id, node.id)

    def test_write_through_boundaries_d_parent_before_h_child(self):
        cache = self._cache()
        parent = _attach(cache.root_node, _node([1], value=True, host=False))
        child = _attach(parent, _node([2], value=True, host=False))
        cache.write_through_threshold = 1

        cache._inc_hit_count(child)

        self.assertIsNotNone(parent.value)
        self.assertIsNotNone(parent.host_value)
        self.assertIsNotNone(child.value)
        self.assertIsNotNone(child.host_value)
        cache.sanity_check_boundary_invariant()

    def test_boundary_mode_can_use_host_pool_smaller_than_device_pool(self):
        device_pool = SimpleNamespace(
            size=10,
            store_dtype=torch.uint8,
            start_layer=0,
            end_layer=1,
        )

        host_pool = _TinyHostKVCache(
            device_pool,
            host_to_device_ratio=0.5,
            host_size=0,
            page_size=1,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )
        self.assertLess(host_pool.size, device_pool.size)


class TestBoundaryHiRadixCacheRegistry(CustomTestCase):
    def _ctx(self, *, storage_backend=None, enable_hierarchical_cache=True):
        from sglang.srt.mem_cache.registry import TreeCacheBuildContext

        server_args = MagicMock()
        server_args.radix_cache_backend = "boundary_hicache"
        server_args.enable_streaming_session = False
        server_args.hicache_storage_backend = storage_backend
        return TreeCacheBuildContext(
            server_args=server_args,
            params=MagicMock(),
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            enable_hierarchical_cache=enable_hierarchical_cache,
            disable_radix_cache=False,
            effective_chunked_prefill_size=None,
            tp_worker=MagicMock(),
            model_config=MagicMock(),
            tp_size=1,
            tp_rank=0,
            tp_group=MagicMock(),
        )

    def test_registered_backend_selects_boundary_cache(self):
        fake_module = MagicMock()
        with patch.dict(
            "sys.modules",
            {"sglang.srt.mem_cache.boundary_hiradix_cache": fake_module},
        ):
            result = create_tree_cache(self._ctx())

        fake_module.BoundaryHiRadixCache.assert_called_once()
        self.assertIs(result, fake_module.BoundaryHiRadixCache.return_value)

    def test_registered_backend_rejects_storage_backend(self):
        with self.assertRaisesRegex(ValueError, "does not support L3"):
            create_tree_cache(self._ctx(storage_backend="mooncake"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
