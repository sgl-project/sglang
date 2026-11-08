"""CPU coverage for Unified host-source attribution.

The UnifiedTreeCore boundary uses NodeIds, while the tree itself keeps source
metadata on UnifiedTreeNode objects. These tests exercise both sides of that
boundary without requiring a host-pool backend.
"""

import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv.plan import TREE_TOKEN, TreeSlot
from sglang.srt.mem_cache.radix_cache import HostSource, RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    BASE_COMPONENT_TYPE,
    ComponentType,
    EvictLayer,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE = 4


def _make_cache() -> UnifiedRadixCache:
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=PAGE_SIZE,
        tree_slots=(TreeSlot(component="FULL", family=TREE_TOKEN),),
        tree_core_backend="python",
    )
    cache = UnifiedRadixCache(params)
    cache.tree_core.set_hicache_enabled()
    return cache


def _tokens(start: int, count: int) -> list[int]:
    return list(range(start, start + count))


def _add_device_node(cache, parent, start, count):
    core = cache.tree_core
    node = core._new_node()
    node.parent = parent
    node.key = RadixKey(array("q", _tokens(start, count)))
    node.component_data[BASE_COMPONENT_TYPE].value = torch.tensor(
        _tokens(start, count), dtype=torch.int64
    )
    parent.children[node.key.child_key(core.page_size)] = node
    core.component_evictable_size_[BASE_COMPONENT_TYPE] += count
    core._update_evictable_leaf_sets(node)
    return node


def _add_host_node(cache, parent, start, count, host_source):
    core = cache.tree_core
    node = core._new_node()
    node.parent = parent
    node.key = RadixKey(array("q", _tokens(start, count)))
    node.component_data[BASE_COMPONENT_TYPE].host_value = torch.arange(
        count, dtype=torch.int64
    )
    node.host_source = host_source
    parent.children[node.key.child_key(core.page_size)] = node
    core._update_evictable_leaf_sets(node)
    return node


def _match(cache, count) -> object:
    # RadixKey is page-aligned by the current UnifiedTreeCore matcher.
    return cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", _tokens(0, count))))
    )


class TestBackupSource(CustomTestCase):
    def test_backup_success_marks_device_backup(self):
        cache = _make_cache()
        node = _add_device_node(cache, cache.tree_core.root_node, 0, PAGE_SIZE)
        host_indices = torch.arange(PAGE_SIZE, dtype=torch.int64)
        cache._execute_kv_backup = mock.Mock(return_value=host_indices)

        action = cache.tree_core._build_backup_kv_action(node, write_back=True)
        written = cache._execute_and_commit_kv_backup(action, write_back=True)

        self.assertEqual(written, PAGE_SIZE)
        self.assertEqual(node.host_source, HostSource.DEVICE_BACKUP)
        self.assertIsNotNone(node.component_data[BASE_COMPONENT_TYPE].host_value)

    def test_backup_failure_does_not_mark(self):
        cache = _make_cache()
        node = _add_device_node(cache, cache.tree_core.root_node, 0, PAGE_SIZE)
        cache._execute_kv_backup = mock.Mock(return_value=None)

        action = cache.tree_core._build_backup_kv_action(node, write_back=True)
        self.assertEqual(
            cache._execute_and_commit_kv_backup(action, write_back=True), 0
        )
        self.assertIsNone(node.host_source)
        self.assertIsNone(node.component_data[BASE_COMPONENT_TYPE].host_value)


class TestPrefetchInsertSource(CustomTestCase):
    def test_prefetch_refills_evicted_host_copy_with_new_source(self):
        cache = _make_cache()
        core = cache.tree_core
        node = _add_device_node(cache, core.root_node, 0, PAGE_SIZE)
        node.host_source = None
        host_value = torch.arange(PAGE_SIZE, dtype=torch.int64) + 100

        result = core.insert_host(core.root_node.id, node.key, host_value, ["h0"])

        self.assertEqual(result.inserted_host_node, node.id)
        self.assertEqual(node.host_source, HostSource.L3_PREFETCH)
        torch.testing.assert_close(
            node.component_data[BASE_COMPONENT_TYPE].host_value, host_value
        )

    def test_new_host_insert_marks_l3_prefetch(self):
        cache = _make_cache()
        key = RadixKey(array("q", _tokens(0, PAGE_SIZE)))
        host_value = torch.arange(PAGE_SIZE, dtype=torch.int64) + 100

        result = cache.tree_core.insert_host(
            cache.tree_core.root_node.id, key, host_value, ["h0"]
        )

        new_node = cache.tree_core.node_by_id(result.inserted_host_node)
        self.assertEqual(new_node.host_source, HostSource.L3_PREFETCH)
        self.assertEqual(
            len(new_node.component_data[BASE_COMPONENT_TYPE].host_value), PAGE_SIZE
        )

    def test_fully_overlapping_reinsert_keeps_old_source(self):
        cache = _make_cache()
        key = RadixKey(array("q", _tokens(0, PAGE_SIZE)))
        first = cache.tree_core.insert_host(
            cache.tree_core.root_node.id,
            key,
            torch.arange(PAGE_SIZE, dtype=torch.int64) + 100,
            ["h0"],
        )
        node = cache.tree_core.node_by_id(first.inserted_host_node)
        node.host_source = HostSource.DEVICE_BACKUP

        result = cache.tree_core.insert_host(
            cache.tree_core.root_node.id,
            key,
            torch.arange(PAGE_SIZE, dtype=torch.int64) + 200,
            ["h0"],
        )

        self.assertEqual(result.inserted_host_node, node.id)
        self.assertEqual(node.host_source, HostSource.DEVICE_BACKUP)


class TestSplitInheritance(CustomTestCase):
    def test_split_parent_inherits_child_source(self):
        cache = _make_cache()
        child = _add_host_node(
            cache, cache.tree_core.root_node, 0, 2 * PAGE_SIZE, HostSource.DEVICE_BACKUP
        )

        new_node, _ = cache.tree_core._split_node(child.key, child, PAGE_SIZE)

        self.assertEqual(new_node.host_source, HostSource.DEVICE_BACKUP)
        self.assertEqual(child.host_source, HostSource.DEVICE_BACKUP)
        self.assertEqual(
            len(new_node.component_data[BASE_COMPONENT_TYPE].host_value), PAGE_SIZE
        )
        self.assertEqual(
            len(child.component_data[BASE_COMPONENT_TYPE].host_value), PAGE_SIZE
        )

    def test_split_inherits_l3_prefetch_source(self):
        cache = _make_cache()
        child = _add_host_node(
            cache, cache.tree_core.root_node, 0, 2 * PAGE_SIZE, HostSource.L3_PREFETCH
        )

        new_node, _ = cache.tree_core._split_node(child.key, child, PAGE_SIZE)

        self.assertEqual(new_node.host_source, HostSource.L3_PREFETCH)
        self.assertEqual(child.host_source, HostSource.L3_PREFETCH)


class TestHostEvictClearsSource(CustomTestCase):
    def test_host_leaf_eviction_clears_host_source(self):
        cache = _make_cache()
        parent = _add_device_node(cache, cache.tree_core.root_node, 0, PAGE_SIZE)
        node = _add_host_node(
            cache, parent, PAGE_SIZE, PAGE_SIZE, HostSource.L3_PREFETCH
        )
        tracker = {ComponentType.FULL: 0}
        device_frees = {ComponentType.FULL: []}
        host_frees = {ComponentType.FULL: []}

        cache.tree_core._evict_host_leaf(node, tracker, device_frees, host_frees, [])

        self.assertIsNone(node.host_source)
        self.assertIsNone(node.component_data[BASE_COMPONENT_TYPE].host_value)
        self.assertEqual(tracker[ComponentType.FULL], PAGE_SIZE)
        self.assertNotIn(node.key.child_key(cache.tree_core.page_size), parent.children)

    def test_host_component_evict_clears_host_source(self):
        cache = _make_cache()
        parent = _add_device_node(cache, cache.tree_core.root_node, 0, PAGE_SIZE)
        node = _add_host_node(
            cache, parent, PAGE_SIZE, PAGE_SIZE, HostSource.DEVICE_BACKUP
        )
        tracker = {ComponentType.FULL: 0}
        device_frees = {ComponentType.FULL: []}
        host_frees = {ComponentType.FULL: []}

        cache.tree_core._evict_component_and_detach_lru(
            node,
            cache.tree_core.components_by_type[ComponentType.FULL],
            device_frees=device_frees,
            host_frees=host_frees,
            target=EvictLayer.HOST,
            tracker=tracker,
            cache_actions=[],
        )

        self.assertIsNone(node.host_source)
        self.assertIsNone(node.component_data[BASE_COMPONENT_TYPE].host_value)
        self.assertEqual(tracker[ComponentType.FULL], PAGE_SIZE)
        self.assertIn(node.key.child_key(cache.tree_core.page_size), parent.children)


class TestMatchAttribution(CustomTestCase):
    def _assert_attribution(self, host_source, expected_wb, expected_pf):
        cache = _make_cache()
        parent = _add_device_node(cache, cache.tree_core.root_node, 0, PAGE_SIZE)
        _add_host_node(cache, parent, PAGE_SIZE, PAGE_SIZE, host_source)

        result = _match(cache, 2 * PAGE_SIZE)

        self.assertEqual(result.host_hit_length, PAGE_SIZE)
        self.assertEqual(result.host_hit_write_back, expected_wb)
        self.assertEqual(result.host_hit_prefetch, expected_pf)
        self.assertEqual(
            result.host_hit_write_back + result.host_hit_prefetch,
            PAGE_SIZE if host_source is not None else 0,
        )

    def test_pure_write_back_hit(self):
        self._assert_attribution(HostSource.DEVICE_BACKUP, PAGE_SIZE, 0)

    def test_pure_prefetch_hit(self):
        self._assert_attribution(HostSource.L3_PREFETCH, 0, PAGE_SIZE)

    def test_legacy_untagged_hit_has_unknown_provenance(self):
        self._assert_attribution(None, 0, 0)

    def test_mixed_hits_along_a_chain(self):
        cache = _make_cache()
        n1 = _add_device_node(cache, cache.tree_core.root_node, 0, PAGE_SIZE)
        n2 = _add_host_node(cache, n1, PAGE_SIZE, PAGE_SIZE, HostSource.L3_PREFETCH)
        _add_host_node(cache, n2, 2 * PAGE_SIZE, PAGE_SIZE, HostSource.DEVICE_BACKUP)

        result = _match(cache, 3 * PAGE_SIZE)

        self.assertEqual(result.host_hit_length, 2 * PAGE_SIZE)
        self.assertEqual(result.host_hit_write_back, PAGE_SIZE)
        self.assertEqual(result.host_hit_prefetch, PAGE_SIZE)
        self.assertEqual(
            result.host_hit_write_back + result.host_hit_prefetch,
            result.host_hit_length,
        )

    def test_ranges_preserve_prefix_order_and_forced_miss_clears_them(self):
        from sglang.srt.mem_cache.base_prefix_cache import zero_match_result

        cache = _make_cache()
        device = _add_device_node(cache, cache.tree_core.root_node, 0, PAGE_SIZE)
        prefetched = _add_host_node(
            cache, device, PAGE_SIZE, PAGE_SIZE, HostSource.L3_PREFETCH
        )
        _add_host_node(
            cache, prefetched, 2 * PAGE_SIZE, PAGE_SIZE, HostSource.DEVICE_BACKUP
        )
        result = _match(cache, 3 * PAGE_SIZE)
        self.assertEqual(
            result.host_hit_source_ranges,
            (
                (PAGE_SIZE, 2 * PAGE_SIZE, HostSource.L3_PREFETCH),
                (2 * PAGE_SIZE, 3 * PAGE_SIZE, HostSource.DEVICE_BACKUP),
            ),
        )
        miss = zero_match_result(cache, result)
        self.assertEqual(
            (
                miss.host_hit_write_back,
                miss.host_hit_prefetch,
                miss.host_hit_source_ranges,
            ),
            (0, 0, ()),
        )

    def test_no_host_hit_keeps_breakdown_zero(self):
        cache = _make_cache()
        _add_device_node(cache, cache.tree_core.root_node, 0, PAGE_SIZE)

        result = _match(cache, PAGE_SIZE)

        self.assertEqual(result.host_hit_length, 0)
        self.assertEqual(result.host_hit_write_back, 0)
        self.assertEqual(result.host_hit_prefetch, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
