"""CPU-only tests for allocation-aware UnifiedRadixCache eviction."""

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import index_slots

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestUnifiedRadixAllocationEviction(CustomTestCase):
    @staticmethod
    def _build_cache(*, collateral_capacity_gain: int):
        cache = object.__new__(UnifiedRadixCache)
        cache.disable = False
        cache.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
        index_slots(cache, cache.tree_components)
        cache.is_swa_enabled = False
        cache.cache_controller = None
        cache.metrics_collector = None
        cache.tree_core = MagicMock()
        cache.tree_core.full_evictable_size.return_value = 80

        capacity = {"available": 30}
        allocator = MagicMock()
        allocator.available_size.side_effect = lambda: capacity["available"]
        # Pools that share no memory: the tree walks each target itself.
        allocator.reclaim_to_availability.return_value = None
        cache.token_to_kv_pool_allocator = allocator
        cache.req_to_token_pool = MagicMock()

        leaf_count = {"value": 0}

        def next_node(component_type, tracker):
            if tracker[component_type] >= 70:
                return None, False
            return leaf_count["value"] + 1, True

        def evict_leaf(_node_id, tracker):
            leaf_count["value"] += 1
            tracker[ComponentType.FULL] += 20
            tracker[ComponentType.MAMBA] += 1
            capacity["available"] += (
                collateral_capacity_gain if leaf_count["value"] == 1 else 20
            )
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)
        return cache, capacity, leaf_count

    def test_allocation_eviction_stops_when_shared_capacity_is_sufficient(self):
        cache, capacity, leaf_count = self._build_cache(collateral_capacity_gain=70)

        result = cache.evict_for_alloc(EvictParams(num_tokens=70))

        self.assertEqual(capacity["available"], 100)
        self.assertEqual(leaf_count["value"], 1)
        self.assertEqual(result.num_tokens_evicted, 20)
        self.assertEqual(result.mamba_num_evicted, 1)

    def test_allocation_eviction_runs_to_the_target_when_frees_stay_hidden(self):
        """An allocator can hide freed pages behind a clamp or page rounding, so
        freeing exactly the shortfall may leave its availability target unmet;
        an allocation-driven FULL walk must keep going until the target shows."""
        cache, capacity, leaf_count = self._build_cache(collateral_capacity_gain=0)
        cache.tree_core.full_evictable_size.return_value = 60
        quota = {}
        cache.tree_core.evict_device_start.side_effect = quota.__setitem__

        def next_node(component_type, tracker):
            # The tree yields victims only up to the quota the walk opened with.
            if tracker[component_type] >= quota[component_type]:
                return None, False
            return leaf_count["value"] + 1, True

        cache._evict_device_next_node.side_effect = next_node

        result = cache.evict_for_alloc(EvictParams(num_tokens=20))

        self.assertEqual(leaf_count["value"], 2)
        self.assertEqual(capacity["available"], 50)
        self.assertEqual(result.num_tokens_evicted, 40)
        cache.tree_core.evict_device_start.assert_called_once_with(
            ComponentType.FULL, 60
        )

    def test_explicit_evict_preserves_component_count_semantics(self):
        cache, _, leaf_count = self._build_cache(collateral_capacity_gain=70)

        result = cache.evict(EvictParams(num_tokens=70))

        self.assertEqual(leaf_count["value"], 4)
        self.assertEqual(result.num_tokens_evicted, 80)
        self.assertEqual(result.mamba_num_evicted, 4)

    def test_c128_component_keeps_zero_quota(self):
        cache, _, _ = self._build_cache(collateral_capacity_gain=70)
        cache.tree_components = (ComponentType.FULL, ComponentType.C128)
        cache._evict_device_next_node.side_effect = None
        cache._evict_device_next_node.return_value = (None, False)

        result = cache.evict(EvictParams(num_tokens=1))

        self.assertEqual(result.num_tokens_evicted, 0)
        cache.tree_core.evict_device_start.assert_called_once_with(
            ComponentType.FULL, 1
        )

    def test_mamba_allocation_counts_collateral_full_capacity(self):
        cache = object.__new__(UnifiedRadixCache)
        cache.disable = False
        cache.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
        index_slots(cache, cache.tree_components)
        cache.is_swa_enabled = False
        cache.cache_controller = None
        cache.metrics_collector = None
        cache.tree_core = MagicMock()
        cache.token_to_kv_pool_allocator = MagicMock()
        cache.token_to_kv_pool_allocator.reclaim_to_availability.return_value = None

        capacity = {"available": 0}
        mamba_allocator = MagicMock()
        mamba_allocator.schedulable_available_size.side_effect = lambda: capacity[
            "available"
        ]
        cache.token_to_kv_pool_allocator.state_slots = mamba_allocator
        cache.req_to_token_pool = MagicMock()

        def next_node(component_type, tracker):
            return (None, False) if tracker[component_type] >= 3 else (1, True)

        def evict_leaf(_node_id, tracker):
            tracker[ComponentType.FULL] += 20
            tracker[ComponentType.MAMBA] += 1
            capacity["available"] += 3
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=3))

        self.assertEqual(capacity["available"], 3)
        self.assertEqual(result.num_tokens_evicted, 20)
        self.assertEqual(result.mamba_num_evicted, 1)

    def test_common_helper_uses_allocation_aware_entry_point(self):
        tree_cache = MagicMock()
        tree_cache.supports_prefix_sharing.return_value = True
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        allocator = object.__new__(TokenToKVPoolAllocator)
        allocator.available_size = lambda: 30
        tree_cache.token_to_kv_pool_allocator = allocator

        evict_from_tree_cache(tree_cache, num_tokens=100)

        tree_cache.evict_for_alloc.assert_called_once_with(EvictParams(num_tokens=70))
        tree_cache.evict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
