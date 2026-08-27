"""CPU-only tests for allocation-aware UnifiedRadixCache eviction."""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaSWATokenToKVPoolAllocator,
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestUnifiedRadixAllocationEviction(CustomTestCase):
    @staticmethod
    def _build_cache(*, collateral_capacity_gain: int):
        cache = object.__new__(UnifiedRadixCache)
        cache.disable = False
        cache.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
        cache.is_swa_enabled = False
        cache.cache_controller = None
        cache.metrics_collector = None
        cache.tree_core = MagicMock()

        capacity = {"available": 30}
        allocator = MagicMock()
        allocator.available_size.side_effect = lambda: capacity["available"]
        allocator.mamba_full_cache_donor.return_value = None
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

    @staticmethod
    def _build_unified_mamba_donor_cache(
        *, free_ids: int, byte_slots: int, tri_pool: bool = False
    ):
        cache = object.__new__(UnifiedRadixCache)
        cache.disable = False
        cache.tree_components = (
            (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
            if tri_pool
            else (ComponentType.FULL, ComponentType.MAMBA)
        )
        cache.is_swa_enabled = tri_pool
        cache.cache_controller = None
        cache.metrics_collector = None
        cache.tree_core = MagicMock()
        cache.tree_core.full_evictable_size.return_value = 16
        cache.tree_core.mamba_evictable_size.return_value = 8

        capacity = {"free_ids": free_ids, "byte_slots": byte_slots}
        allocator_cls = (
            UnifiedMambaSWATokenToKVPoolAllocator
            if tri_pool
            else UnifiedMambaTokenToKVPoolAllocator
        )
        allocator = object.__new__(allocator_cls)
        allocator.free_group = None
        allocator.free_page_reps_group = None
        allocator.full_attn_allocator = MagicMock()
        allocator.full_attn_allocator.schedulable_available_size.return_value = 100
        allocator.mamba_allocator = MagicMock()
        allocator.mamba_allocator.available_size.side_effect = lambda: capacity[
            "free_ids"
        ]
        allocator.mamba_allocator.schedulable_available_size.side_effect = lambda: min(
            capacity["free_ids"], capacity["byte_slots"]
        )
        cache.token_to_kv_pool_allocator = allocator
        cache.req_to_token_pool = MagicMock(mamba_allocator=allocator.mamba_allocator)
        return cache, capacity, allocator

    def test_allocation_eviction_stops_when_shared_capacity_is_sufficient(self):
        cache, capacity, leaf_count = self._build_cache(collateral_capacity_gain=70)

        result = cache.evict_for_alloc(EvictParams(num_tokens=70))

        self.assertEqual(capacity["available"], 100)
        self.assertEqual(leaf_count["value"], 1)
        self.assertEqual(result.num_tokens_evicted, 20)
        self.assertEqual(result.mamba_num_evicted, 1)

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
        cache.is_swa_enabled = False
        cache.cache_controller = None
        cache.metrics_collector = None
        cache.tree_core = MagicMock()
        cache.token_to_kv_pool_allocator = MagicMock()
        cache.token_to_kv_pool_allocator.mamba_full_cache_donor.return_value = None

        capacity = {"available": 0}
        mamba_allocator = MagicMock()
        mamba_allocator.schedulable_available_size.side_effect = lambda: capacity[
            "available"
        ]
        cache.req_to_token_pool = MagicMock(mamba_allocator=mamba_allocator)

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

    def test_mamba_allocation_uses_full_as_donor_for_byte_shortfall(self):
        cache, capacity, _ = self._build_unified_mamba_donor_cache(
            free_ids=2, byte_slots=1
        )

        def next_node(component_type, tracker):
            if (
                component_type == ComponentType.FULL
                and tracker[ComponentType.FULL] < 16
            ):
                return 1, True
            return None, False

        def evict_leaf(_node_id, tracker):
            tracker[ComponentType.FULL] += 4
            capacity["byte_slots"] += 1
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(capacity, {"free_ids": 2, "byte_slots": 2})
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.mamba_num_evicted, 0)
        self.assertEqual(
            cache.tree_core.evict_device_start.call_args_list,
            [
                unittest.mock.call(ComponentType.FULL, 16),
            ],
        )

    def test_mamba_allocation_recycles_mamba_for_id_shortfall(self):
        cache, capacity, _ = self._build_unified_mamba_donor_cache(
            free_ids=0, byte_slots=1
        )

        def next_node(component_type, tracker):
            if (
                component_type == ComponentType.MAMBA
                and tracker[ComponentType.MAMBA] < 1
            ):
                return 1, True
            return None, False

        def evict_leaf(_node_id, tracker):
            tracker[ComponentType.MAMBA] += 1
            capacity["free_ids"] += 1
            capacity["byte_slots"] += 1
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(capacity, {"free_ids": 1, "byte_slots": 2})
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(result.mamba_num_evicted, 1)
        cache.tree_core.evict_device_start.assert_called_once_with(
            ComponentType.MAMBA, 1
        )

    def test_mamba_allocation_does_not_evict_full_while_id_bound(self):
        cache, capacity, _ = self._build_unified_mamba_donor_cache(
            free_ids=0, byte_slots=10
        )

        def next_node(component_type, tracker):
            if component_type == ComponentType.FULL and tracker[ComponentType.FULL] < 4:
                return 1, True
            return None, False

        def evict_leaf(_node_id, tracker):
            tracker[ComponentType.FULL] += 4
            capacity["byte_slots"] += 1
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(capacity, {"free_ids": 0, "byte_slots": 10})
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(result.mamba_num_evicted, 0)
        cache.tree_core.evict_device_start.assert_called_once_with(
            ComponentType.MAMBA, 1
        )

    def test_mamba_allocation_does_not_evict_full_after_partial_id_recovery(self):
        cache, capacity, _ = self._build_unified_mamba_donor_cache(
            free_ids=0, byte_slots=10
        )

        def next_node(component_type, tracker):
            if (
                component_type == ComponentType.MAMBA
                and tracker[ComponentType.MAMBA] < 1
            ):
                return 1, True
            if component_type == ComponentType.FULL and tracker[ComponentType.FULL] < 4:
                return 2, True
            return None, False

        def evict_leaf(node_id, tracker):
            if node_id == 1:
                tracker[ComponentType.MAMBA] += 1
                capacity["free_ids"] += 1
                capacity["byte_slots"] += 1
            else:
                tracker[ComponentType.FULL] += 4
                capacity["byte_slots"] += 1
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=2))

        self.assertEqual(capacity, {"free_ids": 1, "byte_slots": 11})
        self.assertEqual(result.num_tokens_evicted, 0)
        self.assertEqual(result.mamba_num_evicted, 1)
        cache.tree_core.evict_device_start.assert_called_once_with(
            ComponentType.MAMBA, 2
        )

    def test_mamba_allocation_splits_mixed_id_and_byte_pressure(self):
        cache, capacity, _ = self._build_unified_mamba_donor_cache(
            free_ids=3, byte_slots=2
        )

        def next_node(component_type, tracker):
            if (
                component_type == ComponentType.MAMBA
                and tracker[ComponentType.MAMBA] < 1
            ):
                return 1, True
            if component_type == ComponentType.FULL and tracker[ComponentType.FULL] < 4:
                return 2, True
            return None, False

        def evict_leaf(node_id, tracker):
            if node_id == 1:
                tracker[ComponentType.MAMBA] += 1
                capacity["free_ids"] += 1
                capacity["byte_slots"] += 1
            else:
                tracker[ComponentType.FULL] += 4
                capacity["byte_slots"] += 1
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=2))

        self.assertEqual(capacity, {"free_ids": 4, "byte_slots": 4})
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertEqual(
            cache.tree_core.evict_device_start.call_args_list,
            [
                unittest.mock.call(ComponentType.MAMBA, 1),
                unittest.mock.call(ComponentType.FULL, 16),
            ],
        )

    def test_full_donor_walk_continues_until_mamba_capacity_is_visible(self):
        cache, capacity, _ = self._build_unified_mamba_donor_cache(
            free_ids=2, byte_slots=1
        )
        leaf_count = 0

        def next_node(component_type, tracker):
            if (
                component_type == ComponentType.FULL
                and tracker[ComponentType.FULL] < 16
            ):
                return tracker[ComponentType.FULL] // 4 + 1, True
            return None, False

        def evict_leaf(_node_id, tracker):
            nonlocal leaf_count
            leaf_count += 1
            tracker[ComponentType.FULL] += 4
            if leaf_count == 2:
                capacity["byte_slots"] += 1
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(leaf_count, 2)
        self.assertEqual(result.num_tokens_evicted, 8)
        cache.tree_core.evict_device_start.assert_called_once_with(
            ComponentType.FULL, 16
        )

    def test_mamba_cache_is_last_resort_when_full_donor_is_exhausted(self):
        cache, capacity, _ = self._build_unified_mamba_donor_cache(
            free_ids=2, byte_slots=0
        )
        cache.tree_core.full_evictable_size.return_value = 4

        def next_node(component_type, tracker):
            if component_type == ComponentType.FULL and tracker[component_type] < 4:
                return 1, True
            if component_type == ComponentType.MAMBA and tracker[component_type] < 8:
                return 2, True
            return None, False

        def evict_leaf(node_id, tracker):
            if node_id == 1:
                tracker[ComponentType.FULL] += 4
            else:
                tracker[ComponentType.MAMBA] += 1
                capacity["free_ids"] += 1
                capacity["byte_slots"] += 1
            return None

        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(capacity, {"free_ids": 3, "byte_slots": 1})
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.mamba_num_evicted, 1)
        self.assertEqual(
            cache.tree_core.evict_device_start.call_args_list,
            [
                unittest.mock.call(ComponentType.FULL, 4),
                unittest.mock.call(ComponentType.MAMBA, 8),
            ],
        )

    def test_full_donor_flushes_grouped_frees_before_capacity_recheck(self):
        cache, capacity, allocator = self._build_unified_mamba_donor_cache(
            free_ids=2, byte_slots=1
        )
        allocator.free_group_begin()
        allocator.full_attn_allocator.free.side_effect = lambda _indices: (
            capacity.update(byte_slots=capacity["byte_slots"] + 1)
        )
        allocator.mamba_allocator.alloc.side_effect = lambda need_size: (
            torch.arange(need_size)
            if min(capacity["free_ids"], capacity["byte_slots"]) >= need_size
            else None
        )

        cache._evict_device_next_node = MagicMock(return_value=(1, True))

        def evict_leaf(_node_id, tracker):
            tracker[ComponentType.FULL] += 4
            allocator.free(torch.tensor([1], dtype=torch.int64))
            return None

        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(capacity["byte_slots"], 2)
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(allocator.free_group, [])
        self.assertEqual(allocator.free_page_reps_group, [])
        allocator.full_attn_allocator.free.assert_called_once()
        self.assertIsNotNone(allocator.mamba_allocator.alloc(2))
        allocator.free_group_end()

    def test_tri_pool_uses_the_same_full_donor_capability(self):
        cache, capacity, allocator = self._build_unified_mamba_donor_cache(
            free_ids=2, byte_slots=1, tri_pool=True
        )

        cache._evict_device_next_node = MagicMock(return_value=(1, True))

        def evict_leaf(_node_id, tracker):
            tracker[ComponentType.FULL] += 4
            capacity["byte_slots"] += 1
            return None

        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertIs(allocator.mamba_full_cache_donor(), allocator)
        self.assertEqual(capacity["byte_slots"], 2)
        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.mamba_num_evicted, 0)
        cache.tree_core.evict_device_start.assert_called_once_with(
            ComponentType.FULL, 16
        )

    def test_common_helper_uses_allocation_aware_entry_point(self):
        tree_cache = MagicMock()
        tree_cache.is_chunk_cache.return_value = False
        tree_cache.token_to_kv_pool_allocator.available_size.return_value = 30

        evict_from_tree_cache(tree_cache, num_tokens=100)

        tree_cache.evict_for_alloc.assert_called_once_with(EvictParams(num_tokens=70))
        tree_cache.evict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
