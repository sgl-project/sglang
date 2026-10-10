"""CPU-only tests for allocation-aware UnifiedRadixCache eviction."""

import random
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        if tri_pool:
            allocator.full_page_reps_group = []
            allocator.full_free_group = []
        allocator.full_attn_allocator = MagicMock()
        allocator.full_attn_allocator.schedulable_available_size.return_value = 100
        allocator.mamba_allocator = MagicMock()
        allocator.mamba_allocator.available_size.side_effect = lambda: capacity[
            "free_ids"
        ]
        allocator.mamba_allocator.schedulable_available_size.side_effect = lambda: min(
            capacity["free_ids"], capacity["byte_slots"]
        )
        allocator.full_tokens_before_mamba_recheck = MagicMock(return_value=1)
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

    def test_full_donor_defers_preparation_until_safe_lower_bound(self):
        cache, capacity, allocator = self._build_unified_mamba_donor_cache(
            free_ids=2, byte_slots=1
        )
        leaf_count = 0

        allocator.full_tokens_before_mamba_recheck.return_value = 8

        def prepare(_target_size):
            if leaf_count >= 2:
                capacity["byte_slots"] = 2

        allocator.prepare_mamba_allocation = MagicMock(side_effect=prepare)
        cache._evict_device_next_node = MagicMock(return_value=(1, True))

        def evict_leaf(_node_id, tracker):
            nonlocal leaf_count
            leaf_count += 1
            tracker[ComponentType.FULL] += 4
            return None

        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(leaf_count, 2)
        self.assertEqual(result.num_tokens_evicted, 8)
        # One initial layout preparation plus one check at the lower bound.
        self.assertEqual(allocator.prepare_mamba_allocation.call_count, 2)

    def test_full_donor_observes_cascade_before_lower_bound(self):
        cache, capacity, allocator = self._build_unified_mamba_donor_cache(
            free_ids=2, byte_slots=1
        )
        allocator.full_tokens_before_mamba_recheck.return_value = 100
        allocator.prepare_mamba_allocation = MagicMock()
        cache._evict_device_next_node = MagicMock(return_value=(1, True))

        def evict_leaf(_node_id, tracker):
            tracker[ComponentType.FULL] += 4
            tracker[ComponentType.MAMBA] += 1
            capacity["byte_slots"] = 2
            return None

        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(result.num_tokens_evicted, 4)
        self.assertEqual(result.mamba_num_evicted, 1)
        # Only the pre-donor preparation runs; the cheap capacity check stops
        # before the Full-byte lower bound after the Mamba cascade is visible.
        allocator.prepare_mamba_allocation.assert_called_once()

    def test_full_donor_rechecks_each_leaf_after_lower_bound(self):
        cache, capacity, allocator = self._build_unified_mamba_donor_cache(
            free_ids=2, byte_slots=1
        )
        leaf_count = 0

        allocator.full_tokens_before_mamba_recheck.return_value = 8

        def prepare(_target_size):
            if leaf_count >= 3:
                capacity["byte_slots"] = 2

        allocator.prepare_mamba_allocation = MagicMock(side_effect=prepare)
        cache._evict_device_next_node = MagicMock(return_value=(1, True))

        def evict_leaf(_node_id, tracker):
            nonlocal leaf_count
            leaf_count += 1
            tracker[ComponentType.FULL] += 4
            return None

        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)

        result = cache.evict_for_alloc(EvictParams(mamba_num=1))

        self.assertEqual(leaf_count, 3)
        self.assertEqual(result.num_tokens_evicted, 12)
        # The failed lower-bound check falls back to leaf-granular checks.
        self.assertEqual(allocator.prepare_mamba_allocation.call_count, 3)

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
        tree_cache.supports_prefix_sharing.return_value = True
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

        allocator = object.__new__(TokenToKVPoolAllocator)
        allocator.available_size = lambda: 30
        tree_cache.token_to_kv_pool_allocator = allocator

        evict_from_tree_cache(tree_cache, num_tokens=100)

        tree_cache.evict_for_alloc.assert_called_once_with(EvictParams(num_tokens=70))
        tree_cache.evict.assert_not_called()

    @staticmethod
    def _build_demand_cache(*, fits_after, view_gain=0):
        """FULL/SWA cache over a mocked allocator: the allocation fits once
        ``fits_after`` leaves are gone (never when None), and each evicted leaf
        adds ``view_gain`` to both per-side views."""
        cache = object.__new__(UnifiedRadixCache)
        cache.disable = False
        cache.tree_components = (ComponentType.FULL, ComponentType.SWA)
        cache.is_swa_enabled = True
        cache.cache_controller = None
        cache.metrics_collector = None
        cache.tree_core = MagicMock()
        cache.req_to_token_pool = MagicMock()

        walk = {"component": None, "quota": 0}
        evicted = {ComponentType.FULL: 0, ComponentType.SWA: 0}

        def start(component_type, quota):
            walk.update(component=component_type, quota=quota)

        def gone():
            return sum(evicted.values())

        allocator = MagicMock()
        allocator.full_available_size.side_effect = lambda: gone() * view_gain
        allocator.swa_available_size.side_effect = lambda: gone() * view_gain
        allocator.allocation_fits.side_effect = lambda full, swa: (
            fits_after is not None and gone() >= fits_after
        )
        cache.token_to_kv_pool_allocator = allocator

        def next_node(component_type, tracker):
            if tracker[component_type] >= walk["quota"]:
                return None, False
            return gone() + 1, True

        def evict_leaf(_node_id, tracker):
            tracker[walk["component"]] += 1
            evicted[walk["component"]] += 1
            return None

        cache.tree_core.evict_device_start.side_effect = start
        cache._evict_device_next_node = MagicMock(side_effect=next_node)
        cache._evict_device_leaf = MagicMock(side_effect=evict_leaf)
        return cache, allocator, evicted

    def test_alloc_demand_stops_on_the_allocation_fit_not_the_views(self):
        # Views that credit every leaf generously stop the per-side walk after
        # one leaf; views that never move run it to its quota.
        for view_gain, per_side_leaves in ((100, 1), (0, 10)):
            with self.subTest(view_gain=view_gain):
                cache, allocator, evicted = self._build_demand_cache(
                    fits_after=4, view_gain=view_gain
                )
                cache.evict_for_alloc(EvictParams(num_tokens=10))
                self.assertEqual(evicted[ComponentType.FULL], per_side_leaves)

                cache, allocator, evicted = self._build_demand_cache(
                    fits_after=4, view_gain=view_gain
                )
                result = cache.evict_for_alloc(
                    EvictParams(num_tokens=10, alloc_demand=(7, 5))
                )
                self.assertEqual(evicted[ComponentType.FULL], 4)
                self.assertEqual(result.num_tokens_evicted, 4)
                allocator.allocation_fits.assert_called_with(7, 5)

    def test_alloc_demand_stops_the_swa_walk_too(self):
        cache, _, evicted = self._build_demand_cache(fits_after=3)

        cache.evict_for_alloc(
            EvictParams(num_tokens=0, swa_num_tokens=10, alloc_demand=(0, 3))
        )

        self.assertEqual(evicted, {ComponentType.FULL: 0, ComponentType.SWA: 3})

    def test_alloc_demand_keeps_the_counts_as_caps(self):
        cache, _, evicted = self._build_demand_cache(fits_after=None)

        cache.evict_for_alloc(
            EvictParams(num_tokens=3, swa_num_tokens=2, alloc_demand=(9, 9))
        )

        self.assertEqual(evicted, {ComponentType.FULL: 3, ComponentType.SWA: 2})

    def test_alloc_demand_rechecks_after_a_write_back_demote(self):
        cache, _, evicted = self._build_demand_cache(fits_after=2)
        cache._evict_device_leaf.side_effect = lambda _node_id, _tracker: MagicMock()
        cache._execute_and_commit_kv_backup = MagicMock(return_value=1)
        cache.writing_check = MagicMock()

        def demote(_node_id, tracker):
            tracker[ComponentType.FULL] += 1
            evicted[ComponentType.FULL] += 1

        cache._demote = MagicMock(side_effect=demote)

        cache.evict_for_alloc(EvictParams(num_tokens=10, alloc_demand=(4, 4)))

        self.assertEqual(cache._demote.call_count, 2)
        self.assertEqual(evicted[ComponentType.FULL], 2)


def _full_swa_allocator(*, page_size: int, pages: int):
    """A unified FULL/SWA pool whose SWA entry is 8x its FULL entry, like MiMo's."""
    return init_unified_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=8,
        v_head_dim=8,
        swa_head_num=1,
        swa_head_dim=8,
        swa_v_head_dim=8,
        page_size=page_size,
        start_layer=0,
        end_layer=9,
        swa_attention_layer_ids=list(range(1, 9)),
        full_attention_layer_ids=[0],
        # 32 B of FULL and 256 B of SWA per token.
        total_bytes=pages * page_size * 288,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=True,
    ).token_to_kv_pool_allocator


class _CachedNodes:
    """The tree's device eviction walk over cached nodes, oldest first. A FULL
    eviction frees both sides of a node; a SWA eviction tombstones its SWA."""

    def __init__(self, allocator, node_pages, *, swa_dead=(), locked=()):
        ps = allocator.page_size
        live = allocator.alloc(sum(node_pages) * ps)
        assert live is not None
        self.allocator = allocator
        self.ids = list(torch.split(live, [n * ps for n in node_pages]))
        self.full = [True] * len(node_pages)
        self.swa = [True] * len(node_pages)
        self.locked = set(locked)
        for i in swa_dead:
            allocator.free_swa(self.ids[i])
            self.swa[i] = False
        self.evicted = {ComponentType.FULL: 0, ComponentType.SWA: 0}
        self._component = None
        self._quota = 0

    def _candidates(self, component_type):
        live = self.full if component_type == ComponentType.FULL else self.swa
        return [i for i, on in enumerate(live) if on and i not in self.locked]

    def evictable(self, component_type) -> int:
        return sum(len(self.ids[i]) for i in self._candidates(component_type))

    def start(self, component_type, quota):
        self._component, self._quota = component_type, quota

    def next_node(self, component_type, tracker):
        candidates = self._candidates(component_type)
        if tracker[component_type] >= self._quota or not candidates:
            return None, False
        return candidates[0], True

    def evict_leaf(self, node_id, tracker):
        n = len(self.ids[node_id])
        if self._component == ComponentType.FULL:
            self.allocator.free(self.ids[node_id])
            tracker[ComponentType.FULL] += n
            if self.swa[node_id]:
                tracker[ComponentType.SWA] += n
            self.full[node_id] = self.swa[node_id] = False
        else:
            self.allocator.free_swa(self.ids[node_id])
            tracker[ComponentType.SWA] += n
            self.swa[node_id] = False
        self.evicted[self._component] += n
        return None


def _cache_over(nodes: _CachedNodes) -> UnifiedRadixCache:
    cache = object.__new__(UnifiedRadixCache)
    cache.disable = False
    cache.tree_components = (ComponentType.FULL, ComponentType.SWA)
    cache.is_swa_enabled = True
    cache.cache_controller = None
    cache.metrics_collector = None
    cache.token_to_kv_pool_allocator = nodes.allocator
    cache.req_to_token_pool = MagicMock()
    cache.tree_core = MagicMock()
    cache.tree_core.full_evictable_size.side_effect = lambda: nodes.evictable(
        ComponentType.FULL
    )
    cache.tree_core.swa_evictable_size.side_effect = lambda: nodes.evictable(
        ComponentType.SWA
    )
    cache.tree_core.evict_device_start.side_effect = nodes.start
    cache._evict_device_next_node = nodes.next_node
    cache._evict_device_leaf = nodes.evict_leaf
    return cache


class TestSharedPoolMakeRoom(CustomTestCase):
    """Every cached token holds bytes on both sides of the unified FULL/SWA pool,
    so making room must continue until the allocation fits the two jointly."""

    def test_make_room_evicts_until_the_allocation_fits(self):
        # Each case fills the pool with one-page nodes, so the minimal eviction
        # is exactly the demand's page count.
        for page_size, pages, demand in ((1, 64, 10), (1, 64, 30), (4, 16, 12)):
            for entry in ("alloc", "prealloc", "decode_check"):
                with self.subTest(page_size=page_size, demand=demand, entry=entry):
                    allocator = _full_swa_allocator(page_size=page_size, pages=pages)
                    nodes = _CachedNodes(
                        allocator, [1] * (allocator.available_size() // page_size)
                    )
                    cache = _cache_over(nodes)
                    if entry == "alloc":
                        ok = allocator.evict_to_free_tokens(cache, demand)
                    elif entry == "prealloc":
                        ok = (
                            allocator.reclaim_for_prealloc(cache, demand, demand)
                            is None
                        )
                    else:
                        ok = allocator.check_decode_capacity(
                            num_tokens=demand, tree_cache=cache
                        )
                    evicted_pages = nodes.evicted[ComponentType.FULL] // page_size
                    self.assertEqual((ok, evicted_pages), (True, demand // page_size))
                    self.assertIsNotNone(allocator.alloc(demand))

    @staticmethod
    def _set_moves_allowed(allocator, allowed: bool):
        for sub_pool in (allocator.full_attn_allocator, allocator.swa_attn_allocator):
            sub_pool.disagg_move_gate = lambda: allowed

    @staticmethod
    def _layout(allocator):
        fa, sa = allocator.full_attn_allocator, allocator.swa_attn_allocator
        return (
            fa.watermark_physical,
            sa.watermark_physical,
            len(fa._free_phys_pages),
            len(sa._free_phys_pages),
        )

    def test_allocation_fits_is_pure_and_agrees_with_ensure_capacity(self):
        compacted = {True: 0, False: 0}
        for page_size, pages in ((1, 24), (4, 12)):
            for moves_allowed in (True, False):
                for full_pages in range(8):
                    for swa_pages in range(8):
                        allocator = _full_swa_allocator(
                            page_size=page_size, pages=pages
                        )
                        self._set_moves_allowed(allocator, moves_allowed)
                        nodes = _CachedNodes(
                            allocator,
                            [1] * (allocator.available_size() // page_size),
                            swa_dead=(2, 5, 7),
                        )
                        # Interior holes: two on both sides, three on SWA only.
                        allocator.free(nodes.ids[1])
                        allocator.free(nodes.ids[4])
                        before = self._layout(allocator)
                        demand = (full_pages * page_size, swa_pages * page_size)
                        with self.subTest(
                            page_size=page_size,
                            moves_allowed=moves_allowed,
                            demand=demand,
                        ):
                            fits = allocator.allocation_fits(*demand)
                            self.assertEqual(self._layout(allocator), before)
                            self.assertEqual(allocator.ensure_capacity(*demand), fits)
                            compacted[moves_allowed] += (
                                self._layout(allocator) != before
                            )
        # The compacting branch ran, and only where moves are allowed.
        self.assertGreater(compacted[True], 0)
        self.assertEqual(compacted[False], 0)

    @staticmethod
    def _run_random_case(case, *, per_side: bool):
        """Fill a pool with random nodes (some SWA-dead, some locked) and make
        room for a random FULL/SWA demand, either with the per-side stop the
        tree applies without a demand, or through `evict_to_free_tokens`."""
        rng = random.Random(case)
        page_size = rng.choice((1, 4))
        allocator = _full_swa_allocator(page_size=page_size, pages=rng.randint(12, 48))
        TestSharedPoolMakeRoom._set_moves_allowed(allocator, rng.random() < 0.7)
        free_pages = allocator.available_size() // page_size
        node_pages = []
        while sum(node_pages) < free_pages:
            node_pages.append(min(rng.randint(1, 3), free_pages - sum(node_pages)))
        swa_dead = [i for i in range(len(node_pages)) if rng.random() < 0.3]
        locked = [i for i in range(len(node_pages)) if rng.random() < 0.15]
        nodes = _CachedNodes(allocator, node_pages, swa_dead=swa_dead, locked=locked)
        cache = _cache_over(nodes)
        full = rng.randint(0, free_pages) * page_size
        swa = rng.randint(0 if full else 1, free_pages) * page_size
        plan = allocator.reclaim_plan(
            full,
            swa,
            full_evictable_tokens=cache.full_evictable_size(),
            swa_evictable_tokens=cache.swa_evictable_size(),
        )
        if not per_side:
            ok = allocator.evict_to_free_tokens(cache, full, swa_num_tokens=swa)
        elif plan is None:
            ok = None
        else:
            if plan != (0, 0):
                cache.evict_for_alloc(
                    EvictParams(num_tokens=plan[0], swa_num_tokens=plan[1])
                )
            ok = allocator.ensure_capacity(full, swa)
        return ok, plan, nodes, 3 * page_size

    def test_fit_stop_keeps_every_success_of_the_per_side_stop(self):
        fixed = 0
        for case in range(400):
            old_ok, _, _, _ = self._run_random_case(case, per_side=True)
            new_ok, plan, nodes, max_node = self._run_random_case(case, per_side=False)
            with self.subTest(case=case):
                if old_ok:
                    self.assertTrue(new_ok)
                if plan is None:
                    self.assertEqual(sum(nodes.evicted.values()), 0)
                else:
                    self.assertLess(
                        nodes.evicted[ComponentType.FULL], plan[0] + max_node
                    )
                    self.assertLess(
                        nodes.evicted[ComponentType.SWA], plan[1] + max_node
                    )
            fixed += bool(new_ok) and not old_ok
        # The random trees do reach the early stop the fit check repairs.
        self.assertGreater(fixed, 0)


if __name__ == "__main__":
    unittest.main()
