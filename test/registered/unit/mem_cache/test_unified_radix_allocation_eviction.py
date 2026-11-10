"""CPU-only tests for allocation-aware UnifiedRadixCache eviction."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.kv.plan import TREE_TOKEN
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentDeviceSlot
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    FullComponent,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    DemoteResult,
)
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
        cache.evicts_by_joint_lru = False
        cache.cache_controller = None
        cache.metrics_collector = None
        cache.tree_core = MagicMock()
        cache.tree_core.full_evictable_size.return_value = 80

        capacity = {"available": 30}
        allocator = MagicMock()
        allocator.available_size.side_effect = lambda: capacity["available"]
        allocator.owner_available_size.side_effect = lambda: capacity["available"]
        # Pools that share no memory: the tree walks each target itself.
        allocator.pools_share_memory = False
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

        # 30 available: a 100-token demand is 70 short.
        result = cache.evict_for_alloc(KVDemand(full_tokens=100))

        self.assertEqual(capacity["available"], 100)
        self.assertEqual(leaf_count["value"], 1)
        self.assertEqual(result.num_tokens_evicted, 20)
        self.assertEqual(result.mamba_num_evicted, 1)

    def test_allocation_eviction_runs_to_the_target_when_frees_stay_hidden(self):
        """An allocator can hide freed pages behind a clamp or page rounding, so
        freeing exactly the shortfall may leave the demand unmet; an
        allocation-driven FULL walk must keep going until the demand fits."""
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

        # 30 available: a 50-token demand is 20 short.
        result = cache.evict_for_alloc(KVDemand(full_tokens=50))

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
        cache.token_to_kv_pool_allocator.pools_share_memory = False

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

        result = cache.evict_for_alloc(KVDemand(state_slots=3))

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

        tree_cache.evict_for_alloc.assert_called_once_with(KVDemand(full_tokens=100))
        tree_cache.evict.assert_not_called()

    def test_default_cache_evicts_the_shortfall_of_a_demand(self):
        from types import SimpleNamespace

        from sglang.srt.mem_cache.base_prefix_cache import (
            BasePrefixCache,
            EvictResult,
        )

        for supports_swa, demand, expected in [
            (False, KVDemand(full_tokens=100), EvictParams(num_tokens=70)),
            (False, KVDemand(full_tokens=30), None),
            (
                True,
                KVDemand(full_tokens=100, window_tokens={"swa": 100}),
                EvictParams(num_tokens=60, window_num_tokens={"swa": 90}),
            ),
            (True, KVDemand(full_tokens=40, window_tokens={"swa": 10}), None),
        ]:
            with self.subTest(supports_swa=supports_swa, demand=demand):
                # Pools of their own; a window pool beside the token pool
                # leaves the token pool its full side (40 of 30 + 10).
                allocator = SimpleNamespace(
                    pools_share_memory=False,
                    owner_available_size=lambda: 40 if supports_swa else 30,
                    window_available_size=lambda name: 10,
                )
                cache = SimpleNamespace(
                    token_to_kv_pool_allocator=allocator,
                    supports_swa=lambda: supports_swa,
                    window_sub_pools=lambda: (("swa", 4),) if supports_swa else (),
                    window_available_size=lambda name: allocator.window_available_size(
                        name
                    ),
                    evict=MagicMock(return_value=EvictResult(num_tokens_evicted=1)),
                )
                cache.evict_pool_shortfalls = lambda d, cache=cache: (
                    BasePrefixCache.evict_pool_shortfalls(cache, d)
                )
                BasePrefixCache.evict_for_alloc(cache, demand)
                if expected is None:
                    cache.evict.assert_not_called()
                else:
                    cache.evict.assert_called_once_with(expected)


PAGE_SIZE = 4


def _grouped_frees(enabled: bool) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True))
    stack.enter_context(envs.SGLANG_OPT_HICACHE_GROUPED_DEVICE_FREES.override(enabled))
    return stack


class TestUnifiedRadixGroupedDeviceFrees(CustomTestCase):
    @staticmethod
    def _build_cache(need_sort: bool = False):
        allocator = PagedTokenToKVPoolAllocator(
            size=64 * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=need_sort,
        )
        storage = SimpleNamespace(
            _band=lambda sub_pool: allocator,
            free_units=lambda sub_pool, ids: allocator.free_segment(ids, start_pos=0),
        )
        full = object.__new__(FullComponent)
        full.storage_and_sub_pool = lambda: (storage, "full")
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_components = (ComponentType.FULL,)
        index_slots(cache, cache.tree_components)
        cache.token_to_kv_pool_allocator = allocator
        cache.components = {cache.slot_id(TREE_TOKEN): full}
        return cache, allocator

    @staticmethod
    def _segments(allocator, pages_each=(1, 3, 2)):
        return [allocator.alloc(n * PAGE_SIZE) for n in pages_each]

    def test_one_free_leaves_the_per_segment_free_list(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                cache, grouped_alloc = self._build_cache(need_sort)
                _, stock_alloc = self._build_cache(need_sort)
                ct = cache.slot_id(TREE_TOKEN)
                segs = self._segments(grouped_alloc)
                self.assertTrue(
                    all(
                        torch.equal(a, b)
                        for a, b in zip(segs, self._segments(stock_alloc))
                    )
                )
                for seg in segs:
                    stock_alloc.free_segment(seg, start_pos=0)
                with _grouped_frees(True):
                    values = cache._group_page_exact_frees(ct, list(segs))
                self.assertEqual(len(values), 1)
                grouped_alloc.free_segment(values[0], start_pos=0)
                if not need_sort:
                    # Staged pages are sorted before use, so only this order matters.
                    self.assertTrue(
                        torch.equal(grouped_alloc.free_pages, stock_alloc.free_pages)
                    )
                grouped_alloc.merge_and_sort_free()
                stock_alloc.merge_and_sort_free()
                self.assertTrue(
                    torch.equal(grouped_alloc.free_pages, stock_alloc.free_pages)
                )

    def test_segments_stay_apart_unless_page_exact_on_the_stock_allocator(self):
        cache, allocator = self._build_cache()
        ct = cache.slot_id(TREE_TOKEN)
        segs = self._segments(allocator)
        partial = [segs[0], segs[1][: PAGE_SIZE + 1]]
        self.assertIs(cache._group_page_exact_frees(ct, partial), partial)
        allocator.free_group_begin()
        self.assertIs(cache._group_page_exact_frees(ct, segs), segs)
        allocator.free_group_end()
        cache.components[ct] = MagicMock()
        self.assertIs(cache._group_page_exact_frees(ct, segs), segs)

    def test_write_back_batch_frees_its_held_segments_once(self):
        for enabled, expected in ((False, [[0], [1], [2]]), (True, [[1, 0], [2]])):
            with self.subTest(enabled=enabled):
                cache, allocator = self._build_cache()
                ct = cache.slot_id(TREE_TOKEN)
                segs = self._segments(allocator)
                # The second demote carries a cache action, so the first one's
                # held segment is freed with it, ahead of its own.
                results = [
                    DemoteResult(
                        device_frees={ct: [seg]},
                        cache_actions=[MagicMock()] if i == 1 else [],
                    )
                    for i, seg in enumerate(segs)
                ]
                cache.tree_core = MagicMock()
                cache.tree_core.demote.side_effect = results
                cache.writing_check = MagicMock()
                cache._drain_host_frees = MagicMock()
                cache._apply_cache_actions = MagicMock()
                freed = []

                def record(action):
                    self.assertIsInstance(action, FreeComponentDeviceSlot)
                    freed.append(list(action.indices))

                cache._apply_cache_action = record
                node_ids = [11, 12, 13]
                with _grouped_frees(enabled):
                    cache._demote_backed_up(node_ids, {ct: 0})
                self.assertEqual(node_ids, [])
                self.assertEqual(len(freed), len(expected))
                for got, want in zip(freed, expected):
                    self.assertEqual(len(got), 1 if enabled else len(want))
                    want_cat = torch.cat([segs[i] for i in want])
                    self.assertTrue(torch.equal(torch.cat(got), want_cat))


if __name__ == "__main__":
    unittest.main()
