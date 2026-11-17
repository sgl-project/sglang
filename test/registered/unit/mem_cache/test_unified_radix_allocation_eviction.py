"""CPU-only tests for allocation-aware UnifiedRadixCache eviction."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.environ import envs
from sglang.srt.mem_cache import cache_residency
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
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


class _ResidencyCore:
    """A Rust TreeCore stand-in with the cache-residency bindings."""

    def __init__(self):
        self._binding = SimpleNamespace(
            configure_cache_residency=None,
            mark_host_evict_first=None,
            cache_residency_stats=None,
        )
        self.configured = None
        self.marked = []

    def configure_cache_residency(self, **kwargs):
        self.configured = kwargs

    def mark_host_evict_first(self, node_ids):
        self.marked.append(list(node_ids))
        return len(node_ids)

    def cache_residency_stats(self):
        return {"host_evict_first_nodes": len(self.marked)}


def _req(rid, session, parent=None, last_node=None):
    return SimpleNamespace(
        rid=rid, session_id=session, parent_session_id=parent, last_node=last_node
    )


class TestSubagentHostEvictFirst(CustomTestCase):
    """SGLANG_OPT_HICACHE_SUBAGENT_HOST_EVICT_FIRST: a parent session's next request
    ranks its finished sub-agents' last nodes first for host eviction."""

    def _residency(self, *, on=True, write_back=True, core=None):
        with envs.SGLANG_OPT_HICACHE_SUBAGENT_HOST_EVICT_FIRST.override(on):
            return cache_residency.CacheResidency(
                tree_core=core or _ResidencyCore(), is_write_back=write_back
            )

    def test_off_by_default_and_checked_when_on(self):
        self.assertFalse(envs.SGLANG_OPT_HICACHE_SUBAGENT_HOST_EVICT_FIRST.get())
        self.assertIsNone(self._residency(on=False).subagents)
        with self.assertRaisesRegex(ValueError, "write_back"):
            self._residency(write_back=False)
        with self.assertRaisesRegex(RuntimeError, "cache-residency bindings"):
            self._residency(core=SimpleNamespace(_binding=SimpleNamespace()))
        core = _ResidencyCore()
        residency = self._residency(core=core)
        self.assertEqual(
            core.configured,
            {"host_evict_first": True},
        )
        self.assertIsNotNone(residency.subagents)

    def test_finished_children_are_marked_at_the_parents_next_request(self):
        core = _ResidencyCore()
        tracker = self._residency(core=core).subagents
        tracker.on_admit(_req("p1", "parent"))
        tracker.on_admit(_req("c1", "child-a", parent="parent"))
        tracker.on_admit(_req("c2", "child-b", parent="parent"))
        tracker.on_release(_req("c1", "child-a", parent="parent", last_node=11))
        tracker.on_release(_req("p1", "parent", last_node=3))
        # child-b is still in flight: only child-a is finished
        tracker.on_admit(_req("p2", "parent"))
        self.assertEqual(core.marked, [[11]])
        tracker.on_release(_req("c2", "child-b", parent="parent", last_node=12))
        tracker.on_admit(_req("p3", "parent"))
        self.assertEqual(core.marked, [[11], [12]])
        # a request outside any session, and a parent without sub-agents, mark nothing
        tracker.on_admit(_req("x", None))
        tracker.on_admit(_req("p4", "other"))
        self.assertEqual(len(core.marked), 2)

    def test_radix_cache_hooks_reach_the_tracker(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache._residency = SimpleNamespace(subagents=MagicMock())
        cache.session = SimpleNamespace(try_match_prefix=lambda params: "matched")
        cache._components_tuple = ()
        req = _req("r", "s")
        self.assertEqual(
            cache.match_prefix(SimpleNamespace(req=req)),
            "matched",
        )
        cache.on_release(req, checkpointed=True)
        cache._residency.subagents.on_admit.assert_called_once_with(req)
        cache._residency.subagents.on_release.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
