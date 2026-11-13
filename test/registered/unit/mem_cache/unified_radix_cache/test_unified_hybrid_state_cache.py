"""The hybrid-model state cache switches on the unified memory pool.

A FULL + MAMBA ``UnifiedRadixCache`` over the pool stack ``--enable-unified-memory``
builds (declarations, plan, sizing and builder, through ``build_unified_bundle``):
each switch runs on the arena's chain allocator, its storage and its state
slots, and with the switch off the cache behaves as before.
"""

from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocation import alloc_token_slots
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
GDS = {"prefill_cost": [1e-5, 0.0]}


class _UnifiedCache:
    """A unified radix cache over a full + Mamba unified pool on CPU."""

    def __init__(self, *, page_size=1, full_tokens=96, state_slots=16, **switches):
        setup_allocator_context()
        self.bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="mamba",
                full_layer_ids=(0,),
                state_layer_ids=(1,),
                page_size=page_size,
                full_tokens=full_tokens,
                state_slots=state_slots,
                context_len=256,
                max_num_reqs=8,
            )
        )
        self.allocator = self.bundle.token_to_kv_pool_allocator
        self.storage = kv_storage_of(self.bundle.req_to_token_pool, self.allocator)
        self.cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.bundle.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=page_size,
                tree_slots=tree_slots_of((FULL, MAMBA)),
                **switches,
            )
        )

    def insert(self, tokens, *, state=True):
        params = InsertParams(
            key=RadixKey(array("q", tokens)),
            value=self.allocator.alloc(len(tokens)),
        )
        if state:
            params.mamba_value = self.storage.take_state_slots(None, 1)
        else:
            params.allow_missing_mamba = True
        self.cache.insert(params)

    def match(self, tokens):
        return self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )

    def free_kv(self):
        return self.allocator.available_size()


def _tokens(start, n):
    return list(range(start, start + n))


class TestUnitEvictionOnUnifiedPool(CustomTestCase):
    PREFIX, TAIL = _tokens(100, 8), _tokens(200, 8)

    def _evict_interior_state(self, unit_eviction):
        """An interior state over a stateless tail goes: the KV tokens the chain
        can hand out more afterwards are the state slot's bytes, plus the
        prefix's and the tail's KV when the unit goes whole."""
        tree = _UnifiedCache(enable_mamba_unit_eviction=unit_eviction)
        tree.insert(self.PREFIX)
        tree.insert(self.PREFIX + self.TAIL, state=False)
        before = tree.free_kv()
        tree.cache.evict(EvictParams(mamba_num=1))
        tree.cache.sanity_check()
        return tree, tree.free_kv() - before

    def test_a_state_takes_the_kv_only_it_serves_back_to_the_chain(self):
        tree, freed = self._evict_interior_state(unit_eviction=True)
        _, freed_state_only = self._evict_interior_state(unit_eviction=False)
        self.assertGreater(freed_state_only, 0)
        self.assertEqual(freed - freed_state_only, 16)
        self.assertEqual(tree.match(self.PREFIX + self.TAIL).full_kv_hit_length, 0)

    def test_without_the_switch_the_kv_stays_cached(self):
        tree, _ = self._evict_interior_state(unit_eviction=False)
        self.assertEqual(tree.match(self.PREFIX + self.TAIL).full_kv_hit_length, 16)


class TestGreedyDualSizeOnUnifiedPool(CustomTestCase):
    def test_the_unit_whose_hits_save_least_goes_first(self):
        tree = _UnifiedCache(eviction_policy="gds", eviction_policy_config=GDS)
        short, long = _tokens(100, 4), _tokens(300, 40)
        tree.insert(long)
        tree.insert(short)
        tree.cache.evict(EvictParams(num_tokens=1))
        tree.cache.sanity_check()
        self.assertEqual(tree.match(short).device_prefix_len, 0)
        self.assertEqual(tree.match(long).device_prefix_len, 40)

    def test_allocation_pressure_evicts_by_score_until_the_demand_fits(self):
        tree = _UnifiedCache(eviction_policy="gds", eviction_policy_config=GDS)
        short, long = _tokens(100, 8), _tokens(300, 40)
        tree.insert(long)
        tree.insert(short)
        need = tree.free_kv() + 4
        slots = alloc_token_slots(tree.cache, need)
        self.assertIsNotNone(slots)
        self.assertEqual(len(slots), need)
        self.assertEqual(tree.match(short).device_prefix_len, 0)
        self.assertEqual(tree.match(long).device_prefix_len, 40)


class TestMissAttributionOnUnifiedPool(CustomTestCase):
    def test_a_state_evicted_under_cached_kv_is_a_dependency_miss(self):
        tree = _UnifiedCache(
            enable_cache_miss_attribution=True, enable_mamba_unit_eviction=False
        )
        prompt = _tokens(100, 16)
        tree.insert(prompt[:12])
        # A stateless branch below keeps the state's KV cached once it goes.
        tree.insert(prompt[:12] + _tokens(900, 4), state=False)
        tree.cache.miss_attribution.record_computed(prompt[:12], 12)
        tree.cache.evict(EvictParams(mamba_num=1))
        match = tree.match(prompt)
        split = tree.cache.miss_attribution.attribute(
            prompt,
            prompt_len=len(prompt),
            hit_len=match.device_prefix_len,
            kv_len=match.full_kv_hit_length,
        )
        self.assertEqual(match.device_prefix_len, 0)
        self.assertEqual(
            split, {"new": 4, "placement": 0, "dependency": 12, "capacity": 0}
        )

    def test_without_the_switch_nothing_is_recorded(self):
        tree = _UnifiedCache()
        self.assertIsNone(tree.cache.miss_attribution)


class TestExactCheckpointPiecesOnUnifiedPool(CustomTestCase):
    PAGE = 4

    def test_the_cache_builds_with_the_switch_on_a_unified_pool(self):
        tree = _UnifiedCache(
            page_size=self.PAGE,
            enable_mamba_extra_buffer=True,
            enable_mamba_exact_checkpoint=True,
        )
        self.assertTrue(tree.cache.exact_checkpoint)

    def test_a_resume_page_copy_moves_the_rows_through_the_arena(self):
        tree = _UnifiedCache(page_size=self.PAGE, enable_mamba_extra_buffer=True)
        src = tree.allocator.alloc(self.PAGE)
        dst = tree.allocator.alloc(self.PAGE)
        kv = tree.allocator.get_kvcache()
        k_buffer = kv.get_key_buffer(0)
        # The allocator hands out virtual ids; the rows sit at the physical ones.
        src_rows = tree.allocator.translate_kv_loc(src).long()
        dst_rows = tree.allocator.translate_kv_loc(dst).long()
        rows = torch.arange(1, 1 + src.numel() * k_buffer[0].numel())
        k_buffer[src_rows] = rows.view(src.numel(), *k_buffer.shape[1:]).to(
            k_buffer.dtype
        )

        def run_copy(is_draft_worker):
            runner = SimpleNamespace(
                req_to_token_pool=tree.bundle.req_to_token_pool,
                token_to_kv_pool_allocator=tree.allocator,
                token_to_kv_pool=kv,
                is_draft_worker=is_draft_worker,
            )
            ModelRunner._maybe_copy_deferred_kv_pages(
                runner, SimpleNamespace(kv_copy_src_locs=src, kv_copy_dst_locs=dst)
            )

        run_copy(is_draft_worker=False)
        self.assertTrue(torch.equal(k_buffer[dst_rows], k_buffer[src_rows]))
        self.assertFalse(torch.equal(dst_rows, src_rows))
        # The target's copy moved a fused draft's rows too, and the forward
        # has written the page since: the draft runner's copy does nothing.
        k_buffer[dst_rows] = 0
        run_copy(is_draft_worker=True)
        self.assertFalse(k_buffer[dst_rows].any())

    def test_an_exact_leaf_inside_a_page_holds_the_page_and_the_state(self):
        tree = _UnifiedCache(
            page_size=self.PAGE,
            enable_mamba_extra_buffer=True,
            enable_mamba_exact_checkpoint=True,
        )
        prefix = _tokens(100, 2 * self.PAGE)
        tree.insert(prefix)
        parent = tree.match(prefix).last_device_node
        page = tree.allocator.alloc(self.PAGE)
        result = tree.cache._insert_exact_leaf(
            parent,
            RadixKey(array("q", _tokens(500, 2))),
            page[:1],
            tree.storage.take_state_slots(None, 1),
            0,
            array("q", prefix + _tokens(500, 2)),
        )
        tree.cache.sanity_check()
        self.assertFalse(result.mamba_exist)
        match = tree.match(prefix + _tokens(500, 2))
        self.assertEqual(match.device_prefix_len, 2 * self.PAGE + 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
