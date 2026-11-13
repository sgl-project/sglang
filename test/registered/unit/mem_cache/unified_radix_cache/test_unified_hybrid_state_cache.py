"""The hybrid-model state cache switches on the unified memory pool.

A FULL + MAMBA ``UnifiedRadixCache`` over the pool stack ``--enable-unified-memory``
builds (declarations, plan, sizing and builder, through ``build_unified_bundle``):
each switch runs on the arena's chain allocator, its storage and its state
slots, and with the switch off the cache behaves as before.
"""

from array import array

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


if __name__ == "__main__":
    import unittest

    unittest.main()
