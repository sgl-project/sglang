"""SGLANG_HICACHE_WRITE_BACK_MIN_EVICT_TOKENS: a floor on write_back
allocation-driven eviction.

Drives a real UnifiedRadixCache (both tree cores) on CPU with the write-back
fixtures: a raised call frees the floor behind one blocking wait, and nothing
else (a shortfall at the floor, explicit eviction, unified memory, a cache
outside write_back, window shortfalls) changes.
"""

import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.hicache_write_back_fixtures import (
    BATCH_TOKENS,
    PAGE,
    SEQ_TOKENS,
    batch_seqs,
    hybrid_write_back_cache,
    insert_seq,
    write_back_cache,
)
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# 32 two-page leaves fill the device.
LEAVES = 4 * BATCH_TOKENS // SEQ_TOKENS
FLOOR = 8 * SEQ_TOKENS


class _MinEvictTokensSuite:
    tree_core_backend: str

    def _full_device(self):
        """A Full-only write_back cache whose device holds 32 leaves and
        nothing else free."""
        cache, allocator, _ = write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            host_tokens=4 * BATCH_TOKENS,
        )
        for batch in range(4):
            for seq in batch_seqs(batch):
                insert_seq(cache, allocator, seq)
        while allocator.alloc(PAGE) is not None:
            pass
        self.assertEqual(cache.full_evictable_size(), LEAVES * SEQ_TOKENS)
        return cache, allocator

    def _count_waits(self, cache):
        """Record how many queued copies each blocking write_back wait drains."""
        waits = []
        writing_check = cache.writing_check

        def count_waits(write_back=False, finish_count=None):
            if write_back and cache.ongoing_write_through:
                waits.append(len(cache.ongoing_write_through))
            return writing_check(write_back=write_back, finish_count=finish_count)

        return waits, mock.patch.object(cache, "writing_check", side_effect=count_waits)

    def _evict_for_alloc(self, cache, allocator, floor, full_tokens):
        before = allocator.available_size()
        waits, patch = self._count_waits(cache)
        with envs.SGLANG_HICACHE_WRITE_BACK_MIN_EVICT_TOKENS.override(floor), patch:
            result = cache.evict_for_alloc(KVDemand(full_tokens=full_tokens))
        cache.sanity_check()
        return result, waits, allocator.available_size() - before

    def _evict_args(self, cache, floor, demand):
        """The EvictParams and demand evict_pool_shortfalls hands to _evict."""
        with (
            envs.SGLANG_HICACHE_WRITE_BACK_MIN_EVICT_TOKENS.override(floor),
            mock.patch.object(cache, "_evict", return_value=EvictResult()) as evict,
        ):
            cache.evict_pool_shortfalls(demand)
        evict.assert_called_once()
        return evict.call_args.args[0], evict.call_args.kwargs["demand"]

    def test_a_raised_call_frees_the_floor_behind_one_wait(self):
        cache, allocator = self._full_device()

        result, waits, freed = self._evict_for_alloc(
            cache, allocator, FLOOR, SEQ_TOKENS
        )

        # Eight leaves, not the one the shortfall needs, with their eight
        # copies drained by one wait.
        self.assertEqual(result.num_tokens_evicted, FLOOR)
        self.assertEqual(waits, [FLOOR // SEQ_TOKENS])
        self.assertGreaterEqual(freed, FLOOR)

    def test_the_floor_is_bounded_by_the_evictable_tokens(self):
        cache, allocator = self._full_device()

        result, waits, freed = self._evict_for_alloc(
            cache, allocator, 4 * LEAVES * SEQ_TOKENS, SEQ_TOKENS
        )

        self.assertEqual(result.num_tokens_evicted, LEAVES * SEQ_TOKENS)
        self.assertEqual(waits, [LEAVES])
        self.assertEqual(freed, LEAVES * SEQ_TOKENS)
        self.assertEqual(cache.full_evictable_size(), 0)

    def test_off_by_default_frees_the_shortfall(self):
        cache, allocator = self._full_device()

        result, waits, freed = self._evict_for_alloc(cache, allocator, 0, SEQ_TOKENS)

        self.assertEqual(result.num_tokens_evicted, SEQ_TOKENS)
        self.assertEqual(waits, [1])
        self.assertEqual(freed, SEQ_TOKENS)

    def test_a_shortfall_at_the_floor_is_not_raised(self):
        cache, allocator = self._full_device()

        result, waits, _ = self._evict_for_alloc(cache, allocator, FLOOR, FLOOR)

        self.assertEqual(result.num_tokens_evicted, FLOOR)
        self.assertEqual(waits, [FLOOR // SEQ_TOKENS])

    def test_explicit_evict_is_not_raised(self):
        cache, _ = self._full_device()

        with envs.SGLANG_HICACHE_WRITE_BACK_MIN_EVICT_TOKENS.override(FLOOR):
            result = cache.evict(EvictParams(num_tokens=SEQ_TOKENS))

        self.assertEqual(result.num_tokens_evicted, SEQ_TOKENS)
        cache.sanity_check()

    def test_unified_memory_is_not_raised(self):
        """Sub-pools sharing one memory reclaim jointly, outside the floor."""
        setup_allocator_context()
        bundle = build_unified_bundle(
            UnifiedGeometry(
                chain="full",
                full_layer_ids=(0,),
                page_size=PAGE,
                full_tokens=2 * BATCH_TOKENS,
                context_len=2 * BATCH_TOKENS,
            )
        )
        cache, allocator, _ = write_back_cache(
            self.tree_core_backend,
            device_tokens=2 * BATCH_TOKENS,
            host_tokens=BATCH_TOKENS,
            allocator=bundle.token_to_kv_pool_allocator,
        )
        for seq in batch_seqs(1):
            insert_seq(cache, allocator, seq)
        while allocator.alloc(PAGE) is not None:
            pass

        with envs.SGLANG_HICACHE_WRITE_BACK_MIN_EVICT_TOKENS.override(FLOOR):
            result = cache.evict_for_alloc(KVDemand(full_tokens=SEQ_TOKENS))

        self.assertEqual(result.num_tokens_evicted, SEQ_TOKENS)
        cache.sanity_check()

    def test_outside_write_back_is_not_raised(self):
        cache, _ = self._full_device()
        cache.is_write_back = False

        params, demand = self._evict_args(cache, FLOOR, KVDemand(full_tokens=PAGE))

        self.assertEqual(params.num_tokens, PAGE)
        self.assertEqual(demand, {ComponentType.FULL: PAGE})

    def test_a_disabled_floor_does_not_read_the_write_policy(self):
        cache, _ = self._full_device()
        with mock.patch.object(
            type(cache), "is_write_back", new_callable=mock.PropertyMock
        ) as is_write_back:
            params, _ = self._evict_args(cache, 0, KVDemand(full_tokens=PAGE))
        is_write_back.assert_not_called()
        self.assertEqual(params.num_tokens, PAGE)

    def test_window_shortfalls_pass_through(self):
        cache, _, _, _ = hybrid_write_back_cache(
            self.tree_core_backend,
            device_tokens=4 * BATCH_TOKENS,
            swa_device_tokens=4 * BATCH_TOKENS,
            host_tokens=4 * BATCH_TOKENS,
            swa_host_tokens=4 * BATCH_TOKENS,
            sliding_window=3 * PAGE,
        )
        ((window_ct, window),) = cache._window_sub_pools_by_slot().items()
        with mock.patch.object(cache, "_pool_available", return_value=0):
            # A window-only shortfall: nothing raised.
            params, demand = self._evict_args(cache, FLOOR, KVDemand(0, {window: PAGE}))
            self.assertEqual(
                (params.num_tokens, dict(params.window_num_tokens)), (0, {window: PAGE})
            )
            self.assertEqual(demand, {window_ct: PAGE})

            # With a token shortfall, only the token entry rises.
            params, demand = self._evict_args(
                cache, FLOOR, KVDemand(PAGE, {window: PAGE})
            )
            self.assertEqual(
                (params.num_tokens, dict(params.window_num_tokens)),
                (FLOOR, {window: PAGE}),
            )
            self.assertEqual(demand, {ComponentType.FULL: FLOOR, window_ct: PAGE})


class TestMinEvictTokensRustCore(_MinEvictTokensSuite, CustomTestCase):
    tree_core_backend = "rust"


class TestMinEvictTokensPythonCore(_MinEvictTokensSuite, CustomTestCase):
    tree_core_backend = "python"


if __name__ == "__main__":
    unittest.main()
