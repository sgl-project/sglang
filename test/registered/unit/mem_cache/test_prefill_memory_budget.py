"""CPU regressions for allocator-owned prefill admission and pending demand."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget
from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _shared_allocator():
    return init_unified_swa_pools(
        device="cpu",
        kv_cache_dtype=torch.float16,
        head_num=1,
        head_dim=4,
        v_head_dim=4,
        swa_head_num=1,
        swa_head_dim=4,
        swa_v_head_dim=4,
        page_size=4,
        start_layer=0,
        end_layer=2,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        total_bytes=1024,
        enable_memory_saver=False,
        need_sort=False,
        lazy_compaction=True,
    ).token_to_kv_pool_allocator


def _cache():
    return SimpleNamespace(
        sliding_window_size=8,
        full_evictable_size=lambda: 0,
        swa_evictable_size=lambda: 0,
        is_chunk_cache=lambda: False,
    )


class TestSharedPrefillMemoryBudget(unittest.TestCase):
    def setUp(self):
        self.allocator = _shared_allocator()
        self.cache = _cache()
        self.budget = self.allocator.create_prefill_budget(self.cache)
        self.request = dict(
            extend_input_len=12,
            total_tokens=20,
            max_new_tokens=4,
            input_tokens=12,
            swa_host_hit_length=0,
            chunk_limit=16,
        )

    def test_pending_batch_cannot_spend_shared_bytes_twice(self):
        self.assertEqual(self.budget.check_prefill(**self.request), (True, 16))
        self.budget.reserve(12, 4, chunk_limit=16)
        # Each side separately has room, but their combined reservation does not.
        self.assertGreater(self.budget.remaining_total, 20)
        self.assertGreater(self.budget.remaining_swa, 16)
        self.assertEqual(self.budget.check_prefill(**self.request), (False, None))
        self.assertEqual(self.allocator.full_attn_allocator.allocated_count(), 0)
        self.assertEqual(self.allocator.swa_attn_allocator.allocated_count(), 0)

    def test_mixed_decode_reserves_both_sides(self):
        budget = self.allocator.create_prefill_budget(
            self.cache, num_mixed_decode_tokens=4
        )
        budget.reserve(12, 4, chunk_limit=16)
        self.assertEqual(
            (budget.total_offset, budget.current_offset, budget.swa_offset),
            (24, 20, 20),
        )
        self.assertEqual(budget.check_prefill(**self.request), (False, None))

    def test_prefix_lock_changes_admission_without_rebuilding_budget(self):
        self.assertIsNotNone(self.allocator.alloc(24))
        self.cache.full_evictable_size = lambda: 24
        self.cache.swa_evictable_size = lambda: 24
        self.assertEqual(self.budget.check_prefill(**self.request), (True, 16))
        # Locking the cached prefix removes its eviction credit.
        self.cache.full_evictable_size = lambda: 0
        self.cache.swa_evictable_size = lambda: 0
        self.assertEqual(self.budget.check_prefill(**self.request), (False, None))

    def test_final_chunk_reserves_decode_headroom(self):
        limit = self.budget.fit_chunk(
            extend_input_len=12, max_new_tokens=80, chunk_limit=16
        )
        self.assertEqual(limit, 8)
        self.budget.reserve(limit, 0, chunk_limit=16, is_chunked_continuation=True)
        self.assertIsNotNone(self.allocator.alloc(limit))

    def test_host_swa_load_is_part_of_joint_demand(self):
        self.assertEqual(self.budget.check_prefill(**self.request), (True, 16))
        request = {**self.request, "swa_host_hit_length": 32}
        self.assertEqual(self.budget.check_prefill(**request), (False, None))

    def test_prompt_clipping_uses_the_empty_pool(self):
        kwargs = dict(token_capacity=1, sliding_window_size=8, chunk_size=16)
        limit = self.allocator.max_new_tokens_for_memory(12, 80, **kwargs)
        self.assertIsNotNone(limit)
        self.assertGreater(limit, 0)
        self.assertIsNotNone(self.allocator.alloc(24))
        self.assertEqual(
            self.allocator.max_new_tokens_for_memory(12, 80, **kwargs), limit
        )
        self.assertIsNone(self.allocator.max_new_tokens_for_memory(100, 0, **kwargs))

    def test_shared_stats_pair_available_tokens_with_current_capacity(self):
        self.assertIsNotNone(self.allocator.alloc(12))
        (full_capacity, full_free), (swa_capacity, swa_free) = (
            self.allocator.swa_capacity_and_available(full_capacity=1, swa_capacity=1)
        )
        self.assertEqual(full_capacity - full_free, 12)
        self.assertEqual(swa_capacity - swa_free, 12)

    def test_common_eviction_dispatches_joint_reclaim(self):
        self.cache.token_to_kv_pool_allocator = self.allocator
        self.allocator.evict_to_free_tokens = MagicMock()
        evict_from_tree_cache(self.cache, 8)
        self.allocator.evict_to_free_tokens.assert_called_once_with(self.cache, 8)


class TestFixedPrefillMemoryBudget(unittest.TestCase):
    def _allocator(self, cls=SWATokenToKVPoolAllocator):
        allocator = object.__new__(cls)
        allocator.page_size = 4
        allocator._size_full = 128
        allocator._size_swa = 64
        allocator.full_available_size = lambda: 128
        allocator.swa_available_size = lambda: 64
        return allocator

    def test_ring_slot_reserved_once_and_evictable_tokens_give_no_credit(self):
        allocator = self._allocator()
        allocator._swa_req_ring = True
        allocator._swa_ring_cost = 32
        cache = _cache()
        cache.swa_evictable_size = lambda: 1000
        budget = allocator.create_prefill_budget(cache)
        budget.reserve(12, 4, chunk_limit=16)
        self.assertEqual(budget.remaining_swa, 32)
        budget.reserve(12, 4, chunk_limit=16, is_chunked_continuation=True)
        self.assertEqual(budget.remaining_swa, 32)
        # The last exact ring slot remains admissible.
        self.assertEqual(
            budget.check_prefill(
                extend_input_len=4,
                total_tokens=12,
                max_new_tokens=4,
                input_tokens=4,
                swa_host_hit_length=0,
                chunk_limit=16,
            ),
            (True, 16),
        )

    def test_pure_swa_budget_reads_swa_capacity(self):
        allocator = self._allocator(PureSWATokenToKVPoolAllocator)
        allocator.full_available_size = lambda: 0
        budget = allocator.create_prefill_budget(_cache())
        self.assertEqual(budget.remaining_total, 64)
        self.assertTrue(budget.has_capacity())

    def test_hisparse_budget_reads_wrapper_capacity(self):
        allocator = self._allocator(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        allocator.full_available_size = lambda: 12
        budget = allocator.create_prefill_budget(_cache())
        self.assertEqual(budget.remaining_total, 12)
        self.assertEqual(budget.remaining_swa, 64)

    def test_tri_pool_keeps_fixed_admission_and_clipping(self):
        allocator = self._allocator(UnifiedMambaSWATokenToKVPoolAllocator)
        allocator.can_reserve = MagicMock(
            side_effect=AssertionError("two-pool reservation")
        )
        budget = allocator.create_prefill_budget(_cache())
        self.assertIs(type(budget), SWAPrefillBudget)
        self.assertTrue(budget.has_capacity())
        self.assertEqual(
            allocator.max_new_tokens_for_memory(
                5,
                100,
                token_capacity=32,
                sliding_window_size=8,
                chunk_size=16,
            ),
            19,
        )

    def test_tri_pool_eviction_does_not_reenter_common(self):
        allocator = self._allocator(UnifiedMambaSWATokenToKVPoolAllocator)
        allocator.available_size = lambda: 0
        allocator.full_available_size = lambda: 0
        allocator.swa_available_size = lambda: 0
        cache = _cache()
        cache.token_to_kv_pool_allocator = allocator
        cache.evict_for_alloc = MagicMock()
        evict_from_tree_cache(cache, 8)
        cache.evict_for_alloc.assert_called_once()
        params = cache.evict_for_alloc.call_args.args[0]
        self.assertEqual((params.num_tokens, params.swa_num_tokens), (8, 8))


if __name__ == "__main__":
    unittest.main()
