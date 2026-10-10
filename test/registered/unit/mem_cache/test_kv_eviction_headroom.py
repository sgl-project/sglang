"""CPU tests for bounded L1 KV-cache eviction headroom."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.environ import envs
from sglang.srt.mem_cache import common
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestKVEvictionHeadroom(CustomTestCase):
    def test_target_adds_bounded_headroom_only_under_pressure(self):
        with envs.SGLANG_OPT_KV_CACHE_EVICTION_HEADROOM_TOKENS.override(128):
            self.assertEqual(common._eviction_target_with_headroom(64), 192)
            self.assertEqual(common._eviction_target_with_headroom(64, 100), 100)
            self.assertEqual(common._eviction_target_with_headroom(64, 32), 64)
            self.assertEqual(common._eviction_target_with_headroom(0, 100), 0)

    def test_zero_headroom_preserves_exact_shortfall(self):
        for headroom in (0, -1):
            with envs.SGLANG_OPT_KV_CACHE_EVICTION_HEADROOM_TOKENS.override(headroom):
                self.assertEqual(common._eviction_target_with_headroom(64, 1000), 64)

    def test_standard_allocator_adds_capped_headroom(self):
        allocator = object.__new__(TokenToKVPoolAllocator)
        allocator.available_size = MagicMock(return_value=100)
        allocator.page_size = 1
        tree_cache = MagicMock()
        tree_cache.supports_prefix_sharing.return_value = True
        tree_cache.evictable_size.return_value = 150
        tree_cache.token_to_kv_pool_allocator = allocator

        with envs.SGLANG_OPT_KV_CACHE_EVICTION_HEADROOM_TOKENS.override(128):
            evict_from_tree_cache(tree_cache, num_tokens=200)

        tree_cache.evict_for_alloc.assert_called_once_with(EvictParams(num_tokens=150))

    def test_no_pressure_does_not_evict(self):
        allocator = object.__new__(TokenToKVPoolAllocator)
        allocator.available_size = MagicMock(return_value=256)
        allocator.page_size = 1
        tree_cache = MagicMock()
        tree_cache.supports_prefix_sharing.return_value = True
        tree_cache.token_to_kv_pool_allocator = allocator

        with envs.SGLANG_OPT_KV_CACHE_EVICTION_HEADROOM_TOKENS.override(128):
            evict_from_tree_cache(tree_cache, num_tokens=200)

        tree_cache.evict_for_alloc.assert_not_called()

    def test_swa_adds_headroom_only_to_the_short_pool(self):
        allocator = SimpleNamespace(
            full_available_size=MagicMock(return_value=100),
            swa_available_size=MagicMock(return_value=500),
        )
        tree_cache = MagicMock()
        tree_cache.supports_prefix_sharing.return_value = True
        tree_cache.full_evictable_size.return_value = 1000

        with envs.SGLANG_OPT_KV_CACHE_EVICTION_HEADROOM_TOKENS.override(128):
            SWATokenToKVPoolAllocator.evict_to_free_tokens(
                allocator, tree_cache, num_tokens=200
            )

        tree_cache.evict_for_alloc.assert_called_once_with(
            EvictParams(num_tokens=228, swa_num_tokens=0)
        )
        tree_cache.swa_evictable_size.assert_not_called()

    def test_unified_swa_adds_headroom_after_planning(self):
        allocator = SimpleNamespace(
            reclaim_plan=MagicMock(return_value=(100, 0)),
            ensure_capacity=MagicMock(return_value=True),
        )
        tree_cache = MagicMock()
        tree_cache.supports_prefix_sharing.return_value = True
        tree_cache.full_evictable_size.return_value = 1000
        tree_cache.swa_evictable_size.return_value = 1000

        with envs.SGLANG_OPT_KV_CACHE_EVICTION_HEADROOM_TOKENS.override(128):
            ready = UnifiedSWATokenToKVPoolAllocator.evict_to_free_tokens(
                allocator, tree_cache, num_tokens=200
            )

        self.assertTrue(ready)
        tree_cache.evict.assert_called_once_with(
            EvictParams(num_tokens=228, swa_num_tokens=0)
        )
        allocator.ensure_capacity.assert_called_once_with(200, 200)


if __name__ == "__main__":
    unittest.main()
