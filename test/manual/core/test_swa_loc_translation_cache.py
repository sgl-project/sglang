"""Manual tests for SWAKVPool.translate_loc_from_full_to_swa cache behaviour.

These tests cover three properties introduced by PR #25824:

  1. Cache key uses data_ptr() — correctly distinguishes views at different
     offsets within the same storage (untyped_storage().data_ptr() would not).

  2. Allocator mutations invalidate the cache — alloc/free/clear/
     set_full_to_swa_mapping each call invalidate_loc_cache() so the next
     translation sees the fresh mapping.

  3. BaseSWAKVPool.invalidate_loc_cache is a no-op default — subclasses that
     don't cache (e.g. DSV4) can be called safely without AttributeError.

Run with:
    python -m pytest test/manual/core/test_swa_loc_translation_cache.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase


def _build_pool(
    kv_size: int = 32,
    kv_size_swa: int = 32,
    page_size: int = 1,
):
    device = get_device()
    num_layers = 8
    full_layer_ids = [0, 4]
    swa_layer_ids = [i for i in range(num_layers) if i not in set(full_layer_ids)]

    pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=4,
        head_dim=64,
        swa_attention_layer_ids=swa_layer_ids,
        full_attention_layer_ids=full_layer_ids,
        enable_kvcache_transpose=False,
        device=device,
    )
    allocator = SWATokenToKVPoolAllocator(
        size=kv_size,
        size_swa=kv_size_swa,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=device,
        kvcache=pool,
        need_sort=False,
    )
    return pool, allocator, device


class TestCacheKeyDataPtr(CustomTestCase):
    """Cache key uses data_ptr(), which encodes the storage offset."""

    def test_same_offset_view_is_cache_hit(self):
        """Two different Python objects pointing to the same base are a hit."""
        pool, allocator, device = _build_pool()
        loc = allocator.alloc(4)
        self.assertIsNotNone(loc)

        # Create two slice objects at offset 0 — same data_ptr, same numel.
        view_a = loc[:4]
        view_b = loc[:4]
        self.assertIsNot(view_a, view_b)  # different Python objects
        self.assertEqual(view_a.data_ptr(), view_b.data_ptr())

        result_a = pool.translate_loc_from_full_to_swa(view_a)
        result_b = pool.translate_loc_from_full_to_swa(view_b)
        # Both should return the identical tensor (cache hit).
        self.assertIs(result_a, result_b)

    def test_different_offset_view_is_cache_miss(self):
        """Views at different offsets produce different data_ptr → cache miss."""
        pool, allocator, device = _build_pool(kv_size=32, kv_size_swa=32)
        loc = allocator.alloc(10)
        self.assertIsNotNone(loc)
        self.assertGreaterEqual(loc.numel(), 10)

        view_lo = loc[0:5]
        view_hi = loc[5:10]
        self.assertEqual(view_lo.numel(), view_hi.numel())  # same numel
        # Different data_ptr (different storage offset).
        self.assertNotEqual(view_lo.data_ptr(), view_hi.data_ptr())

        # Prime the cache with view_lo.
        result_lo = pool.translate_loc_from_full_to_swa(view_lo)
        # view_hi should be a cache miss and produce a distinct translation.
        result_hi = pool.translate_loc_from_full_to_swa(view_hi)
        # They should NOT be the same object (different cache entries).
        self.assertIsNot(result_lo, result_hi)
        # And the content must differ (different full indices → different swa).
        self.assertFalse(torch.equal(result_lo, result_hi))

    def test_storage_base_ptr_would_collide(self):
        """Demonstrate that untyped_storage().data_ptr() WOULD collide for the
        two views above — confirming data_ptr() is the right key."""
        t = torch.arange(20, device=get_device())
        a, b = t[0:10], t[5:15]
        # Same storage base — old key would collide.
        self.assertEqual(a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr())
        self.assertEqual(a.numel(), b.numel())
        # But data_ptr differs — new key is safe.
        self.assertNotEqual(a.data_ptr(), b.data_ptr())


class TestAllocatorMutationInvalidation(CustomTestCase):
    """Each allocator method that writes the mapping calls invalidate_loc_cache."""

    def _prime_and_check_invalidation(self, pool, allocator, mutate_fn):
        """Helper: prime cache, mutate, assert fresh translation."""
        loc = allocator.alloc(4)
        self.assertIsNotNone(loc)
        # Prime the cache.
        first = pool.translate_loc_from_full_to_swa(loc)
        self.assertIsNotNone(pool._cached_loc_key)

        # Mutate — should invalidate.
        mutate_fn(allocator, loc)

        # Cache must be cleared after mutation.
        self.assertIsNone(pool._cached_loc_key)
        self.assertIsNone(pool._cached_swa_loc)

    def test_alloc_invalidates(self):
        pool, allocator, _ = _build_pool()
        loc = allocator.alloc(4)
        pool.translate_loc_from_full_to_swa(loc)
        self.assertIsNotNone(pool._cached_loc_key)
        # Another alloc should invalidate.
        allocator.alloc(4)
        self.assertIsNone(pool._cached_loc_key)

    def test_free_swa_invalidates(self):
        pool, allocator, _ = _build_pool()
        loc = allocator.alloc(4)
        pool.translate_loc_from_full_to_swa(loc)
        self.assertIsNotNone(pool._cached_loc_key)
        allocator.free_swa(loc)
        self.assertIsNone(pool._cached_loc_key)

    def test_clear_invalidates(self):
        pool, allocator, _ = _build_pool()
        loc = allocator.alloc(4)
        pool.translate_loc_from_full_to_swa(loc)
        self.assertIsNotNone(pool._cached_loc_key)
        allocator.clear()
        self.assertIsNone(pool._cached_loc_key)

    def test_set_full_to_swa_mapping_invalidates(self):
        """HiCache load-back path: set_full_to_swa_mapping must invalidate."""
        pool, allocator, device = _build_pool(kv_size=32, kv_size_swa=32)
        loc = allocator.alloc(4)
        pool.translate_loc_from_full_to_swa(loc)
        self.assertIsNotNone(pool._cached_loc_key)

        # Simulate HiCache rebuild with new swa indices.
        new_swa = torch.arange(4, dtype=torch.int64, device=device)
        allocator.set_full_to_swa_mapping(loc, new_swa)

        self.assertIsNone(pool._cached_loc_key)
        # Translation after rebuild should reflect the new mapping.
        result = pool.translate_loc_from_full_to_swa(loc)
        self.assertEqual(result.tolist(), new_swa.tolist())


class TestBaseClassNoOp(CustomTestCase):
    """BaseSWAKVPool.invalidate_loc_cache is a no-op default — must not raise."""

    def test_noop_does_not_raise(self):
        # BaseSWAKVPool is abstract; instantiate via SWAKVPool which inherits.
        pool, _, _ = _build_pool()
        # Calling on the concrete class uses the override — that's fine.
        pool.invalidate_loc_cache()  # must not raise
        pool.invalidate_loc_cache()  # idempotent

    def test_base_class_noop_directly(self):
        """Call the base-class method directly to verify it's a true no-op."""
        pool, _, _ = _build_pool()
        # Prime the cache first.
        loc = pool.full_to_swa_index_mapping  # any tensor
        pool._cached_loc_key = ("dummy", 1)
        pool._cached_swa_loc = torch.zeros(1)
        # Call the BASE class method directly — should not clear the cache
        # (it's a no-op; the concrete override is what clears).
        BaseSWAKVPool.invalidate_loc_cache(pool)
        # base no-op: cache untouched
        self.assertIsNotNone(pool._cached_loc_key)


class TestExplicitInvalidationCycle(CustomTestCase):
    """Simulates the per-forward-pass invalidation done by model_runner."""

    def test_fresh_translation_after_explicit_invalidation(self):
        """After invalidate_loc_cache(), a new alloc produces the right mapping."""
        pool, allocator, device = _build_pool(kv_size=32, kv_size_swa=32)

        # First "forward pass": alloc 4 tokens, translate.
        loc1 = allocator.alloc(4)
        trans1 = pool.translate_loc_from_full_to_swa(loc1).clone()

        # Simulate start of next forward pass: model_runner calls invalidate.
        pool.invalidate_loc_cache()
        self.assertIsNone(pool._cached_loc_key)

        # Alloc 4 more (mapping changes), translate loc1 again.
        loc2 = allocator.alloc(4)
        # Alloc already invalidated; translate loc1 with fresh mapping.
        trans1_after = pool.translate_loc_from_full_to_swa(loc1)

        # loc1's SWA mapping hasn't changed (same full→swa assignment),
        # so result should be equal — but it must have been recomputed
        # (cache key was None before this call).
        self.assertEqual(trans1.tolist(), trans1_after.tolist())

        # loc2 should have different translation than loc1.
        trans2 = pool.translate_loc_from_full_to_swa(loc2)
        # They have different indices, so translation differs.
        self.assertFalse(torch.equal(trans1_after, trans2))


if __name__ == "__main__":
    unittest.main()
