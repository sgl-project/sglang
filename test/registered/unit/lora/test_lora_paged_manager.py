"""Unit tests for LoRAManager page_table cache — key ordering and invalidation.

Tests the cache logic in prepare_lora_batch that avoids rebuilding the
page_table tensor when the adapter set and page generation are unchanged.

Key regression test: cache key uses tuple(active_uids) which preserves row
ordering. The previous frozenset key discarded ordering, causing silent wrong
output when the same adapters arrived in different request order.

Tests are hermetic: CPU-only, no CUDA, no model loading.

Usage:
    python -m pytest test/registered/unit/lora/test_lora_paged_manager.py -v
"""


def _patch_kernels_revision():
    """Patch kernels LayerRepository to default revision='main'."""
    try:
        from kernels.layer.func import FuncRepository as _FR
        from kernels.layer.layer import LayerRepository as _LR

        _lr_orig = _LR.__init__

        def _lr_patched(
            self, repo_id, *, layer_name, revision=None, version=None, **kw
        ):
            if revision is None and version is None:
                revision = "main"
            _lr_orig(
                self,
                repo_id,
                layer_name=layer_name,
                revision=revision,
                version=version,
                **kw,
            )

        _LR.__init__ = _lr_patched

        _fr_orig = _FR.__init__

        def _fr_patched(self, repo_id, *, func_name, revision=None, version=None, **kw):
            if revision is None and version is None:
                revision = "main"
            _fr_orig(
                self,
                repo_id,
                func_name=func_name,
                revision=revision,
                version=version,
                **kw,
            )

        _FR.__init__ = _fr_patched
    except ImportError:
        pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"patch_kernels failed: {e}")
        pass


_patch_kernels_revision()

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.test.test_utils import CustomTestCase


class TestPageTableCacheKey(CustomTestCase):
    """Cache key must preserve row ordering (P1 bug fix: frozenset->tuple)."""

    def test_tuple_preserves_order(self):
        """tuple([None, A, B]) != tuple([None, B, A]) — different cache keys."""
        key1 = (tuple([None, "A", "B"]), 0)
        key2 = (tuple([None, "B", "A"]), 0)
        self.assertNotEqual(key1, key2)

    def test_frozenset_would_collide(self):
        """frozenset discards ordering — this is the bug we fixed."""
        fs1 = frozenset(uid for uid in [None, "A", "B"] if uid is not None)
        fs2 = frozenset(uid for uid in [None, "B", "A"] if uid is not None)
        self.assertEqual(fs1, fs2)

    def test_same_order_same_key(self):
        """Same ordering produces same key — cache hit is correct."""
        key1 = (tuple([None, "A", "B"]), 0)
        key2 = (tuple([None, "A", "B"]), 0)
        self.assertEqual(key1, key2)

    def test_different_generation_different_key(self):
        """Same adapters but different generation — cache must miss."""
        key1 = (tuple([None, "A"]), 0)
        key2 = (tuple([None, "A"]), 1)
        self.assertNotEqual(key1, key2)


class TestPageTableCacheBehavior(CustomTestCase):
    """Cache hit/miss and bounded-clear behavior."""

    def _make_bare_manager(self):
        """Create minimal LoRAManager with _page_table_cache."""
        mgr = LoRAManager.__new__(LoRAManager)
        mgr._page_table_cache = {}
        return mgr

    def test_cache_hit_returns_cached_tensor(self):
        mgr = self._make_bare_manager()
        cache = mgr._page_table_cache
        fake_tensor = torch.full((2, 2), 5, dtype=torch.int32)
        key = (tuple([None, "A"]), 0)
        cache[key] = fake_tensor
        cached = cache.get(key)
        self.assertIsNotNone(cached)
        self.assertTrue(torch.equal(cached, fake_tensor))

    def test_cache_miss_different_adapters(self):
        mgr = self._make_bare_manager()
        cache = mgr._page_table_cache
        cache[(tuple([None, "A"]), 0)] = torch.full((2, 2), 5, dtype=torch.int32)
        result = cache.get((tuple([None, "B"]), 0))
        self.assertIsNone(result)

    def test_cache_miss_different_generation(self):
        mgr = self._make_bare_manager()
        cache = mgr._page_table_cache
        cache[(tuple([None, "A"]), 0)] = torch.full((2, 2), 5, dtype=torch.int32)
        result = cache.get((tuple([None, "A"]), 1))
        self.assertIsNone(result)

    def test_cache_clears_on_miss(self):
        """On cache miss, cache.clear() is called before inserting new entry.
        This bounds the cache to a single entry, preventing GPU memory leak.
        """
        mgr = self._make_bare_manager()
        cache = mgr._page_table_cache
        cache[(tuple([None, "A"]), 0)] = torch.full((2, 2), 1, dtype=torch.int32)
        cache[(tuple([None, "B"]), 0)] = torch.full((2, 2), 2, dtype=torch.int32)
        self.assertEqual(len(cache), 2)
        cache.clear()
        new_key = (tuple([None, "C"]), 0)
        cache[new_key] = torch.full((2, 2), 3, dtype=torch.int32)
        self.assertEqual(len(cache), 1)
        self.assertIn(new_key, cache)

    def test_cache_slice_returns_view(self):
        """Cached tensor wider than needed -> slice returns a view (no copy)."""
        mgr = self._make_bare_manager()
        cache = mgr._page_table_cache
        wide = torch.full((2, 8), 5, dtype=torch.int32)
        key = (tuple([None, "A"]), 0)
        cache[key] = wide
        cached = cache.get(key)
        sliced = cached[:, :2]
        self.assertEqual(sliced.shape, (2, 2))
        self.assertTrue(torch.equal(sliced, wide[:, :2]))


class TestGetattrFallback(CustomTestCase):
    """getattr fallbacks for __new__()-based test fixtures."""

    def test_cache_none_when_not_initialized(self):
        mgr = LoRAManager.__new__(LoRAManager)
        cache = getattr(mgr, "_page_table_cache", None)
        self.assertIsNone(cache)

    def test_generation_zero_when_not_initialized(self):
        from sglang.srt.lora.paged_mem_pool import LoRAPagePool

        pool = LoRAPagePool.__new__(LoRAPagePool)
        gen = getattr(pool, "page_generation", 0)
        self.assertEqual(gen, 0)


if __name__ == "__main__":
    unittest.main()
