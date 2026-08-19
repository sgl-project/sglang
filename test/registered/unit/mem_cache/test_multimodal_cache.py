"""Unit tests for mem_cache/multimodal_cache.py — no server, no model loading.

Covers ``MultiModalStaticCache``: the byte-budgeted LRU cache used to reuse
precomputed multimodal embeddings across requests. All cases run on CPU with
tiny tensors; the only tensor property the cache reads is byte size
(``element_size() * numel()``).
"""

import unittest

import torch

from sglang.srt.mem_cache.multimodal_cache import (
    EmbeddingResult,
    MultiModalStaticCache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _emb(num_floats: int = 10) -> EmbeddingResult:
    """An EmbeddingResult wrapping a float32 tensor of ``4 * num_floats`` bytes."""
    return EmbeddingResult(embedding=torch.zeros(num_floats, dtype=torch.float32))


class TestCombineHashes(CustomTestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(MultiModalStaticCache.combine_hashes([]))

    def test_deterministic(self):
        self.assertEqual(
            MultiModalStaticCache.combine_hashes([1, 2, 3]),
            MultiModalStaticCache.combine_hashes([1, 2, 3]),
        )

    def test_matches_tuple_hash(self):
        self.assertEqual(MultiModalStaticCache.combine_hashes([7, 8]), hash((7, 8)))

    def test_order_sensitive(self):
        self.assertNotEqual(
            MultiModalStaticCache.combine_hashes([1, 2]),
            MultiModalStaticCache.combine_hashes([2, 1]),
        )


class TestBasicSetGet(CustomTestCase):
    def setUp(self):
        self.cache = MultiModalStaticCache(max_size=1000)

    def test_set_then_get_single(self):
        emb = _emb()
        self.assertTrue(self.cache.set(123, emb))
        self.assertTrue(self.cache.has(123))
        self.assertIs(self.cache.get_single(123), emb)
        self.assertEqual(len(self.cache), 1)
        self.assertEqual(self.cache.available_size(), 1)

    def test_get_single_missing_returns_none(self):
        self.assertIsNone(self.cache.get_single(999))

    def test_has_false_for_missing(self):
        self.assertFalse(self.cache.has(999))

    def test_set_rejects_non_embedding_result(self):
        # set() asserts the value is an EmbeddingResult, not a raw tensor.
        with self.assertRaises(AssertionError):
            self.cache.set(1, torch.zeros(3, dtype=torch.float32))

    def test_reset_existing_hash_keeps_single_entry(self):
        emb = _emb()
        self.assertTrue(self.cache.set(1, emb))
        size_after_first = self.cache.current_size
        # Re-inserting the same key must not double-count size or length.
        self.assertTrue(self.cache.set(1, emb))
        self.assertEqual(self.cache.current_size, size_after_first)
        self.assertEqual(len(self.cache), 1)


class TestGetByCombinedHash(CustomTestCase):
    def setUp(self):
        self.cache = MultiModalStaticCache(max_size=1000)

    def test_get_hits_via_combined_hash(self):
        combined = MultiModalStaticCache.combine_hashes([11, 22])
        emb = _emb()
        self.cache.set(combined, emb)
        # get() recomputes the combined hash from the item list.
        self.assertIs(self.cache.get([11, 22]), emb)

    def test_get_miss_returns_none(self):
        self.assertIsNone(self.cache.get([33]))

    def test_get_empty_list_returns_none(self):
        # combine_hashes([]) is None, which is never a stored key.
        self.assertIsNone(self.cache.get([]))


class TestEviction(CustomTestCase):
    def test_evicts_least_recently_used(self):
        # max_size fits exactly two 40-byte tensors.
        cache = MultiModalStaticCache(max_size=100)
        cache.set(1, _emb())
        cache.set(2, _emb())
        self.assertEqual(len(cache), 2)

        cache.set(3, _emb())  # forces eviction of the oldest (key 1)
        self.assertFalse(cache.has(1))
        self.assertTrue(cache.has(2))
        self.assertTrue(cache.has(3))
        self.assertLessEqual(cache.current_size, 100)

    def test_access_refreshes_recency(self):
        cache = MultiModalStaticCache(max_size=100)
        cache.set(1, _emb())
        cache.set(2, _emb())

        # Touch key 1 so key 2 becomes the least recently used.
        self.assertIsNotNone(cache.get_single(1))
        cache.set(3, _emb())

        self.assertTrue(cache.has(1))
        self.assertFalse(cache.has(2))
        self.assertTrue(cache.has(3))

    def test_oversized_item_rejected_on_empty_cache(self):
        cache = MultiModalStaticCache(max_size=30)  # smaller than one 40-byte tensor
        self.assertFalse(cache.set(1, _emb()))
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.current_size, 0)

    def test_current_size_tracks_inserts(self):
        cache = MultiModalStaticCache(max_size=1000)
        cache.set(1, _emb(10))  # 40 bytes
        self.assertEqual(cache.current_size, 40)
        cache.set(2, _emb(5))  # 20 bytes
        self.assertEqual(cache.current_size, 60)


class TestFreeAndClear(CustomTestCase):
    def setUp(self):
        self.cache = MultiModalStaticCache(max_size=1000)

    def test_free_removes_and_updates_size(self):
        self.cache.set(1, _emb())
        self.cache.set(2, _emb())
        self.assertTrue(self.cache.free(1, None))
        self.assertFalse(self.cache.has(1))
        self.assertEqual(len(self.cache), 1)
        self.assertEqual(self.cache.current_size, 40)

    def test_free_missing_returns_false(self):
        self.assertFalse(self.cache.free(999, None))

    def test_clear_resets_cache(self):
        self.cache.set(1, _emb())
        self.cache.set(2, _emb())
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.current_size, 0)
        self.assertFalse(self.cache.has(1))


if __name__ == "__main__":
    unittest.main()
