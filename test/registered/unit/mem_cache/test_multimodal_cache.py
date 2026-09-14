"""Unit tests for ``python/sglang/srt/mem_cache/multimodal_cache.py``.

These tests exercise the in-process multimodal embedding cache
(``MultiModalStaticCache``) used to reuse precomputed VLM image/video
embeddings across requests (see ``managers/mm_schedule.py`` and
``disaggregation/encode_server.py``).

The cache is a byte-budgeted LRU over ``EmbeddingResult`` objects. The focus is
on the real invariants a regression could silently break:

* byte-level capacity accounting (``current_size``) staying consistent with the
  stored tensors after arbitrary ``set``/``get``/``free``/``clear`` sequences;
* LRU eviction order tracking access recency (not insertion order);
* the oversized-item rejection path leaving the cache in a consistent state;
* the ``set``/``get`` (combined-hash, multi-item) and
  ``set``/``get_single`` (raw-hash, single-item) retrieval contracts, which is
  how the two production call sites use this cache.

No server is launched and no model weights are loaded. Tensors are kept on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.multimodal_cache import (
    EmbeddingResult,
    MultiModalStaticCache,
    _get_tensor_size,
)
from sglang.test.test_utils import CustomTestCase


def _emb(num_bytes: int, dtype: torch.dtype = torch.float32) -> EmbeddingResult:
    """Build an ``EmbeddingResult`` whose tensor occupies exactly ``num_bytes``.

    ``_get_tensor_size`` is ``element_size * numel``; float32 elements are 4
    bytes each, so passing ``num_bytes`` that is a multiple of the dtype
    element size yields an exact byte budget.
    """
    assert num_bytes % torch.tensor([], dtype=dtype).element_size() == 0
    numel = num_bytes // torch.tensor([], dtype=dtype).element_size()
    return EmbeddingResult(embedding=torch.zeros(numel, dtype=dtype))


class TestCombineHashes(CustomTestCase):
    def test_empty_list_returns_none(self):
        # An empty hash list has no combined identity; the production callers
        # rely on None to signal "nothing to look up".
        self.assertIsNone(MultiModalStaticCache.combine_hashes([]))

    def test_single_element_equals_tuple_hash(self):
        self.assertEqual(MultiModalStaticCache.combine_hashes([42]), hash((42,)))

    def test_multiple_elements_equals_tuple_hash(self):
        self.assertEqual(
            MultiModalStaticCache.combine_hashes([1, 2, 3]), hash((1, 2, 3))
        )

    def test_is_deterministic(self):
        hashes = [7, 13, -5, 0]
        self.assertEqual(
            MultiModalStaticCache.combine_hashes(hashes),
            MultiModalStaticCache.combine_hashes(list(hashes)),
        )

    def test_order_matters(self):
        # Reordering the items changes the tuple, hence the hash. This guards
        # against accidentally sorting/normalizing the input inside the cache.
        self.assertNotEqual(
            MultiModalStaticCache.combine_hashes([1, 2]),
            MultiModalStaticCache.combine_hashes([2, 1]),
        )


class TestTensorSizeHelper(CustomTestCase):
    def test_byte_size_is_element_size_times_numel(self):
        # Capacity accounting is built on this helper; pin its contract across
        # the dtypes a VLM embedding tensor can actually take.
        self.assertEqual(_get_tensor_size(torch.zeros(8, dtype=torch.float32)), 32)
        self.assertEqual(_get_tensor_size(torch.zeros(8, dtype=torch.float16)), 16)
        self.assertEqual(_get_tensor_size(torch.zeros(8, dtype=torch.int64)), 64)

    def test_zero_element_tensor_is_zero_bytes(self):
        self.assertEqual(_get_tensor_size(torch.zeros(0, dtype=torch.float32)), 0)


class TestSetAndCapacityAccounting(CustomTestCase):
    def test_set_new_returns_true_and_increments_size(self):
        cache = MultiModalStaticCache(max_size=1024)
        self.assertTrue(cache.set(1, _emb(32)))
        self.assertEqual(cache.current_size, 32)
        self.assertEqual(len(cache), 1)

    def test_size_accounts_for_dtype(self):
        # A regression that hard-coded a 4-byte element assumption would make
        # float16 / int64 accounting drift; cover both explicitly.
        cache = MultiModalStaticCache(max_size=1024)
        self.assertTrue(cache.set(1, _emb(16, dtype=torch.float16)))
        self.assertEqual(cache.current_size, 16)
        self.assertTrue(cache.set(2, _emb(32, dtype=torch.int64)))
        self.assertEqual(cache.current_size, 48)

    def test_repeated_key_refreshes_lru_without_replacing(self):
        # Re-setting an existing key is an LRU refresh only: the original
        # embedding is kept and the byte budget is NOT touched. A bug that
        # re-appended (double counting) or that replaced in place (changing
        # the stored size) would both break this invariant.
        cache = MultiModalStaticCache(max_size=1024)
        cache.set(1, _emb(32))
        self.assertTrue(cache.set(1, _emb(48)))
        self.assertEqual(cache.current_size, 32)
        self.assertEqual(len(cache), 1)
        self.assertEqual(_get_tensor_size(cache.get_single(1).embedding), 32)

    def test_current_size_matches_sum_of_stored_bytes(self):
        # Core invariant: after an arbitrary operation mix, the recorded
        # current_size equals the true sum of stored tensor byte sizes.
        cache = MultiModalStaticCache(max_size=512)
        cache.set(1, _emb(32))
        cache.set(2, _emb(16, dtype=torch.float16))
        cache.set(3, _emb(64))
        cache.get_single(1)  # touches LRU, must not change accounting
        cache.free(2, None)
        cache.set(1, _emb(20))  # refresh LRU on existing key, must not accumulate

        expected = sum(_get_tensor_size(cache.get_single(k).embedding) for k in [1, 3])
        self.assertEqual(cache.current_size, expected)

    def test_set_rejects_non_embedding_result(self):
        cache = MultiModalStaticCache(max_size=1024)
        with self.assertRaises(AssertionError):
            cache.set(1, torch.zeros(4))  # raw tensor, not an EmbeddingResult


class TestLRUEviction(CustomTestCase):
    def test_evicts_least_recently_used_first(self):
        # Fill to exactly the byte budget, then force one eviction. The oldest
        # inserted key must be the one dropped.
        cache = MultiModalStaticCache(max_size=16)
        for key in (1, 2, 3, 4):  # 4 * 4 bytes = 16, cache is full
            self.assertTrue(cache.set(key, _emb(4)))
        self.assertTrue(cache.set(5, _emb(4)))  # needs to evict key 1

        self.assertFalse(cache.has(1))
        for still_present in (2, 3, 4, 5):
            self.assertTrue(cache.has(still_present))
        self.assertEqual(cache.current_size, 16)

    def test_get_single_updates_recency(self):
        # Touching an item via get_single promotes it to most-recently-used, so
        # a subsequent insertion must evict a different (older) key.
        cache = MultiModalStaticCache(max_size=16)
        for key in (1, 2, 3, 4):
            cache.set(key, _emb(4))
        self.assertIsNotNone(cache.get_single(1))  # key 1 is now most recent

        cache.set(5, _emb(4))  # oldest is now key 2, not key 1
        self.assertTrue(cache.has(1))
        self.assertFalse(cache.has(2))

    def test_repeated_set_moves_key_to_end(self):
        # set on an existing key refreshes recency via move_to_end.
        cache = MultiModalStaticCache(max_size=16)
        for key in (1, 2, 3, 4):
            cache.set(key, _emb(4))
        cache.set(1, _emb(4))  # key 1 promoted to most recent

        cache.set(5, _emb(4))  # oldest is now key 2
        self.assertTrue(cache.has(1))
        self.assertFalse(cache.has(2))
        self.assertEqual(cache.current_size, 16)


class TestOversizedItemAndBoundaries(CustomTestCase):
    def test_oversized_item_rejected_and_clears_cache(self):
        # An item that cannot fit even in an empty cache must be rejected and
        # must leave the cache empty and accounting-consistent.
        cache = MultiModalStaticCache(max_size=16)
        cache.set(1, _emb(4))
        self.assertFalse(cache.set(99, _emb(64)))  # 64 > 16 max

        self.assertEqual(cache.current_size, 0)
        self.assertEqual(len(cache), 0)
        self.assertFalse(cache.has(1))
        self.assertFalse(cache.has(99))

    def test_item_requiring_multiple_evictions(self):
        # A medium item that fits only after evicting several small entries.
        cache = MultiModalStaticCache(max_size=8)
        cache.set(1, _emb(4))
        cache.set(2, _emb(4))  # full (8 bytes)
        self.assertTrue(cache.set(3, _emb(8)))  # evict 1 and 2, then store 3

        self.assertTrue(cache.has(3))
        self.assertFalse(cache.has(1))
        self.assertFalse(cache.has(2))
        self.assertEqual(cache.current_size, 8)

    def test_exact_remaining_capacity_fits_without_eviction(self):
        # An item whose byte size exactly equals the remaining budget must not
        # trigger any eviction (the guard is strictly greater-than).
        cache = MultiModalStaticCache(max_size=16)
        cache.set(1, _emb(4))  # remaining = 12
        self.assertTrue(cache.set(2, _emb(12)))  # 4 + 12 == 16, fits exactly

        self.assertTrue(cache.has(1))
        self.assertEqual(cache.current_size, 16)

    def test_one_byte_over_triggers_eviction(self):
        # One byte beyond remaining capacity forces a single eviction, proving
        # the boundary is strict ">" rather than ">=". Uses a 1-byte-element
        # dtype so the budget can be exercised to single-byte precision.
        cache = MultiModalStaticCache(max_size=16)
        cache.set(1, _emb(4, dtype=torch.uint8))  # remaining = 12
        self.assertTrue(cache.set(2, _emb(13, dtype=torch.uint8)))  # 4+13>16

        self.assertFalse(cache.has(1))
        self.assertTrue(cache.has(2))
        self.assertEqual(cache.current_size, 13)


class TestGetCombinedHashPath(CustomTestCase):
    def test_get_miss_returns_none(self):
        cache = MultiModalStaticCache(max_size=64)
        self.assertIsNone(cache.get([12345]))

    def test_get_hits_when_stored_under_combined_hash(self):
        # Multi-item production contract: store under combine_hashes(...) and
        # retrieve via get(item_hashes), which recomputes the same combined key.
        cache = MultiModalStaticCache(max_size=64)
        combined = MultiModalStaticCache.combine_hashes([7, 8])
        cache.set(combined, _emb(16))

        result = cache.get([7, 8])
        self.assertIsNotNone(result)
        self.assertIsInstance(result, EmbeddingResult)

    def test_get_updates_recency(self):
        # Accessing an entry via get() must promote it to most-recently-used,
        # so a later insertion evicts a different (older) entry. Store each
        # entry under its combined hash so get() can actually hit them.
        cache = MultiModalStaticCache(max_size=16)
        for key in (1, 2, 3, 4):
            cache.set(MultiModalStaticCache.combine_hashes([key]), _emb(4))
        self.assertIsNotNone(cache.get([1]))  # promote combined-[1]

        cache.set(MultiModalStaticCache.combine_hashes([5]), _emb(4))  # evict [2]
        self.assertIsNotNone(cache.get([1]))  # still present after promotion
        self.assertIsNone(cache.get([2]))  # evicted as oldest

    def test_explicit_combined_hash_is_consistent_with_default(self):
        # A matching combined_hash must behave identically to the default path
        # (hit + EmbeddingResult). This holds whether the cache honors the
        # parameter as a fast path or recomputes it internally, so it does not
        # lock in either implementation choice.
        cache = MultiModalStaticCache(max_size=64)
        combined = MultiModalStaticCache.combine_hashes([7, 8])
        cache.set(combined, _emb(16))

        default = cache.get([7, 8])
        explicit = cache.get([7, 8], combined_hash=combined)
        self.assertIsNotNone(default)
        self.assertIsNotNone(explicit)
        self.assertIsInstance(explicit, EmbeddingResult)
        self.assertEqual(type(explicit), type(default))


class TestGetSingleRawHashPath(CustomTestCase):
    def test_get_single_miss_returns_none(self):
        cache = MultiModalStaticCache(max_size=64)
        self.assertIsNone(cache.get_single(5))

    def test_get_single_hits_raw_key(self):
        # Single-item production contract: store and retrieve under the same raw
        # hash (no combine).
        cache = MultiModalStaticCache(max_size=64)
        cache.set(5, _emb(16))
        result = cache.get_single(5)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, EmbeddingResult)

    def test_get_single_updates_recency(self):
        cache = MultiModalStaticCache(max_size=16)
        for key in (1, 2, 3, 4):
            cache.set(key, _emb(4))
        self.assertIsNotNone(cache.get_single(2))  # promote key 2
        cache.set(5, _emb(4))  # oldest is now key 1
        self.assertTrue(cache.has(2))
        self.assertFalse(cache.has(1))


class TestFreeClearAndAvailability(CustomTestCase):
    def test_free_present_returns_true_and_decrements(self):
        cache = MultiModalStaticCache(max_size=64)
        cache.set(1, _emb(32))
        self.assertTrue(cache.free(1, None))
        self.assertEqual(cache.current_size, 0)
        self.assertFalse(cache.has(1))

    def test_free_absent_returns_false_and_is_noop(self):
        cache = MultiModalStaticCache(max_size=64)
        cache.set(1, _emb(16))
        self.assertFalse(cache.free(999, None))
        self.assertEqual(cache.current_size, 16)
        self.assertTrue(cache.has(1))

    def test_free_keeps_accounting_consistent(self):
        # After frees interleaved with sets, current_size must still equal the
        # sum of the remaining stored tensors.
        cache = MultiModalStaticCache(max_size=256)
        cache.set(1, _emb(32))
        cache.set(2, _emb(16))
        cache.free(1, None)
        cache.set(3, _emb(48))
        cache.free(2, None)

        expected = _get_tensor_size(cache.get_single(3).embedding)
        self.assertEqual(cache.current_size, expected)
        self.assertEqual(len(cache), 1)

    def test_clear_resets_state(self):
        cache = MultiModalStaticCache(max_size=64)
        cache.set(1, _emb(16))
        cache.set(2, _emb(16))
        cache.clear()
        self.assertEqual(cache.current_size, 0)
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.available_size(), 0)

    def test_available_size_equals_entry_count(self):
        # available_size / __len__ report entry count (not byte usage), matching
        # the semantics callers like mm_schedule rely on.
        cache = MultiModalStaticCache(max_size=64)
        cache.set(1, _emb(16))
        cache.set(2, _emb(16))
        self.assertEqual(cache.available_size(), len(cache))
        self.assertEqual(cache.available_size(), 2)


class TestEmbeddingResult(CustomTestCase):
    def test_wraps_embedding_tensor(self):
        tensor = torch.arange(4, dtype=torch.float32)
        result = EmbeddingResult(embedding=tensor)
        self.assertTrue(torch.equal(result.embedding, tensor))


if __name__ == "__main__":
    unittest.main()
