"""Byte-budget admission tests for the multimodal embedding cache."""

import unittest

import torch

from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMultimodalCacheAdmission(unittest.TestCase):
    def test_oversized_item_preserves_entries_and_lru_order(self):
        for tensor in (torch.ones(9), torch.ones(18)[::2]):
            with self.subTest(stride=tensor.stride()):
                cache = MultiModalStaticCache(max_size=32)
                first = EmbeddingResult(embedding=torch.arange(4, dtype=torch.float32))
                second = EmbeddingResult(embedding=torch.full((4,), 2.0))
                self.assertTrue(cache.set(1, first))
                self.assertTrue(cache.set(2, second))

                self.assertFalse(cache.set(3, EmbeddingResult(embedding=tensor)))
                self.assertEqual(list(cache.mm_cache), [1, 2])
                self.assertEqual(cache.current_size, 32)
                self.assertIs(cache.mm_cache[1], first)
                self.assertIs(cache.mm_cache[2], second)

                # A later admissible entry still evicts the original LRU only.
                self.assertTrue(
                    cache.set(4, EmbeddingResult(embedding=torch.full((4,), 4.0)))
                )
                self.assertEqual(list(cache.mm_cache), [2, 4])
                self.assertEqual(cache.current_size, 32)

    def test_admissible_view_uses_payload_bytes(self):
        cache = MultiModalStaticCache(max_size=16)
        backing = torch.arange(100, dtype=torch.float32)
        view = backing[20:24]
        self.assertGreater(view.untyped_storage().nbytes(), cache.max_size)

        self.assertTrue(cache.set(1, EmbeddingResult(embedding=view)))
        cached = cache.get_single(1).embedding
        torch.testing.assert_close(cached, view)
        self.assertEqual(cache.current_size, 16)
        self.assertEqual(cached.untyped_storage().nbytes(), 16)
        self.assertNotEqual(cached.data_ptr(), view.data_ptr())

    def test_duplicate_key_keeps_existing_value_and_updates_lru(self):
        cache = MultiModalStaticCache(max_size=32)
        first = EmbeddingResult(embedding=torch.ones(4))
        self.assertTrue(cache.set(1, first))
        self.assertTrue(cache.set(2, EmbeddingResult(embedding=torch.ones(4))))

        self.assertTrue(cache.set(1, EmbeddingResult(embedding=torch.ones(100))))
        self.assertEqual(list(cache.mm_cache), [2, 1])
        self.assertIs(cache.get_single(1), first)
        self.assertEqual(cache.current_size, 32)

    def test_exact_and_zero_capacity_boundaries(self):
        for capacity in (0, 16):
            with self.subTest(capacity=capacity):
                cache = MultiModalStaticCache(max_size=capacity)
                exact = EmbeddingResult(embedding=torch.ones(capacity // 4))
                self.assertTrue(cache.set(1, exact))
                self.assertFalse(
                    cache.set(
                        2, EmbeddingResult(embedding=torch.ones(capacity // 4 + 1))
                    )
                )
                self.assertIs(cache.get_single(1), exact)
                self.assertEqual(cache.current_size, capacity)


if __name__ == "__main__":
    unittest.main()
