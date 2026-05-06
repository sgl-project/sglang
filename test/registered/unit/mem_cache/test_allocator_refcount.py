import unittest

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)


class DummyKVCache:
    def get_cpu_copy(self, indices):
        return indices

    def load_cpu_copy(self, kv_cache_cpu, indices):
        return kv_cache_cpu, indices


class TestAllocatorRefcount(unittest.TestCase):
    def test_token_allocator_retain_delays_reuse(self):
        allocator = TokenToKVPoolAllocator(
            size=8,
            dtype=torch.float16,
            device="cpu",
            kvcache=DummyKVCache(),
            need_sort=False,
        )

        indices = allocator.alloc(3)
        self.assertEqual(indices.tolist(), [1, 2, 3])
        allocator.retain(indices)

        allocator.free(indices)
        self.assertEqual(allocator.available_size(), 5)
        self.assertTrue(
            torch.equal(allocator.ref_counts[indices], torch.ones(3, dtype=torch.int32))
        )

        allocator.free(indices)
        self.assertEqual(allocator.available_size(), 8)
        self.assertTrue(
            torch.equal(
                allocator.ref_counts[indices], torch.zeros(3, dtype=torch.int32)
            )
        )

    def test_paged_allocator_retain_delays_page_reuse(self):
        allocator = PagedTokenToKVPoolAllocator(
            size=8,
            page_size=2,
            dtype=torch.float16,
            device="cpu",
            kvcache=DummyKVCache(),
            need_sort=False,
        )

        indices = allocator.alloc(4)
        self.assertEqual(indices.tolist(), [2, 3, 4, 5])
        allocator.retain(indices)

        allocator.free(indices)
        self.assertEqual(allocator.available_size(), 4)
        self.assertTrue(
            torch.equal(
                allocator.ref_counts[torch.tensor([1, 2])],
                torch.ones(2, dtype=torch.int32),
            )
        )

        allocator.free(indices)
        self.assertEqual(allocator.available_size(), 8)
        self.assertTrue(
            torch.equal(
                allocator.ref_counts[torch.tensor([1, 2])],
                torch.zeros(2, dtype=torch.int32),
            )
        )

    def test_double_free_raises(self):
        allocator = TokenToKVPoolAllocator(
            size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=DummyKVCache(),
            need_sort=False,
        )

        indices = allocator.alloc(2)
        allocator.free(indices)

        with self.assertRaises(ValueError):
            allocator.free(indices)


if __name__ == "__main__":
    unittest.main()
