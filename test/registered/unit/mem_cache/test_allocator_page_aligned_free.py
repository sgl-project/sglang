import unittest
from pathlib import Path
from unittest import mock

import sglang
import torch

import sglang.srt.mem_cache.allocator.base as allocator_base
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAllocatorPageAlignedFree(unittest.TestCase):
    def _paged_allocator(self, *, need_sort: bool) -> PagedTokenToKVPoolAllocator:
        return PagedTokenToKVPoolAllocator(
            size=16,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=need_sort,
        )

    def test_paged_free_preserves_need_sort_release_order(self) -> None:
        """Paged free returns strided page representatives to either free list."""
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = self._paged_allocator(need_sort=need_sort)
                indices = allocator.alloc(8)

                allocator.free(indices)

                if need_sort:
                    self.assertEqual(allocator.free_pages.tolist(), [3, 4])
                    self.assertEqual(allocator.release_pages.tolist(), [1, 2])
                else:
                    self.assertEqual(allocator.free_pages.tolist(), [1, 2, 3, 4])
                    self.assertEqual(allocator.release_pages.tolist(), [])

    def test_paged_free_rejects_partial_fragment_before_mutation(self) -> None:
        """Paged free rejects a partial page before direct or grouped mutation."""
        allocator = self._paged_allocator(need_sort=False)
        indices = allocator.alloc(4)
        free_pages_before = allocator.free_pages.clone()

        with self.assertRaisesRegex(AssertionError, "complete page blocks"):
            allocator.free(indices[:3])

        self.assertTrue(torch.equal(allocator.free_pages, free_pages_before))
        allocator.free_group_begin()

        with self.assertRaisesRegex(AssertionError, "complete page blocks"):
            allocator.free(indices[:3])

        self.assertEqual(allocator.free_group, [])
        self.assertTrue(torch.equal(allocator.free_pages, free_pages_before))

    def test_debug_free_rejects_malformed_or_duplicate_page_blocks(self) -> None:
        """Debug validation rejects alignment, continuity, and uniqueness errors."""
        invalid_blocks = (
            torch.tensor([5, 6, 7, 8], dtype=torch.int64),
            torch.tensor([4, 5, 7, 8], dtype=torch.int64),
            torch.tensor([4, 5, 6, 7, 4, 5, 6, 7], dtype=torch.int64),
        )

        with mock.patch.object(allocator_base, "_DEBUG_MEMORY_POOL", True):
            for free_index in invalid_blocks:
                with self.subTest(free_index=free_index.tolist()):
                    allocator = self._paged_allocator(need_sort=False)
                    free_pages_before = allocator.free_pages.clone()

                    with self.assertRaisesRegex(RuntimeError, "unique, aligned"):
                        allocator.free(free_index)

                    self.assertTrue(
                        torch.equal(allocator.free_pages, free_pages_before)
                    )

    def test_debug_free_group_rejects_duplicate_merged_page(self) -> None:
        """Debug validation rechecks merged groups for duplicate complete pages."""
        allocator = self._paged_allocator(need_sort=False)
        indices = allocator.alloc(4)
        allocator.free_group_begin()

        with mock.patch.object(allocator_base, "_DEBUG_MEMORY_POOL", True):
            allocator.free(indices)
            allocator.free(indices)

            with self.assertRaisesRegex(RuntimeError, "unique, aligned"):
                allocator.free_group_end()

        self.assertEqual(allocator.free_pages.tolist(), [2, 3, 4])

    def test_token_free_group_and_debug_duplicate_contract(self) -> None:
        """Token free groups work while debug mode rejects duplicate page-one IDs."""
        allocator = TokenToKVPoolAllocator(
            size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        indices = allocator.alloc(2)
        allocator.free_group_begin()
        allocator.free(indices[:1])
        allocator.free_group_end()
        self.assertEqual(allocator.free_pages.tolist(), [1, 3, 4])

        with mock.patch.object(allocator_base, "_DEBUG_MEMORY_POOL", True):
            with self.assertRaisesRegex(RuntimeError, "unique, aligned"):
                allocator.free(indices[1:].repeat(2))

        self.assertEqual(allocator.free_pages.tolist(), [1, 3, 4])

    def test_npu_paged_free_keeps_legacy_cpu_unique(self) -> None:
        """NPU paged free retains its partial and unordered CPU unique path."""
        source_path = (
            Path(sglang.__file__).resolve().parent
            / "srt/hardware_backend/npu/allocator_npu.py"
        )
        source = source_path.read_text()

        self.assertIn(
            "torch.unique(free_index.cpu() // self.page_size)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
