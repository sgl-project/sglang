import inspect
import unittest
from pathlib import Path
from unittest import mock

import torch

import sglang
import sglang.srt.mem_cache.allocator.base as allocator_base
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import PureSWATokenToKVPoolAllocator
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

    def test_pure_swa_group_defers_sentinel_normalization(self) -> None:
        """PureSWA groups raw fragments and normalizes sentinels once at merge."""
        allocator = object.__new__(PureSWATokenToKVPoolAllocator)
        allocator.page_size = 1
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator.swa_attn_allocator = mock.Mock(size=8)
        allocator.swa_attn_allocator.available_size.return_value = 8
        first_fragment = torch.tensor([0, 1, -1, 0], dtype=torch.int64)
        second_fragment = torch.tensor([2, 0, 3, -1], dtype=torch.int64)

        allocator.free_group_begin()
        with mock.patch.object(allocator_base, "_DEBUG_MEMORY_POOL", True):
            allocator.free(first_fragment)
            allocator.free(second_fragment)

            self.assertIs(allocator.free_group[0], first_fragment)
            self.assertIs(allocator.free_group[1], second_fragment)
            allocator.free_group_end()

        allocator.swa_attn_allocator.free.assert_called_once()
        released = allocator.swa_attn_allocator.free.call_args.args[0]
        self.assertEqual(released.tolist(), [1, 2, 3])
        source = inspect.getsource(PureSWATokenToKVPoolAllocator.free)
        self.assertEqual(source.count("free_index[free_index > 0]"), 1)
        self.assertLess(
            source.index("self.free_group.append(free_index)"),
            source.index("sanitized_free_index ="),
        )

    def test_pure_swa_debug_rejects_duplicate_positive_tokens(self) -> None:
        """PureSWA debug validation ignores sentinels but rejects positive reuse."""
        allocator = object.__new__(PureSWATokenToKVPoolAllocator)
        allocator.page_size = 1
        allocator.is_not_in_free_group = True
        allocator.free_group = []
        allocator.swa_attn_allocator = mock.Mock(size=8)
        allocator.swa_attn_allocator.available_size.return_value = 8
        free_index = torch.tensor([0, 4, -1, 0, 4], dtype=torch.int64)

        with (
            mock.patch.object(allocator_base, "_DEBUG_MEMORY_POOL", True),
            self.assertRaisesRegex(RuntimeError, "unique, aligned"),
        ):
            allocator.free(free_index)

        allocator.swa_attn_allocator.free.assert_not_called()
        helper_source = inspect.getsource(allocator_base._validate_page_aligned_free)
        self.assertNotIn("free_index[free_index > 0]", helper_source)
        self.assertIn("torch.sort", helper_source)


if __name__ == "__main__":
    unittest.main()
