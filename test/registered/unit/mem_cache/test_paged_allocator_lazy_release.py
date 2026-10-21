"""need_sort allocators stage released pages and merge only under allocation pressure."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

PAGE_SIZE = 2
NUM_PAGES = 16


def _make_allocator(*, need_sort: bool) -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _page_ids(indices: torch.Tensor) -> torch.Tensor:
    return indices[::PAGE_SIZE] // PAGE_SIZE


class TestPagedAllocatorLazyRelease(CustomTestCase):
    def test_generations_are_opt_in_for_external_linkers(self):
        allocator = _make_allocator(need_sort=False)
        allocated = allocator.alloc(2)

        self.assertIsNone(allocator.allocation_generations(allocated))

    def test_allocation_generations_change_only_on_writable_reuse(self):
        allocator = _make_allocator(need_sort=False)
        allocator.enable_allocation_generations()
        allocated = allocator.alloc(4)
        first = allocator.allocation_generations(allocated)

        self.assertIsNotNone(first)
        self.assertTrue(allocator.allocation_generations_match(allocated, first))
        allocator.free(allocated)
        self.assertTrue(allocator.allocation_generations_match(allocated, first))

        reused = allocator.alloc(4)
        self.assertTrue(torch.equal(reused, allocated))
        second = allocator.allocation_generations(reused)
        self.assertIsNotNone(second)
        self.assertFalse(torch.equal(first, second))
        self.assertFalse(allocator.allocation_generations_match(reused, first))

    def test_clear_keeps_process_generation_monotonic(self):
        allocator = _make_allocator(need_sort=False)
        allocator.enable_allocation_generations()
        allocated = allocator.alloc(2)
        first = allocator.allocation_generations(allocated)

        allocator.clear()
        replacement = allocator.alloc(2)
        second = allocator.allocation_generations(replacement)

        self.assertTrue(torch.equal(replacement, allocated))
        self.assertGreater(int(second[0]), int(first[0]))

    def test_release_stages_until_free_pages_run_dry(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(24)
        free_pages_before = allocator.free_pages

        with patch.object(torch, "cat", wraps=torch.cat) as cat_mock:
            allocator.free(allocated[:2])
            allocator.free(allocated[4:6])
            self.assertIs(allocator.free_pages, free_pages_before)
            self.assertEqual(allocator.num_staged_pages, 2)
            self.assertEqual(allocator.available_size(), 12)
            cat_mock.assert_not_called()

            # free_pages still covers this: no merge.
            primary = allocator.alloc(8)
            self.assertTrue(torch.equal(_page_ids(primary), torch.arange(13, 17)))
            cat_mock.assert_not_called()

            # Pressure: one merge, staged pages come back sorted.
            reused = allocator.alloc(4)
            self.assertTrue(torch.equal(_page_ids(reused), torch.tensor([1, 3])))
            self.assertEqual(cat_mock.call_count, 1)

        self.assertEqual(allocator.num_staged_pages, 0)
        self.assertEqual(allocator.staged_pages, [])

    def test_clear_drops_staged_pages(self):
        allocator = _make_allocator(need_sort=True)
        allocator.free(allocator.alloc(4))

        allocator.clear()
        self.assertEqual(allocator.available_size(), allocator.size)
        self.assertEqual(allocator.staged_pages, [])


if __name__ == "__main__":
    unittest.main()
