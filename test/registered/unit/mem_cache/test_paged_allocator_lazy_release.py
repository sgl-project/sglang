"""Regression tests for deferred page release in the paged allocator."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

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
    return indices.reshape(-1, PAGE_SIZE)[:, 0] // PAGE_SIZE


class TestPagedAllocatorLazyRelease(CustomTestCase):
    def test_pd_release_defers_concat_until_primary_pages_are_exhausted(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(24)
        free_pages_before = allocator.free_pages

        with patch.object(torch, "cat", wraps=torch.cat) as cat_mock:
            allocator.free(allocated[:2])
            allocator.free(allocated[4:6])

            self.assertIs(allocator.free_pages, free_pages_before)
            self.assertEqual(allocator.release_pages.numel(), 0)
            self.assertEqual(len(allocator.pending_release_page_chunks), 2)
            self.assertEqual(allocator.num_pending_release_pages, 2)
            self.assertEqual(allocator.available_size(), 12)
            cat_mock.assert_not_called()

            primary = allocator.alloc(8)
            self.assertTrue(torch.equal(_page_ids(primary), torch.arange(13, 17)))
            cat_mock.assert_not_called()

            reused = allocator.alloc(4)
            self.assertTrue(torch.equal(_page_ids(reused), torch.tensor([1, 3])))
            self.assertEqual(cat_mock.call_count, 1)

        self.assertEqual(allocator.num_pending_release_pages, 0)
        self.assertEqual(allocator.pending_release_page_chunks, [])

    def test_available_size_debug_view_and_clear_include_pending_chunks(self):
        allocator = _make_allocator(need_sort=True)
        allocated = allocator.alloc(NUM_PAGES * PAGE_SIZE)
        allocator.free(allocated[:2])
        allocator.free(allocated[2:4])

        self.assertEqual(allocator.available_size(), 4)
        self.assertEqual(allocator.get_all_free_pages().numel(), 2)

        allocator.clear()
        self.assertEqual(allocator.available_size(), allocator.size)
        self.assertEqual(allocator.num_pending_release_pages, 0)
        self.assertEqual(allocator.pending_release_page_chunks, [])

    def test_non_pd_allocator_preserves_eager_prepend_order(self):
        allocator = _make_allocator(need_sort=False)
        allocated = allocator.alloc(PAGE_SIZE)
        allocator.free(allocated)

        self.assertEqual(allocator.free_pages[0].item(), 1)
        self.assertEqual(allocator.num_pending_release_pages, 0)
        self.assertEqual(allocator.pending_release_page_chunks, [])


if __name__ == "__main__":
    unittest.main()
