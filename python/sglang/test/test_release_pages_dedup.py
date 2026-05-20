"""Regression for release_pages deduplication in PagedTokenToKVPoolAllocator.

Bug: Three independent free paths (retraction, eviction, batch processing)
within a single overlap-dispatch iteration could append overlapping page
indices to release_pages without cross-call deduplication. Per-call
torch.unique(self.release_pages) was the old fix, but it degrades performance
as release_pages grows.

Fix (lazy dedup): remove the per-call torch.unique(self.release_pages) in
PagedTokenToKVPoolAllocator.free(). Instead rely on:
  1. Intra-call  dedup: torch.unique(free_index // page_size) – unchanged.
  2. Safety cap:    min(len(free_pages)+len(release_pages), self.size)
                    in available_size() prevents over-counting assertions.
  3. Lazy dedup:    torch.unique(self.free_pages) in merge_and_sort_free() –
                    runs once per iteration, deduplicates before allocation.

Tests cover:
  - Duplicate pages across multiple free() calls → correct after merge
  - available_size() never exceeds size even with duplicates in release_pages
  - free_group_begin/end path does not introduce duplicates
  - SWA tail allocator tight-pool scenario (assertion regression)
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE = 16


def _make_allocator(
    num_pages: int, *, need_sort: bool = True
) -> PagedTokenToKVPoolAllocator:
    """Create a PagedTokenToKVPoolAllocator with a trivial kvcache stub.

    The allocator's __init__ stores kvcache as self._kvcache without any
    isinstance checks, so any object is acceptable. Our tests only exercise
    free(), merge_and_sort_free(), available_size(), and alloc() — none of
    which touch self._kvcache.
    """
    tokens = num_pages * PAGE_SIZE
    # Use a plain object; the allocator never calls any kvcache method
    # in the code paths under test.
    allocator = PagedTokenToKVPoolAllocator(
        size=tokens,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=object(),
        need_sort=need_sort,
    )
    return allocator


class TestReleasePagesLazyDedup(CustomTestCase):
    """Verify that duplicate pages in release_pages are handled correctly."""

    def setUp(self):
        self.allocator = _make_allocator(num_pages=64)

    def test_duplicate_free_merge_dedup(self):
        """Multiple overlapping free() → merge_and_sort_free removes dups."""
        # Path 1 (retraction): free tokens [16:48] → pages 1,2
        self.allocator.free(torch.arange(16, 48, dtype=torch.int64))
        # Path 2 (eviction):  free tokens [32:80] → pages 2,3,4
        self.allocator.free(torch.arange(32, 80, dtype=torch.int64))
        # Path 3 (batch):    free tokens [48:64] → page 3 (again)
        self.allocator.free(torch.arange(48, 64, dtype=torch.int64))

        # available_size must stay within bounds despite duplicates
        capacity = self.allocator.size
        self.assertLessEqual(
            self.allocator.available_size(),
            capacity,
            "available_size() exceeded size despite min cap",
        )

        self.allocator.merge_and_sort_free()
        free_pages = self.allocator.free_pages

        # After merge, no duplicate pages
        unique_pages = torch.unique(free_pages)
        self.assertEqual(
            len(unique_pages),
            len(free_pages),
            f"free_pages has duplicates: {len(free_pages)} entries, "
            f"{len(unique_pages)} unique",
        )

    def test_available_size_capped_with_triple_dups(self):
        """available_size() stays <= self.size even with triple duplicates."""
        num_pages = 10
        allocator_size = self.allocator.size

        for _ in range(3):  # triple each page
            for i in range(num_pages):
                start = i * PAGE_SIZE
                end = start + PAGE_SIZE
                self.allocator.free(torch.arange(start, end, dtype=torch.int64))

        self.assertLessEqual(
            self.allocator.available_size(),
            allocator_size,
            "available_size() exceeded allocator size with triple duplicates",
        )

        # merge_and_sort should still produce unique free_pages
        self.allocator.merge_and_sort_free()
        unique_pages = torch.unique(self.allocator.free_pages)
        self.assertEqual(len(unique_pages), len(self.allocator.free_pages))

    def test_free_group_path_no_duplicates(self):
        """free_group_begin/end does not introduce page duplication."""
        self.allocator.merge_and_sort_free()  # start from clean state

        # Simulate Path 3 (batch processing) with overlapping pages:
        #   free_group_begin → accumulate frees → free_group_end
        self.allocator.free_group_begin()
        self.allocator.free(torch.arange(0, 32, dtype=torch.int64))  # pages 0,1
        self.allocator.free(torch.arange(16, 48, dtype=torch.int64))  # pages 1,2
        self.allocator.free_group_end()  # flushes into free() → release_pages

        capacity = self.allocator.size
        self.assertLessEqual(
            self.allocator.available_size(),
            capacity,
        )

        self.allocator.merge_and_sort_free()
        free_pages = self.allocator.free_pages
        unique_pages = torch.unique(free_pages)
        self.assertEqual(len(unique_pages), len(free_pages))

    def test_mixed_free_group_and_direct(self):
        """Mix of direct frees and free_group frees → no duplicates after merge."""
        # Direct free (Path 1): pages 3,4
        self.allocator.free(torch.arange(48, 80, dtype=torch.int64))

        # free_group free (Path 3): pages 4,5,6
        self.allocator.free_group_begin()
        self.allocator.free(torch.arange(64, 112, dtype=torch.int64))
        self.allocator.free_group_end()

        self.assertLessEqual(self.allocator.available_size(), self.allocator.size)

        self.allocator.merge_and_sort_free()
        unique_pages = torch.unique(self.allocator.free_pages)
        self.assertEqual(len(unique_pages), len(self.allocator.free_pages))

    def test_free_empty_tensor_noop(self):
        """free() with empty tensor is a no-op."""
        pages_before = len(self.allocator.free_pages)
        release_before = len(self.allocator.release_pages)

        self.allocator.free(torch.empty(0, dtype=torch.int64))

        self.assertEqual(len(self.allocator.free_pages), pages_before)
        self.assertEqual(len(self.allocator.release_pages), release_before)


class TestSWATailTightPool(CustomTestCase):
    """Regression: SWA assertion when pool is tight after tail preallocation.

    With commit d6d3d0f59 (alloc_extend_swa_tail), the SWA pool operates
    close to capacity. Even modest duplicate-page overcounting in
    release_pages pushes available_size() past size.
    """

    def setUp(self):
        # Full pool: 64 pages (generous)
        self.full = _make_allocator(num_pages=64)
        # SWA pool: 8 pages (tight — simulates tail-only allocation)
        self.swa = _make_allocator(num_pages=8)

    def test_swa_pool_stays_within_bounds(self):
        """SWA pool with duplicates must never exceed size."""
        swa_capacity = self.swa.size

        # Allocate half the SWA pool
        swa_alloc = self.swa.alloc(4 * PAGE_SIZE)
        self.assertIsNotNone(swa_alloc)

        # Simulate 3-path free scenario on the tight pool
        # Path 1: retraction
        self.swa.free(torch.arange(0, 2 * PAGE_SIZE, dtype=torch.int64))
        # Path 2: eviction (overlapping)
        self.swa.free(torch.arange(PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64))
        # Path 3: batch processing
        self.swa.free_group_begin()
        self.swa.free(torch.arange(2 * PAGE_SIZE, 4 * PAGE_SIZE, dtype=torch.int64))
        self.swa.free_group_end()

        self.assertLessEqual(
            self.swa.available_size(),
            swa_capacity,
            "SWA available_size() exceeded capacity — would have fired assertion",
        )

        self.swa.merge_and_sort_free()
        unique = torch.unique(self.swa.free_pages)
        self.assertEqual(len(unique), len(self.swa.free_pages))

    def test_full_pool_also_protected(self):
        """Full pool safety cap also works."""
        capacity = self.full.size

        for _ in range(2):
            for i in range(8):
                self.full.free(
                    torch.arange(i * PAGE_SIZE, (i + 1) * PAGE_SIZE, dtype=torch.int64)
                )

        self.assertLessEqual(self.full.available_size(), capacity)
        self.full.merge_and_sort_free()
        unique = torch.unique(self.full.free_pages)
        self.assertEqual(len(unique), len(self.full.free_pages))


if __name__ == "__main__":
    unittest.main()
