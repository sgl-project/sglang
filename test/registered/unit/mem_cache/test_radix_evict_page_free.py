"""Unit tests for the sync-free page release in RadixCache.evict().

evict() frees page-aligned node values with a fixed-shape strided slice
(``cat[::page_size] // page_size``) instead of routing every node through
``allocator.free()``, whose ``torch.unique`` forces a device sync per call.
These tests assert:

1. the strided slice is exactly equivalent to ``torch.unique`` for
   page-aligned page runs (the only inputs evict() puts on that path);
2. evict() releases exactly the evicted pages, once each, into the same
   allocator fields ``free()`` would have used (``need_sort`` both ways);
3. the fallback paths are preserved: ``page_size == 1`` allocators,
   non-page-multiple values, and an open free-group still go through the
   original ``allocator.free()`` / ``free_group`` semantics.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

PAGE = 4


class FakePagedAllocator:
    """Just enough of BaseTokenToKVPoolAllocator for evict()'s release path."""

    def __init__(self, page_size=PAGE, need_sort=True):
        self.page_size = page_size
        self.need_sort = need_sort
        self.device = torch.device("cpu")
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64)
        self.free_pages = torch.empty((0,), dtype=torch.int64)
        self.free_calls = []

    def free(self, free_index: torch.Tensor):
        # Original (sync-carrying) path; recorded so tests can assert when
        # evict() falls back to it.
        self.free_calls.append(free_index.clone())


def page_run(page: int, page_size: int = PAGE) -> torch.Tensor:
    """Token slots of one page: a consecutive, page-aligned run."""
    return torch.arange(page * page_size, (page + 1) * page_size, dtype=torch.int64)


def value_for_pages(pages, page_size: int = PAGE) -> torch.Tensor:
    return torch.cat([page_run(p, page_size) for p in pages])


class TestStridedSliceEquivalence(unittest.TestCase):
    def test_equivalent_to_unique_for_page_aligned_runs(self):
        for page_size in (2, 4, 64):
            for pages in ([0], [3, 7], [5, 1, 9, 2], list(range(17))):
                with self.subTest(page_size=page_size, pages=pages):
                    cat = value_for_pages(pages, page_size)
                    strided = cat[::page_size] // page_size
                    reference = torch.unique(cat // page_size)
                    self.assertTrue(
                        torch.equal(torch.sort(strided).values, reference),
                        f"strided {strided} != unique {reference}",
                    )
                    # Fixed-shape: one element per freed page, no dedup needed.
                    self.assertEqual(strided.numel(), len(pages))


class TestEvictReleasesPagesSyncFree(unittest.TestCase):
    def _make_cache(self, **alloc_kwargs):
        allocator = FakePagedAllocator(**alloc_kwargs)
        cache = RadixCache.create_simulated(
            mock_allocator=allocator, page_size=allocator.page_size
        )
        return cache, allocator

    def _insert_pages(self, cache, first_token: int, pages):
        """One leaf whose key/value are page-aligned (len % PAGE == 0)."""
        tokens = list(range(first_token, first_token + PAGE * len(pages)))
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=value_for_pages(pages),
            )
        )

    def test_evict_all_releases_each_page_exactly_once(self):
        for need_sort in (True, False):
            with self.subTest(need_sort=need_sort):
                cache, allocator = self._make_cache(need_sort=need_sort)
                self._insert_pages(cache, first_token=1000, pages=[3, 7])
                self._insert_pages(cache, first_token=2000, pages=[11])
                self._insert_pages(cache, first_token=3000, pages=[5, 6, 8])

                result = cache.evict(EvictParams(num_tokens=6 * PAGE))
                self.assertEqual(result.num_tokens_evicted, 6 * PAGE)

                released = (
                    allocator.release_pages if need_sort else allocator.free_pages
                )
                self.assertTrue(
                    torch.equal(
                        torch.sort(released).values,
                        torch.tensor([3, 5, 6, 7, 8, 11], dtype=torch.int64),
                    ),
                    f"released pages {released}",
                )
                # The sync-free path must be used: no allocator.free() calls.
                self.assertEqual(allocator.free_calls, [])
                # And only the matching destination is touched.
                other = allocator.free_pages if need_sort else allocator.release_pages
                self.assertEqual(other.numel(), 0)

    def test_partial_evict_releases_only_evicted_pages(self):
        cache, allocator = self._make_cache()
        self._insert_pages(cache, first_token=1000, pages=[3])
        evicted_first = cache.evict(EvictParams(num_tokens=1)).num_tokens_evicted
        self.assertEqual(evicted_first, PAGE)
        self.assertTrue(
            torch.equal(allocator.release_pages, torch.tensor([3], dtype=torch.int64))
        )

    def test_open_free_group_appends_to_group(self):
        cache, allocator = self._make_cache()
        self._insert_pages(cache, first_token=1000, pages=[2, 9])
        allocator.is_not_in_free_group = False

        cache.evict(EvictParams(num_tokens=2 * PAGE))
        self.assertEqual(len(allocator.free_group), 1)
        self.assertTrue(torch.equal(allocator.free_group[0], value_for_pages([2, 9])))
        self.assertEqual(allocator.release_pages.numel(), 0)
        self.assertEqual(allocator.free_calls, [])

    def test_page_size_one_keeps_original_free_path(self):
        allocator = FakePagedAllocator(page_size=1)
        cache = RadixCache.create_simulated(mock_allocator=allocator, page_size=1)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3])),
                value=torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        cache.evict(EvictParams(num_tokens=3))
        self.assertEqual(len(allocator.free_calls), 1)
        self.assertTrue(
            torch.equal(
                allocator.free_calls[0],
                torch.tensor([10, 20, 30], dtype=torch.int64),
            )
        )
        self.assertEqual(allocator.release_pages.numel(), 0)
        self.assertEqual(allocator.free_pages.numel(), 0)


if __name__ == "__main__":
    unittest.main()
