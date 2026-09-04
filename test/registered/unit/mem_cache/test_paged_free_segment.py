"""free_segment / free_segments vs the torch.unique reference: page-aligned
starts over every tail alignment, the page-disjoint contract, and free-group
deferral. See PagedTokenToKVPoolAllocator.free_segment for why unique is avoided.

    python -m pytest test/registered/unit/mem_cache/test_paged_free_segment.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import _release_overallocated_kv_indices
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE_SIZE = 4
NUM_PAGES = 64


def _make_allocator(need_sort=False):
    return PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _make_kv_row(alloc, num_tokens):
    # Page-aligned allocation, then trim to num_tokens: mirrors a request's
    # req_to_token row (token position t lives at page*page_size + t%page_size).
    num_pages = -(num_tokens // -PAGE_SIZE)
    indices = alloc.alloc(num_pages * PAGE_SIZE)
    return indices[:num_tokens]


class TestFreeSegment(unittest.TestCase):
    def test_matches_unique_over_tail_alignments(self):
        for num_tokens in (1, PAGE_SIZE, PAGE_SIZE + 1, 3 * PAGE_SIZE - 1):
            for start in range(0, num_tokens, PAGE_SIZE):
                for end in range(start + 1, num_tokens + 1):
                    alloc = _make_allocator()
                    row = _make_kv_row(alloc, num_tokens)
                    expected = torch.unique(row[start:end] // PAGE_SIZE)
                    before = len(alloc.free_pages)
                    alloc.free_segment(row[start:end], start_pos=start)
                    freed = alloc.free_pages[: len(alloc.free_pages) - before]
                    self.assertTrue(
                        torch.equal(torch.sort(freed)[0], expected),
                        f"{num_tokens=} {start=} {end=}",
                    )

    def test_empty_segment_is_noop(self):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, PAGE_SIZE)
        before = len(alloc.free_pages)
        alloc.free_segment(row[:0], start_pos=0)
        self.assertEqual(len(alloc.free_pages), before)

    def test_unaligned_start_is_rejected(self):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, 2 * PAGE_SIZE)
        for start in (1, PAGE_SIZE - 1, PAGE_SIZE + 1):
            with self.assertRaises(AssertionError):
                alloc.free_segment(row[start:], start_pos=start)

    def test_need_sort_defers_released_pages(self):
        alloc = _make_allocator(need_sort=True)
        row = _make_kv_row(alloc, 2 * PAGE_SIZE)
        alloc.free_segment(row, start_pos=0)
        self.assertEqual(len(alloc.staged_pages), 1)
        self.assertEqual(alloc.num_staged_pages, 2)

    def test_group_defers_until_group_end(self):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, 2 * PAGE_SIZE)
        before = len(alloc.free_pages)
        alloc.free_group_begin()
        alloc.free_segment(row, start_pos=0)
        self.assertEqual(len(alloc.free_pages), before)
        alloc.free_group_end()
        self.assertEqual(len(alloc.free_pages), before + 2)

    def test_group_owns_deferred_page_representatives(self):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, 2 * PAGE_SIZE)
        expected_pages = torch.unique(row // PAGE_SIZE)

        alloc.free_group_begin()
        alloc.free_segment(row, start_pos=0)
        row.zero_()
        alloc.free_group_end()

        freed_pages = alloc.free_pages[: expected_pages.numel()]
        self.assertTrue(torch.equal(torch.sort(freed_pages)[0], expected_pages))

    def test_group_end_debug_assert_catches_cross_call_double_free(self):
        # legacy free() + free_segment() on the same page in one group must
        # trip free_group_end's debug assert
        alloc = _make_allocator()
        alloc.debug_mode = True
        row = _make_kv_row(alloc, PAGE_SIZE)
        alloc.free_group_begin()
        alloc.free(row)
        alloc.free_segment(row, start_pos=0)
        with self.assertRaises(AssertionError):
            alloc.free_group_end()

    def test_group_end_debug_assert_covers_staged_releases(self):
        # need_sort stages frees in chunks; the duplicate check must
        # not go vacuous there (PD disaggregation runs with need_sort=True).
        alloc = _make_allocator(need_sort=True)
        alloc.debug_mode = True
        row = _make_kv_row(alloc, PAGE_SIZE)
        alloc.free_group_begin()
        alloc.free(row)
        alloc.free_segment(row, start_pos=0)
        with self.assertRaises(AssertionError):
            alloc.free_group_end()

    def test_overallocated_tail_uses_allocator_page_size_under_dcp(self):
        # Scaled-down DCP example: the configured logical page is 1 while the
        # allocator page is widened to 4. cache_finished_req has already freed
        # the committed tail [4, 5), so over-allocation cleanup for [5, 7)
        # must not release the same physical page again.
        alloc = _make_allocator()
        alloc.debug_mode = True
        row = _make_kv_row(alloc, 2 * PAGE_SIZE)
        tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=alloc,
            req_to_token_pool=SimpleNamespace(req_to_token=row.unsqueeze(0)),
        )
        req = SimpleNamespace(kv=ReqKvInfo(req_pool_idx=0))

        before = len(alloc.free_pages)
        alloc.free_group_begin()
        alloc.free_segment(row[PAGE_SIZE : PAGE_SIZE + 1], start_pos=PAGE_SIZE)
        with (
            patch(
                "sglang.srt.mem_cache.common.get_spec",
                return_value=SimpleNamespace(speculative_algorithm="DSPARK"),
            ),
            patch(
                "sglang.srt.mem_cache.common.get_serving",
                return_value=SimpleNamespace(strip_thinking_cache=False),
            ),
        ):
            _release_overallocated_kv_indices(
                req,
                start_p=PAGE_SIZE + 1,
                end_p=2 * PAGE_SIZE - 1,
                tree_cache=tree_cache,
            )
        alloc.free_group_end()

        self.assertEqual(len(alloc.free_pages), before + 1)


class TestFreeSegments(unittest.TestCase):
    def _freed_by_segments(self, num_tokens, spans):
        alloc = _make_allocator()
        row = _make_kv_row(alloc, num_tokens)
        before = len(alloc.free_pages)
        alloc.free_segments([(row[a:b], a) for a, b in spans])
        freed = alloc.free_pages[: len(alloc.free_pages) - before]
        reference = torch.unique(torch.cat([row[a:b] for a, b in spans]) // PAGE_SIZE)
        return freed, reference

    def test_partial_tail_then_next_page(self):
        # [0, 5) releases page 1 whole; [8, 11) starts on page 2.
        freed, reference = self._freed_by_segments(11, [(0, 5), (8, 11)])
        self.assertTrue(torch.equal(torch.sort(freed)[0], reference))

    def test_page_aligned_segments(self):
        freed, reference = self._freed_by_segments(
            3 * PAGE_SIZE, [(0, PAGE_SIZE), (PAGE_SIZE, 3 * PAGE_SIZE)]
        )
        self.assertTrue(torch.equal(torch.sort(freed)[0], reference))

    def test_segments_sharing_a_page_are_rejected(self):
        for spans in ([(0, 5), (5, 8)], [(0, 5), (7, 11)], [(0, 6), (4, 11)]):
            with self.assertRaises(AssertionError):
                self._freed_by_segments(11, spans)


class _RecordingBaseAllocator(BaseTokenToKVPoolAllocator):
    """Base-fallback allocator: free_segment inherits the default (ignore
    start_pos, call free()), free() records what it received."""

    def __init__(self):
        super().__init__(
            size=NUM_PAGES * PAGE_SIZE,
            page_size=PAGE_SIZE,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.freed = []

    def alloc(self, need_size: int):
        raise NotImplementedError

    def clear(self):
        pass

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index)


class TestBaseFallbackFreeSegments(unittest.TestCase):
    def test_fallback_forwards_page_disjoint_segments(self):
        # base fallback: each segment reaches free() as-is, no cross-segment dedup
        alloc = _RecordingBaseAllocator()
        row = torch.arange(11)  # position i lives on page i // PAGE_SIZE
        alloc.free_segments([(row[0:6], 0), (row[8:11], 8)])
        per_call_pages = [set((t // PAGE_SIZE).tolist()) for t in alloc.freed]
        self.assertEqual(per_call_pages, [{0, 1}, {2}])

    def test_fallback_rejects_shared_page_and_unaligned_start(self):
        alloc = _RecordingBaseAllocator()
        row = torch.arange(11)
        with self.assertRaises(AssertionError):
            alloc.free_segments([(row[0:6], 0), (row[6:11], 6)])
        with self.assertRaises(AssertionError):
            alloc.free_segment(row[1:], start_pos=1)
        self.assertEqual(len(alloc.freed), 1)
        self.assertTrue(torch.equal(alloc.freed[0], row[0:6]))


if __name__ == "__main__":
    unittest.main()
