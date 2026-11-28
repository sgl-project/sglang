"""Lazy free-page staging for PagedTokenToKVPoolAllocator, TokenToKVPoolAllocator, and RadixCache.evict."""

import os
import random
import unittest
from array import array
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocator.base import _MAX_STAGED_CHUNKS
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE_SIZE = 2
NUM_PAGES = 16


def _make_paged_allocator(
    *, need_sort: bool, num_pages: int = NUM_PAGES, page_size: int = PAGE_SIZE
) -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        size=num_pages * page_size,
        page_size=page_size,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _make_token_allocator(*, need_sort: bool, size: int = 32) -> TokenToKVPoolAllocator:
    return TokenToKVPoolAllocator(
        size=size,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _page_ids(indices: torch.Tensor, page_size: int = PAGE_SIZE) -> torch.Tensor:
    return indices[::page_size] // page_size


class TestPagedAllocatorLazyRelease(CustomTestCase):
    def test_release_stages_until_free_pages_run_dry(self):
        for need_sort in (True, False):
            with self.subTest(need_sort=need_sort):
                allocator = _make_paged_allocator(need_sort=need_sort)
                allocated = allocator.alloc(24)
                free_pages_before = allocator.free_pages

                with (
                    patch.object(torch, "cat", wraps=torch.cat) as cat_mock,
                    patch.object(torch, "sort", wraps=torch.sort) as sort_mock,
                ):
                    allocator.free(allocated[4:6])
                    allocator.free(allocated[:2])
                    self.assertIs(allocator.free_pages, free_pages_before)
                    self.assertEqual(allocator.num_staged_pages, 2)
                    self.assertEqual(allocator.available_size(), 12)
                    cat_mock.assert_not_called()
                    sort_mock.assert_not_called()

                    # free_pages still covers this: no merge.
                    primary = allocator.alloc(8)
                    self.assertTrue(
                        torch.equal(_page_ids(primary), torch.arange(13, 17))
                    )
                    cat_mock.assert_not_called()

                    # Pressure: one merge; sorted iff need_sort.
                    reused = allocator.alloc(4)
                    expected = (
                        torch.tensor([1, 3]) if need_sort else torch.tensor([3, 1])
                    )
                    self.assertTrue(torch.equal(_page_ids(reused), expected))
                    self.assertEqual(cat_mock.call_count, 1)
                    self.assertEqual(sort_mock.call_count, 1 if need_sort else 0)

                self.assertEqual(allocator.num_staged_pages, 0)
                self.assertEqual(allocator.staged_pages, [])

    def test_clear_drops_staged_pages(self):
        for need_sort in (True, False):
            with self.subTest(need_sort=need_sort):
                allocator = _make_paged_allocator(need_sort=need_sort)
                allocator.free(allocator.alloc(4))

                allocator.clear()
                self.assertEqual(allocator.available_size(), allocator.size)
                self.assertEqual(allocator.staged_pages, [])
                self.assertEqual(allocator.num_staged_pages, 0)

    def test_reuse_guard_waits_only_when_staged_pages_are_needed(self):
        for need_sort in (True, False):
            with self.subTest(need_sort=need_sort):
                allocator = _make_paged_allocator(need_sort=need_sort)
                allocated = allocator.alloc(24)
                event = Mock()
                allocator.set_reuse_guard_event(event)
                allocator.free(allocated[:2])
                # Fresh pages are safe to hand to an asynchronous PD transfer.
                allocator.alloc(8)
                event.synchronize.assert_not_called()

                reused = allocator.alloc(2)
                self.assertTrue(torch.equal(_page_ids(reused), torch.tensor([1])))
                event.synchronize.assert_called_once_with()
                allocator.merge_and_sort_free()
                event.synchronize.assert_called_once_with()

    def test_bounded_staged_compaction_conserves_all_pages(self):
        num_pages = _MAX_STAGED_CHUNKS + 64
        allocator = _make_paged_allocator(need_sort=False, num_pages=num_pages)
        allocator.debug_mode = True
        allocated = allocator.alloc(num_pages * PAGE_SIZE)
        for i in range(num_pages):
            allocator.free_segment(
                allocated[i * PAGE_SIZE : (i + 1) * PAGE_SIZE], start_pos=0
            )
            self.assertLess(len(allocator.staged_pages), _MAX_STAGED_CHUNKS)
        self.assertEqual(allocator.available_size(), num_pages * PAGE_SIZE)
        regained = allocator.alloc(num_pages * PAGE_SIZE)
        self.assertTrue(
            torch.equal(
                torch.sort(_page_ids(regained))[0],
                torch.arange(1, num_pages + 1),
            )
        )

    def test_randomized_alloc_free_debug_mode_invariants(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                rng = random.Random(42 + int(need_sort))
                num_pages = 64
                allocator = _make_paged_allocator(
                    need_sort=need_sort, num_pages=num_pages, page_size=4
                )
                allocator.debug_mode = True
                live: list[torch.Tensor] = []
                for _ in range(300):
                    if not live or (rng.random() < 0.55 and allocator.available_size()):
                        max_p = min(8, allocator.available_size() // 4)
                        if max_p > 0:
                            k = rng.randint(1, max_p)
                            seg = allocator.alloc(k * 4)
                            self.assertIsNotNone(seg)
                            live.append(seg)
                    else:
                        idx = rng.randrange(len(live))
                        seg = live.pop(idx)
                        if rng.random() < 0.5:
                            allocator.free_segment(seg, start_pos=0)
                        else:
                            allocator.free(seg)
                    live_tokens = sum(t.numel() for t in live)
                    self.assertEqual(
                        allocator.available_size() + live_tokens, num_pages * 4
                    )
                for seg in live:
                    allocator.free_segment(seg, start_pos=0)
                self.assertEqual(allocator.available_size(), num_pages * 4)
                all_free = allocator.get_all_free_pages()
                self.assertTrue(
                    torch.equal(torch.sort(all_free)[0], torch.arange(1, num_pages + 1))
                )


class TestTokenAllocatorLazyRelease(CustomTestCase):
    def test_token_allocator_stages_without_cat_and_preserves_order(self):
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = _make_token_allocator(need_sort=need_sort, size=16)
                a = allocator.alloc(12)
                free_before = allocator.free_pages

                with (
                    patch.object(torch, "cat", wraps=torch.cat) as cat_mock,
                    patch.object(torch, "sort", wraps=torch.sort) as sort_mock,
                ):
                    allocator.free(a[4:6])
                    allocator.free(a[:2])
                    self.assertIs(allocator.free_pages, free_before)
                    self.assertEqual(allocator.num_staged_pages, 4)
                    self.assertEqual(allocator.available_size(), 8)
                    cat_mock.assert_not_called()
                    sort_mock.assert_not_called()

                    # Remaining fresh pages served without merging.
                    fresh = allocator.alloc(4)
                    self.assertTrue(torch.equal(fresh, torch.arange(13, 17)))
                    cat_mock.assert_not_called()

                    # Next alloc merges staged pages once (sorted iff need_sort).
                    reused = allocator.alloc(4)
                    expected = (
                        torch.tensor([1, 2, 5, 6])
                        if need_sort
                        else torch.tensor([5, 6, 1, 2])
                    )
                    self.assertTrue(torch.equal(reused, expected))
                    self.assertEqual(cat_mock.call_count, 1)
                    self.assertEqual(sort_mock.call_count, 1 if need_sort else 0)

    def test_token_allocator_detaches_mutable_caller_view_and_casts_int32(self):
        allocator = _make_token_allocator(need_sort=False, size=8)
        row = allocator.alloc(8).to(torch.int32)
        expected = row[:4].to(torch.int64).clone()

        allocator.free(row[:4])
        # Overwriting the caller buffer must not corrupt staged pages.
        row.zero_()

        reused = allocator.alloc(4)
        self.assertEqual(reused.dtype, torch.int64)
        self.assertTrue(torch.equal(reused, expected))


class TestRadixCacheEvictBatchesFrees(CustomTestCase):
    def test_evict_coalesces_into_single_free_segment_call(self):
        for allocator, page_size in (
            (_make_paged_allocator(need_sort=False, num_pages=64, page_size=4), 4),
            (_make_token_allocator(need_sort=False, size=256), 1),
        ):
            with self.subTest(allocator=type(allocator).__name__):
                cache = RadixCache.create_simulated(
                    mock_allocator=allocator, page_size=page_size
                )
                for seq_id in range(20):
                    slots = allocator.alloc(4)
                    cache.insert(
                        InsertParams(
                            key=RadixKey(
                                array(
                                    "q",
                                    [seq_id * 10 + j for j in range(4)],
                                )
                            ),
                            value=slots,
                        )
                    )
                avail_before = allocator.available_size()
                free_pages_before = allocator.free_pages
                with (
                    patch.object(torch, "cat", wraps=torch.cat) as cat_mock,
                    patch.object(
                        allocator,
                        "free_segment",
                        wraps=allocator.free_segment,
                    ) as free_seg_mock,
                ):
                    res = cache.evict(EvictParams(num_tokens=40))

                self.assertEqual(res.num_tokens_evicted, 40)
                self.assertEqual(allocator.available_size(), avail_before + 40)
                # 10 evicted leaves collapse into one free_segment call and one
                # cat over only the evicted leaves (free_pages untouched).
                free_seg_mock.assert_called_once()
                self.assertEqual(cat_mock.call_count, 1)
                self.assertIs(allocator.free_pages, free_pages_before)

    def test_evict_works_inside_open_free_group(self):
        allocator = _make_paged_allocator(need_sort=False, num_pages=32, page_size=4)
        cache = RadixCache.create_simulated(mock_allocator=allocator, page_size=4)
        for seq_id in range(4):
            slots = allocator.alloc(4)
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", [seq_id * 10 + j for j in range(4)])),
                    value=slots,
                )
            )
        avail_before = allocator.available_size()
        allocator.free_group_begin()
        res = cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(res.num_tokens_evicted, 8)
        self.assertIsNotNone(allocator.free_group)
        allocator.free_group_end()
        self.assertEqual(allocator.available_size(), avail_before + 8)


class TestPagedAllocatorDebugMode(CustomTestCase):
    def test_debug_mode_accepts_every_spelling_of_the_declared_bool(self):
        """SGLANG_DEBUG_MEMORY_POOL is one switch shared with the KV pool: a
        value that turns on the pool's checks must turn on the allocator's."""
        for value, expected in (("yes", True), ("1", True), ("no", False)):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": value}),
            ):
                self.assertEqual(
                    _make_paged_allocator(need_sort=False).debug_mode, expected
                )


if __name__ == "__main__":
    unittest.main()
