"""PagedTokenToKVPoolAllocator zeroes freed pages before handing them out again."""

import unittest

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_page_zeroer import KVPageZeroer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

PAGE_SIZE = 2
NUM_PAGES = 8


def _buffers():
    # Row 0 is the padded dummy page, as in the real pools.
    return [
        torch.ones((NUM_PAGES + 1, 16), dtype=torch.uint8),
        torch.ones((NUM_PAGES + 1, 8), dtype=torch.uint8),
    ]


def _allocator(zeroer, *, need_sort=False) -> PagedTokenToKVPoolAllocator:
    allocator = PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )
    allocator.enable_zero_on_free(zeroer)
    return allocator


def _page_ids(indices: torch.Tensor) -> torch.Tensor:
    return indices[::PAGE_SIZE] // PAGE_SIZE


class _Event:
    def __init__(self):
        self.done = False
        self.waited = False

    def query(self):
        return self.done

    def synchronize(self):
        self.waited = True
        self.done = True


class _PendingZeroer:
    num_rows = NUM_PAGES + 1

    def __init__(self):
        self.events = []

    def zero(self, page_ids):
        self.events.append(_Event())
        return self.events[-1]


class TestPagedAllocatorZeroOnFree(CustomTestCase):
    def test_enable_zeroes_every_page(self):
        buffers = _buffers()
        _allocator(KVPageZeroer(buffers))
        for buf in buffers:
            self.assertEqual(int(buf[1:].sum()), 0)
            self.assertTrue(torch.all(buf[0] == 1))

    def test_free_zeroes_only_freed_pages(self):
        for need_sort in (False, True):
            buffers = _buffers()
            allocator = _allocator(KVPageZeroer(buffers), need_sort=need_sort)
            indices = allocator.alloc(4 * PAGE_SIZE)
            for buf in buffers:
                buf.fill_(7)

            allocator.free(indices[: 2 * PAGE_SIZE])
            freed = _page_ids(indices[: 2 * PAGE_SIZE])
            kept = _page_ids(indices[2 * PAGE_SIZE :])
            for buf in buffers:
                self.assertEqual(int(buf[freed].sum()), 0)
                self.assertTrue(torch.all(buf[kept] == 7))
            self.assertEqual(allocator.available_size(), 6 * PAGE_SIZE)

            reused = allocator.alloc(6 * PAGE_SIZE)
            self.assertEqual(
                set(_page_ids(reused).tolist()),
                set(range(1, NUM_PAGES + 1)) - set(kept.tolist()),
            )

    def test_pages_wait_for_zeroing(self):
        zeroer = _PendingZeroer()
        allocator = _allocator(zeroer)
        zeroer.events[0].done = True
        indices = allocator.alloc(4 * PAGE_SIZE)
        allocator.free(indices)
        pending = zeroer.events[-1]
        self.assertEqual(allocator.available_size(), NUM_PAGES * PAGE_SIZE)
        self.assertEqual(len(allocator.get_all_free_pages()), NUM_PAGES)

        # Never-freed pages are enough: the freed ones stay in quarantine.
        first = allocator.alloc(4 * PAGE_SIZE)
        self.assertFalse(pending.waited)
        self.assertTrue(
            set(_page_ids(first).tolist()).isdisjoint(_page_ids(indices).tolist())
        )

        # Allocating beyond that waits for the zeroing to finish.
        second = allocator.alloc(4 * PAGE_SIZE)
        self.assertTrue(pending.waited)
        self.assertEqual(
            set(_page_ids(second).tolist()), set(_page_ids(indices).tolist())
        )
        self.assertEqual(allocator.num_zeroing_pages, 0)


if __name__ == "__main__":
    unittest.main()
