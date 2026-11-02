"""Regression for the paged SWA page return under the per-request ring.

Both free paths returned SWA pages to the paged swa_attn_allocator
unconditionally. Under the per-request ring that allocator is vestigial and its
slots are owned by the req slot rather than lent per free, so the return
over-credited available_size() past size and tripped the assert at the end of
free_group_end, killing the scheduler on the first decode that frees.

The mapping clear in _free_swa_pages must still run in ring mode: without it,
translate_loc_from_full_to_swa reads stale peer indices and the failure becomes
wrong KV instead of a crash.
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE_SIZE = 8
POOL_PAGES = 4


class _CountingPagedAllocator:
    """Paged allocator that really tracks credit, so over-crediting is visible.

    A MagicMock would only record the call; the bug is that the call moves
    available_size() past size, which is what the production assert checks.
    """

    def __init__(self, *, size: int):
        self.size = size
        self.debug_mode = False
        self._free_tokens = 0  # fully allocated to start

    def available_size(self) -> int:
        return self._free_tokens

    def free_page_ids(self, page_ids: torch.Tensor) -> None:
        self._free_tokens += int(page_ids.numel()) * PAGE_SIZE

    def free_group_end(self) -> None:
        """The full-side allocator defers its own frees; nothing to settle here."""


def _make_self(*, swa_req_ring: bool, page_size: int = PAGE_SIZE):
    """Build a real instance without __init__; free_group_end calls zero-arg
    super(), which requires an instance of the class rather than a stub."""
    alloc = object.__new__(SWATokenToKVPoolAllocator)

    alloc.page_size = page_size
    alloc._swa_req_ring = swa_req_ring
    alloc.free_group = None
    alloc.swa_free_group = []
    alloc.swa_page_ids_group = []

    mapping = torch.zeros(64, dtype=torch.int64)
    # Peer pages for the rows under test; page 2 of the paged SWA pool.
    mapping[0:page_size] = torch.arange(2 * page_size, 3 * page_size, dtype=torch.int64)
    alloc.full_to_swa_index_mapping = mapping

    alloc.swa_attn_allocator = _CountingPagedAllocator(size=page_size * POOL_PAGES)
    alloc.full_attn_allocator = _CountingPagedAllocator(size=page_size * POOL_PAGES)
    return alloc


class TestSWARingPageReturn(CustomTestCase):
    def test_group_drain_keeps_paged_credit_within_size_in_ring_mode(self):
        """Pre-fix this over-credits and the production assert raises."""
        alloc = _make_self(swa_req_ring=True)
        alloc.swa_page_ids_group = [torch.arange(POOL_PAGES + 4, dtype=torch.int64)]

        alloc.free_group_end()

        self.assertEqual(alloc.swa_attn_allocator.available_size(), 0)
        self.assertLessEqual(
            alloc.swa_attn_allocator.available_size(),
            alloc.swa_attn_allocator.size,
        )
        # Pile still drains, or it leaks into the next group.
        self.assertEqual(alloc.swa_page_ids_group, [])

    def test_group_drain_returns_pages_without_ring(self):
        alloc = _make_self(swa_req_ring=False)
        alloc.swa_page_ids_group = [torch.arange(2, dtype=torch.int64)]

        alloc.free_group_end()

        self.assertEqual(alloc.swa_attn_allocator.available_size(), 2 * PAGE_SIZE)
        self.assertEqual(alloc.swa_page_ids_group, [])

    def test_direct_free_keeps_paged_credit_at_zero_in_ring_mode(self):
        alloc = _make_self(swa_req_ring=True)
        free_index = torch.arange(0, PAGE_SIZE, dtype=torch.int64)

        alloc._free_swa_pages(free_index, start_pos=0)

        self.assertEqual(alloc.swa_attn_allocator.available_size(), 0)
        # Nothing deferred either: the ring must not queue what it never returns.
        self.assertEqual(alloc.swa_page_ids_group, [])

    def test_direct_free_clears_mapping_in_ring_mode(self):
        """Guards the early-return trap: skipping the clear leaves stale peers."""
        alloc = _make_self(swa_req_ring=True)
        free_index = torch.arange(0, PAGE_SIZE, dtype=torch.int64)
        self.assertTrue(bool((alloc.full_to_swa_index_mapping[free_index] > 0).any()))

        alloc._free_swa_pages(free_index, start_pos=0)

        self.assertTrue(
            bool((alloc.full_to_swa_index_mapping[free_index] == 0).all()),
            "ring mode must still clear full_to_swa before returning",
        )

    def test_direct_free_returns_pages_without_ring(self):
        alloc = _make_self(swa_req_ring=False)
        free_index = torch.arange(0, PAGE_SIZE, dtype=torch.int64)

        alloc._free_swa_pages(free_index, start_pos=0)

        self.assertEqual(alloc.swa_attn_allocator.available_size(), PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
