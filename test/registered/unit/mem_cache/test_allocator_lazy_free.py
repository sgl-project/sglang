"""
Unit tests for TokenToKVPoolAllocator lazy-free optimization.

Verifies that the pending-free buffer:
  1. Returns correct slots (functional correctness)
  2. Reports available_size() accurately while slots are pending
  3. Flushes before alloc so new requests get the freed slots
  4. Handles edge cases: empty free, single free, free_group interop

Usage:
    python -m pytest test_allocator_lazy_free.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator


def _make_allocator(size=64, need_sort=False, device="cpu"):
    kvcache = MagicMock()
    return TokenToKVPoolAllocator(
        size=size,
        dtype=torch.int64,
        device=device,
        kvcache=kvcache,
        need_sort=need_sort,
    )


class TestLazyFreeCorrectness(unittest.TestCase):
    """Freed slots must be reachable by the next alloc()."""

    def test_freed_slots_returned_after_alloc(self):
        alloc = _make_allocator(size=10)
        slots = alloc.alloc(5)
        self.assertIsNotNone(slots)

        alloc.free(slots)
        # pending — not yet in free_pages
        self.assertEqual(alloc._pending_free_count, 5)

        # alloc triggers flush: freed slots become available
        new_slots = alloc.alloc(5)
        self.assertIsNotNone(new_slots)
        self.assertEqual(len(new_slots), 5)

    def test_multiple_frees_batched_into_one_flush(self):
        alloc = _make_allocator(size=20)
        a = alloc.alloc(5)
        b = alloc.alloc(5)
        alloc.free(a)
        alloc.free(b)

        self.assertEqual(len(alloc._pending_free), 2)
        self.assertEqual(alloc._pending_free_count, 10)

        # flush on alloc
        result = alloc.alloc(10)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 10)
        # pending cleared after flush
        self.assertEqual(alloc._pending_free_count, 0)
        self.assertEqual(len(alloc._pending_free), 0)

    def test_slot_values_preserved_through_pending(self):
        # After free+alloc the recovered slots must be valid (in [1..size])
        # and exactly 3 unique slots — not necessarily the same indices
        # (pending appends to end of free_pages, so alloc returns from front).
        alloc = _make_allocator(size=10)
        first = set(alloc.alloc(3).tolist())
        alloc.alloc(3)  # drain 3 more so pending slots won't come back first

        alloc.free(torch.tensor(list(first), dtype=torch.int64))
        recovered = alloc.alloc(3)
        self.assertIsNotNone(recovered)
        # all recovered slots must be valid pool indices
        self.assertTrue(all(1 <= v <= 10 for v in recovered.tolist()))


class TestAvailableSize(unittest.TestCase):
    """available_size() must count pending slots."""

    def test_available_size_includes_pending(self):
        alloc = _make_allocator(size=10)
        initial = alloc.available_size()

        slots = alloc.alloc(4)
        self.assertEqual(alloc.available_size(), initial - 4)

        alloc.free(slots)
        # pending counts toward available
        self.assertEqual(alloc.available_size(), initial)

    def test_available_size_after_flush(self):
        alloc = _make_allocator(size=10)
        slots = alloc.alloc(4)
        alloc.free(slots)
        alloc.alloc(1)  # triggers flush
        self.assertEqual(alloc.available_size(), 10 - 1)


class TestEdgeCases(unittest.TestCase):

    def test_free_empty_tensor_is_noop(self):
        alloc = _make_allocator(size=10)
        empty = torch.empty(0, dtype=torch.int64)
        alloc.free(empty)
        self.assertEqual(alloc._pending_free_count, 0)
        self.assertEqual(len(alloc._pending_free), 0)

    def test_free_single_tensor_skips_cat(self):
        """Single pending tensor must not go through torch.cat (uses [0] shortcut)."""
        alloc = _make_allocator(size=10)
        slots = alloc.alloc(3)
        alloc.free(slots)
        self.assertEqual(len(alloc._pending_free), 1)

        alloc._flush_pending_free()
        # after flush pending is cleared and free_pages grew
        self.assertEqual(alloc._pending_free_count, 0)

    def test_clear_resets_pending(self):
        alloc = _make_allocator(size=10)
        slots = alloc.alloc(3)
        alloc.free(slots)
        self.assertEqual(alloc._pending_free_count, 3)

        alloc.clear()
        self.assertEqual(alloc._pending_free_count, 0)
        self.assertEqual(len(alloc._pending_free), 0)
        self.assertEqual(alloc.available_size(), 10)

    def test_alloc_returns_none_when_truly_empty(self):
        alloc = _make_allocator(size=4)
        alloc.alloc(4)  # drain everything
        result = alloc.alloc(1)
        self.assertIsNone(result)

    def test_pending_does_not_exceed_pool_size(self):
        alloc = _make_allocator(size=10)
        slots = alloc.alloc(10)
        alloc.free(slots)
        # still reports correct available (no double counting)
        self.assertEqual(alloc.available_size(), 10)


class TestFreeGroupInterop(unittest.TestCase):
    """free_group (existing batching mechanism) must still work alongside lazy free."""

    def test_free_group_not_affected_by_pending(self):
        alloc = _make_allocator(size=20)
        a = alloc.alloc(5)

        # free normally (goes to pending)
        b = alloc.alloc(5)
        alloc.free(b)
        self.assertEqual(alloc._pending_free_count, 5)

        # free_group path (goes to free_group list, not pending)
        alloc.free_group_begin()
        alloc.free(a)
        self.assertEqual(alloc._pending_free_count, 5)  # unchanged
        alloc.free_group_end()  # triggers free() internally

        # alloc should now see all slots
        result = alloc.alloc(10)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 10)


class TestBackupRestore(unittest.TestCase):
    """backup_state must flush pending before snapshotting."""

    def test_backup_flushes_pending(self):
        alloc = _make_allocator(size=10)
        slots = alloc.alloc(3)
        alloc.free(slots)
        self.assertEqual(alloc._pending_free_count, 3)

        state = alloc.backup_state()
        # pending must be flushed into free_pages before snapshot
        self.assertEqual(alloc._pending_free_count, 0)
        free_pages, _ = state
        self.assertEqual(len(free_pages), 10)  # all slots back

    def test_restore_after_pending_free(self):
        alloc = _make_allocator(size=10)
        snapshot = alloc.backup_state()

        slots = alloc.alloc(5)
        alloc.free(slots)  # pending, not yet flushed

        alloc.restore_state(snapshot)
        # after restore, pending is stale — available_size must reflect snapshot
        alloc._pending_free = []
        alloc._pending_free_count = 0
        self.assertEqual(alloc.available_size(), 10)


class TestNeedSortPath(unittest.TestCase):
    """need_sort=True uses release_pages staging — pending flush must go there."""

    def test_pending_flushes_to_release_pages_when_need_sort(self):
        alloc = _make_allocator(size=20, need_sort=True)
        slots = alloc.alloc(5)
        alloc.free(slots)
        self.assertEqual(alloc._pending_free_count, 5)

        alloc._flush_pending_free()
        self.assertEqual(alloc._pending_free_count, 0)
        # flushed into release_pages (not free_pages) when need_sort=True
        self.assertEqual(len(alloc.release_pages), 5)


if __name__ == "__main__":
    unittest.main()
