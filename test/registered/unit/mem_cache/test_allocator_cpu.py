"""Unit tests for allocator.py — CPU-only tests"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.test.test_utils import CustomTestCase


def _token_alloc(size=16, need_sort=False, device="cpu"):
    return TokenToKVPoolAllocator(
        size=size,
        dtype=torch.int64,
        device=device,
        kvcache=MagicMock(),
        need_sort=need_sort,
    )


def _paged_alloc(size=64, page_size=4, need_sort=False, device="cpu"):
    return PagedTokenToKVPoolAllocator(
        size=size,
        page_size=page_size,
        dtype=torch.int64,
        device=device,
        kvcache=MagicMock(),
        need_sort=need_sort,
    )


# ---------------------------------------------------------------------------
# BaseTokenToKVPoolAllocator
# ---------------------------------------------------------------------------


class TestBaseAllocatorNotImplemented(CustomTestCase):
    def setUp(self):
        super().setUp()

        class MinimalAlloc(BaseTokenToKVPoolAllocator):
            def __init__(self):
                super().__init__(8, 1, torch.int64, "cpu", MagicMock(), False)
                self.free_pages = torch.arange(1, 9, dtype=torch.int64)
                self.release_pages = torch.empty(0, dtype=torch.int64)

            def clear(self):
                pass

            def alloc(self, n):
                pass

            def free(self, idx):
                pass

        self.a = MinimalAlloc()

    def test_get_cpu_copy_raises(self):
        with self.assertRaises(NotImplementedError):
            self.a.get_cpu_copy()

    def test_load_cpu_copy_raises(self):
        with self.assertRaises(NotImplementedError):
            self.a.load_cpu_copy()

    def test_alloc_extend_raises(self):
        with self.assertRaises(NotImplementedError):
            self.a.alloc_extend()

    def test_alloc_decode_raises(self):
        with self.assertRaises(NotImplementedError):
            self.a.alloc_decode()

    def test_available_size(self):
        self.assertEqual(self.a.available_size(), 8)

    def test_debug_print_empty_string(self):
        self.assertEqual(self.a.debug_print(), "")

    def test_get_kvcache(self):
        self.assertEqual(self.a.get_kvcache(), self.a._kvcache)

    def test_merge_and_sort_free(self):
        self.a.release_pages = torch.tensor([3, 1, 2], dtype=torch.int64)
        self.a.free_pages = torch.tensor([6, 5], dtype=torch.int64)
        self.a.merge_and_sort_free()
        self.assertEqual(self.a.free_pages.tolist(), sorted([6, 5, 3, 1, 2]))
        self.assertEqual(len(self.a.release_pages), 0)

    def test_merge_and_sort_free_empty_release_is_noop(self):
        before = self.a.free_pages.tolist()
        self.a.merge_and_sort_free()
        self.assertEqual(self.a.free_pages.tolist(), before)


# ---------------------------------------------------------------------------
# TokenToKVPoolAllocator
# ---------------------------------------------------------------------------


class TestTokenAllocatorAlloc(CustomTestCase):
    def test_returns_correct_count(self):
        self.assertEqual(len(_token_alloc(16).alloc(4)), 4)

    def test_reduces_available_size(self):
        a = _token_alloc(16)
        a.alloc(4)
        self.assertEqual(a.available_size(), 12)

    def test_returns_unique_indices(self):
        out = _token_alloc(16).alloc(8)
        self.assertEqual(len(torch.unique(out)), 8)

    def test_exact_capacity_succeeds(self):
        a = _token_alloc(8)
        self.assertIsNotNone(a.alloc(8))
        self.assertEqual(a.available_size(), 0)

    def test_over_capacity_returns_none(self):
        self.assertIsNone(_token_alloc(8).alloc(9))

    def test_over_capacity_after_need_sort_merge_returns_none(self):
        a = _token_alloc(4, need_sort=True)
        self.assertIsNone(a.alloc(8))

    def test_zero_alloc_is_noop(self):
        a = _token_alloc(8)
        out = a.alloc(0)
        self.assertEqual(len(out), 0)
        self.assertEqual(a.available_size(), 8)

    def test_sequential_allocs_non_overlapping(self):
        a = _token_alloc(16)
        s1 = set(a.alloc(4).tolist())
        s2 = set(a.alloc(4).tolist())
        self.assertEqual(len(s1 & s2), 0)

    def test_slot_zero_never_returned(self):
        self.assertNotIn(0, _token_alloc(16).alloc(16).tolist())

    def test_need_sort_triggers_merge_before_alloc(self):
        a = _token_alloc(8, need_sort=True)
        a.free(a.alloc(4))
        self.assertIsNotNone(a.alloc(8))


class TestTokenAllocatorFree(CustomTestCase):
    def test_restores_available_size(self):
        a = _token_alloc(16)
        a.free(a.alloc(4))
        self.assertEqual(a.available_size(), 16)

    def test_empty_tensor_is_noop(self):
        a = _token_alloc(8)
        before = a.available_size()
        a.free(torch.tensor([], dtype=torch.int64))
        self.assertEqual(a.available_size(), before)

    def test_freed_indices_reusable(self):
        a = _token_alloc(8)
        a.free(a.alloc(8))
        self.assertIsNotNone(a.alloc(8))

    def test_need_sort_free_goes_to_release_pages(self):
        a = _token_alloc(8, need_sort=True)
        a.free(a.alloc(4))
        self.assertEqual(len(a.release_pages), 4)

    def test_no_sort_free_goes_directly_to_free_pages(self):
        a = _token_alloc(8, need_sort=False)
        out = a.alloc(4)
        a.free(out)
        self.assertEqual(len(a.free_pages), 8)


class TestTokenAllocatorBackupRestore(CustomTestCase):
    def test_restore_reverts_alloc(self):
        a = _token_alloc(16)
        state = a.backup_state()
        a.alloc(8)
        a.restore_state(state)
        self.assertEqual(a.available_size(), 16)

    def test_restore_reverts_free(self):
        a = _token_alloc(16)
        out = a.alloc(4)
        state = a.backup_state()
        a.free(out)
        a.restore_state(state)
        self.assertEqual(a.available_size(), 12)

    def test_backup_is_snapshot_not_reference(self):
        a = _token_alloc(16)
        state = a.backup_state()
        a.alloc(8)
        a.alloc(4)
        a.restore_state(state)
        self.assertEqual(a.available_size(), 16)


class TestTokenAllocatorFreeGroup(CustomTestCase):
    def test_batches_frees_until_end(self):
        a = _token_alloc(16)
        x, y = a.alloc(4), a.alloc(4)
        a.free_group_begin()
        a.free(x)
        a.free(y)
        self.assertEqual(a.available_size(), 8)
        a.free_group_end()
        self.assertEqual(a.available_size(), 16)

    def test_empty_group_end_is_noop(self):
        a = _token_alloc(8)
        a.alloc(4)
        a.free_group_begin()
        a.free_group_end()
        self.assertEqual(a.available_size(), 4)


class TestTokenAllocatorKVCacheDelegation(CustomTestCase):
    def test_get_cpu_copy_delegates(self):
        kv = MagicMock()
        kv.get_cpu_copy.return_value = "cpu_data"
        a = TokenToKVPoolAllocator(8, torch.int64, "cpu", kv, False)
        result = a.get_cpu_copy(torch.tensor([1, 2]))
        kv.get_cpu_copy.assert_called_once()
        self.assertEqual(result, "cpu_data")

    def test_load_cpu_copy_delegates(self):
        kv = MagicMock()
        a = TokenToKVPoolAllocator(8, torch.int64, "cpu", kv, False)
        a.load_cpu_copy("data", torch.tensor([1, 2]))
        kv.load_cpu_copy.assert_called_once()


# ---------------------------------------------------------------------------
# alloc_extend_naive
# ---------------------------------------------------------------------------


class TestAllocExtendNaive(CustomTestCase):
    PAGE = 4

    def _run(self, prefix_lens, seq_lens, last_loc, free_pages):
        pl = torch.tensor(prefix_lens, dtype=torch.int64, device="cpu")
        sl = torch.tensor(seq_lens, dtype=torch.int64, device="cpu")
        ll = torch.tensor(last_loc, dtype=torch.int64, device="cpu")
        fp = torch.tensor(free_pages, dtype=torch.int64, device="cpu")
        total = int((sl - pl).sum().item())
        out = torch.empty(total, dtype=torch.int64, device="cpu")
        alloc_extend_naive(pl, sl, ll, fp, out, self.PAGE, "cpu")
        return out

    def test_num1_only(self):
        out = self._run([2], [4], [1], [])
        self.assertEqual(out.tolist(), [2, 3])

    def test_num2_only(self):
        out = self._run([0], [8], [-1], [5, 6])
        self.assertEqual(out.tolist(), list(range(20, 24)) + list(range(24, 28)))

    def test_num3_only(self):
        out = self._run([4], [7], [3], [3])
        self.assertEqual(len(out), 3)
        base = 3 * self.PAGE
        self.assertEqual(out.tolist(), [base, base + 1, base + 2])

    def test_num1_and_num2(self):
        out = self._run([2], [8], [1], [3])
        self.assertEqual(out[:2].tolist(), [2, 3])
        self.assertEqual(out[2:].tolist(), list(range(12, 16)))

    def test_num1_and_num3(self):
        out = self._run([2], [7], [1], [3])
        self.assertEqual(out[:2].tolist(), [2, 3])
        self.assertEqual(out[2:].tolist(), [12, 13, 14])

    def test_num2_and_num3(self):
        out = self._run([4], [11], [3], [2, 3])
        self.assertEqual(out[:4].tolist(), list(range(8, 12)))
        self.assertEqual(out[4:].tolist(), [12, 13, 14])

    def test_num1_and_num2_and_num3(self):
        out = self._run([2], [11], [1], [3, 4])
        self.assertEqual(out[:2].tolist(), [2, 3])
        self.assertEqual(out[2:6].tolist(), list(range(12, 16)))
        self.assertEqual(out[6:].tolist(), [16, 17, 18])

    def test_zero_extend(self):
        out = self._run([4], [4], [3], [])
        self.assertEqual(len(out), 0)

    def test_multi_req_no_overlap(self):
        out = self._run([0, 0], [4, 4], [-1, -1], [1, 2])
        self.assertEqual(len(torch.unique(out)), 8)

    def test_output_length_equals_extend_sum(self):
        out = self._run([0, 2, 4], [4, 6, 8], [-1, 1, 3], list(range(1, 20)))
        self.assertEqual(len(out), 12)


# ---------------------------------------------------------------------------
# PagedTokenToKVPoolAllocator — CPU paths
# ---------------------------------------------------------------------------


class TestPagedAllocatorCPU(CustomTestCase):
    def test_alloc_returns_correct_count(self):
        self.assertEqual(len(_paged_alloc(64, 4).alloc(8)), 8)

    def test_alloc_reduces_available_size(self):
        a = _paged_alloc(64, 4)
        a.alloc(8)
        self.assertEqual(a.available_size(), 56)

    def test_over_capacity_returns_none(self):
        self.assertIsNone(_paged_alloc(16, 4).alloc(20))

    def test_alloc_need_sort_oom(self):
        a = _paged_alloc(16, 4, need_sort=True)
        self.assertIsNone(a.alloc(32))

    def test_indices_are_page_aligned(self):
        out = _paged_alloc(64, 4).alloc(8)
        self.assertEqual(out[0].item() % 4, 0)
        self.assertEqual(out[4].item() % 4, 0)

    def test_indices_are_unique(self):
        out = _paged_alloc(64, 4).alloc(16)
        self.assertEqual(len(torch.unique(out)), 16)

    def test_slot_zero_never_returned(self):
        self.assertNotIn(0, _paged_alloc(64, 4).alloc(64).tolist())

    def test_free_restores_available_size(self):
        a = _paged_alloc(32, 4)
        a.free(a.alloc(8))
        self.assertEqual(a.available_size(), 32)

    def test_free_empty_tensor_is_noop(self):
        a = _paged_alloc(16, 4)
        before = a.available_size()
        a.free(torch.tensor([], dtype=torch.int64))
        self.assertEqual(a.available_size(), before)

    def test_freed_pages_reusable(self):
        a = _paged_alloc(16, 4)
        a.free(a.alloc(16))
        self.assertIsNotNone(a.alloc(16))

    def test_need_sort_free_goes_to_release_pages(self):
        a = _paged_alloc(32, 4, need_sort=True)
        a.free(a.alloc(4))
        self.assertEqual(a.available_size(), 32)

    def test_free_group_batches_until_end(self):
        a = _paged_alloc(32, 4)
        x, y = a.alloc(4), a.alloc(4)
        a.free_group_begin()
        a.free(x)
        a.free(y)
        self.assertEqual(a.available_size(), 24)
        a.free_group_end()
        self.assertEqual(a.available_size(), 32)

    def test_restore_after_alloc(self):
        a = _paged_alloc(32, 4)
        state = a.backup_state()
        a.alloc(16)
        a.restore_state(state)
        self.assertEqual(a.available_size(), 32)

    def test_clear_restores_full_capacity(self):
        a = _paged_alloc(32, 4)
        a.alloc(32)
        a.clear()
        self.assertEqual(a.available_size(), 32)

    def test_get_cpu_copy_delegates(self):
        kv = MagicMock()
        kv.get_cpu_copy.return_value = "data"
        a = PagedTokenToKVPoolAllocator(16, 4, torch.int64, "cpu", kv, False)
        result = a.get_cpu_copy(torch.tensor([4, 8]))
        kv.get_cpu_copy.assert_called_once()
        self.assertEqual(result, "data")

    def test_load_cpu_copy_delegates(self):
        kv = MagicMock()
        a = PagedTokenToKVPoolAllocator(16, 4, torch.int64, "cpu", kv, False)
        a.load_cpu_copy("data", torch.tensor([4, 8]))
        kv.load_cpu_copy.assert_called_once()


# ---------------------------------------------------------------------------
# PagedTokenToKVPoolAllocator — debug_mode CPU paths
# ---------------------------------------------------------------------------


class TestPagedAllocatorDebugModeCPU(CustomTestCase):
    def _debug_alloc(self, size=64, page_size=4, device="cpu"):
        return PagedTokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            dtype=torch.int64,
            device=device,
            kvcache=MagicMock(),
            need_sort=False,
        )

    @patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"})
    def test_alloc_unaligned_size_raises(self):
        a = self._debug_alloc()
        with self.assertRaises(AssertionError):
            a.alloc(3)

    @patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"})
    def test_alloc_aligned_size_passes(self):
        a = self._debug_alloc()
        out = a.alloc(4)
        self.assertIsNotNone(out)

    @patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"})
    def test_free_duplicate_pages_raises(self):
        a = self._debug_alloc()
        out = a.alloc(4)
        a.free(out)
        with self.assertRaises(AssertionError):
            a.free(out)


if __name__ == "__main__":
    unittest.main()
