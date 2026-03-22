"""Unit tests for allocator.py — GPU-required tests"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-test-small-1-gpu")

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.test.test_utils import CustomTestCase

CUDA = torch.cuda.is_available()


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
# PagedTokenToKVPoolAllocator — GPU / Triton paths
# ---------------------------------------------------------------------------


@unittest.skipUnless(CUDA, "CUDA required")
class TestPagedAllocatorGPU(CustomTestCase):
    D = "cuda"

    def _alloc(self, size=256, page_size=4, need_sort=False):
        return _paged_alloc(
            size=size, page_size=page_size, need_sort=need_sort, device=self.D
        )

    def _t(self, data):
        return torch.tensor(data, dtype=torch.int64, device=self.D)

    def test_alloc_extend_output_length(self):
        a = self._alloc()
        pre, seq, ll = self._t([0, 4]), self._t([8, 8]), self._t([-1, 3])
        extend = int((seq - pre).sum())
        out = a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, extend)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), extend)

    def test_alloc_extend_unique_indices(self):
        a = self._alloc()
        pre, seq, ll = self._t([0]), self._t([8]), self._t([-1])
        out = a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, 8)
        self.assertEqual(len(torch.unique(out)), 8)

    def test_alloc_extend_reduces_available(self):
        a = self._alloc()
        before = a.available_size()
        pre, seq, ll = self._t([0]), self._t([8]), self._t([-1])
        a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, 8)
        self.assertLess(a.available_size(), before)

    def test_alloc_extend_backup_restore(self):
        a = self._alloc()
        state = a.backup_state()
        pre, seq, ll = self._t([0]), self._t([8]), self._t([-1])
        a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, 8)
        a.restore_state(state)
        self.assertEqual(a.available_size(), 256)

    def test_alloc_extend_need_sort_triggers_merge(self):
        a = self._alloc(need_sort=True)
        pre, seq, ll = self._t([0]), self._t([8]), self._t([-1])
        out = a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, 8)
        self.assertIsNotNone(out)

    def test_alloc_extend_need_sort_oom(self):
        a = self._alloc(size=16, page_size=4, need_sort=True)
        pre = self._t([0])
        seq = self._t([32])
        ll = self._t([-1])
        out = a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, 32)
        self.assertIsNone(out)

    def test_alloc_decode_output_length(self):
        a = self._alloc()
        seq = self._t([4, 8, 12, 16])
        ll = self._t([3, 7, 11, 15])
        out = a.alloc_decode(seq, seq.cpu(), ll)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 4)

    def test_alloc_decode_crossing_page_boundary(self):
        a = self._alloc()
        seq, ll = self._t([5]), self._t([3])
        out = a.alloc_decode(seq, seq.cpu(), ll)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 1)

    def test_alloc_decode_within_page_no_new_page_consumed(self):
        a = self._alloc(256, 4)
        before = a.available_size()
        seq, ll = self._t([4]), self._t([2])
        a.alloc_decode(seq, seq.cpu(), ll)
        self.assertEqual(a.available_size(), before)

    def test_alloc_decode_reduces_available(self):
        a = self._alloc()
        before = a.available_size()
        seq, ll = self._t([5]), self._t([3])
        a.alloc_decode(seq, seq.cpu(), ll)
        self.assertLess(a.available_size(), before)

    def test_alloc_decode_need_sort_triggers_merge(self):
        a = self._alloc(need_sort=True)
        seq, ll = self._t([5]), self._t([3])
        out = a.alloc_decode(seq, seq.cpu(), ll)
        self.assertIsNotNone(out)

    def test_alloc_decode_need_sort_oom(self):
        a = self._alloc(size=16, page_size=4, need_sort=True)
        seq = self._t([5, 9, 13, 17, 21])
        ll = self._t([3, 7, 11, 15, 19])
        out = a.alloc_decode(seq, seq.cpu(), ll)
        self.assertIsNone(out)


@unittest.skipUnless(CUDA, "CUDA required")
class TestPagedAllocatorGPUTritonCoverage(CustomTestCase):
    D = "cuda"

    def _alloc(self, size=256, page_size=4, need_sort=False):
        return _paged_alloc(
            size=size, page_size=page_size, need_sort=need_sort, device=self.D
        )

    def _t(self, data):
        return torch.tensor(data, dtype=torch.int64, device=self.D)

    def test_alloc_extend_unaligned_branches(self):
        a = self._alloc(size=256, page_size=4)
        pre = self._t([2])
        seq = self._t([11])
        ll = self._t([1])
        out = a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, 9)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 9)
        self.assertEqual(len(torch.unique(out)), 9)

        out_list = out.tolist()
        self.assertEqual(out_list[:2], [2, 3])
        self.assertEqual(out_list[2:6], [4, 5, 6, 7])
        self.assertEqual(out_list[6:], [8, 9, 10])

    def test_alloc_extend_multiple_unaligned_requests(self):
        a = self._alloc(size=256, page_size=4)
        _ = a.alloc(8)
        pre = self._t([1, 3])
        seq = self._t([6, 7])
        ll = self._t([4, 10])
        extend_num = int((seq - pre).sum().item())
        out = a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, extend_num)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 9)
        self.assertEqual(len(torch.unique(out)), 9)


# ---------------------------------------------------------------------------
# PagedTokenToKVPoolAllocator — debug_mode GPU paths
# ---------------------------------------------------------------------------


@unittest.skipUnless(CUDA, "CUDA required")
class TestPagedAllocatorDebugModeGPU(CustomTestCase):
    def _debug_alloc(self, size=256, page_size=4, device="cuda"):
        return PagedTokenToKVPoolAllocator(
            size=size,
            page_size=page_size,
            dtype=torch.int64,
            device=device,
            kvcache=MagicMock(),
            need_sort=False,
        )

    @patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"})
    def test_alloc_extend_debug_unique_assert(self):
        a = self._debug_alloc()
        pre = torch.tensor([0], dtype=torch.int64, device="cuda")
        seq = torch.tensor([8], dtype=torch.int64, device="cuda")
        ll = torch.tensor([-1], dtype=torch.int64, device="cuda")
        out = a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, 8)
        self.assertIsNotNone(out)
        self.assertEqual(len(torch.unique(out)), len(out))

    @patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"})
    def test_alloc_extend_debug_mode_succeeds_when_last_loc_aligned_with_prefix(self):
        a = self._debug_alloc()
        pre = torch.tensor([4], dtype=torch.int64, device="cuda")
        seq = torch.tensor([8], dtype=torch.int64, device="cuda")
        ll = torch.tensor([3], dtype=torch.int64, device="cuda")
        out = a.alloc_extend(pre, pre.cpu(), seq, seq.cpu(), ll, 4)
        self.assertIsNotNone(out)

    @patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"})
    def test_alloc_decode_debug_unique_assert(self):
        a = self._debug_alloc()
        seq = torch.tensor([5, 9], dtype=torch.int64, device="cuda")
        ll = torch.tensor([3, 7], dtype=torch.int64, device="cuda")
        out = a.alloc_decode(seq, seq.cpu(), ll)
        self.assertIsNotNone(out)
        self.assertEqual(len(torch.unique(out)), len(out))

    @patch.dict(os.environ, {"SGLANG_DEBUG_MEMORY_POOL": "1"})
    def test_alloc_decode_debug_mode_succeeds_when_last_loc_aligned_with_seq_len(self):
        a = self._debug_alloc()
        seq = torch.tensor([5], dtype=torch.int64, device="cuda")
        ll = torch.tensor([3], dtype=torch.int64, device="cuda")
        out = a.alloc_decode(seq, seq.cpu(), ll)
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
