"""
Unit tests for sglang.srt.hardware_backend.npu.allocator_npu
"""

import sys
import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator import alloc_extend_naive
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")


class MockExtendKernel:
    """Mocks sgl_kernel_npu.mem_cache.allocator.alloc_extend_kernel."""

    def __getitem__(self, grid):
        return self._launch

    @staticmethod
    def _launch(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        next_power_of_2_bs,
        page_size,
        max_num_extend_tokens,
    ):
        alloc_extend_naive(
            prefix_lens,
            seq_lens,
            last_loc,
            free_pages,
            out_indices,
            page_size,
            out_indices.device,
        )


# Mock the NPU kernel module at module scope so all test methods can import it on CPU.
for _mod_name in [
    "sgl_kernel_npu",
    "sgl_kernel_npu.mem_cache",
    "sgl_kernel_npu.mem_cache.allocator",
]:
    sys.modules.setdefault(_mod_name, MagicMock())
sys.modules["sgl_kernel_npu.mem_cache.allocator"].alloc_extend_kernel = (
    MockExtendKernel()
)


class TestNPUPagedTokenToKVPoolAllocator(unittest.TestCase):
    def setUp(self):
        self.alloc = NPUPagedTokenToKVPoolAllocator(
            size=128,
            page_size=128,
            dtype=torch.int64,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.alloc.free_pages = torch.arange(100, dtype=torch.int64)

    def tearDown(self):
        self.alloc.free_pages = torch.arange(100, dtype=torch.int64)
        self.alloc.need_sort = False
        self.alloc.release_pages = torch.empty((0,), dtype=torch.int64)

    # --- Extend tests (small: <200 pages, via mocked kernel) ---

    def test_extend_small_new_page(self):
        self.alloc.free_pages = torch.arange(100, dtype=torch.int64)
        seq_lens = torch.tensor([256], dtype=torch.int32)
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)
        result = self.alloc.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            256,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(len(self.alloc.free_pages), 98)

    def test_extend_small_returns_none_when_exhausted(self):
        self.alloc.free_pages = torch.arange(0, dtype=torch.int64)
        seq_lens = torch.tensor([256], dtype=torch.int32)
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)
        result = self.alloc.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            256,
        )
        self.assertIsNone(result)

    # --- Extend tests (large: >=200 pages, native naive path) ---

    def test_extend_large_allocate_pages(self):
        self.alloc.free_pages = torch.arange(500, dtype=torch.int64)
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([26000], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)
        result = self.alloc.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            26000,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(len(self.alloc.free_pages), 296)

    def test_extend_large_returns_none_when_exhausted(self):
        self.alloc.free_pages = torch.arange(0, dtype=torch.int64)
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([26000], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)
        result = self.alloc.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            26000,
        )
        self.assertIsNone(result)

    def test_extend_large_mixed_batch(self):
        self.alloc.free_pages = torch.arange(500, dtype=torch.int64)
        prefix_lens = torch.tensor([0, 0], dtype=torch.int32)
        seq_lens = torch.tensor([26000, 100], dtype=torch.int32)
        last_loc = torch.tensor([-1, 5], dtype=torch.int32)
        result = self.alloc.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            26100,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(len(self.alloc.free_pages), 295)

    # --- Decode tests ---

    def test_no_new_page_needed(self):
        seq_lens = torch.tensor([2], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([2], dtype=torch.int32)
        last_loc = torch.tensor([5], dtype=torch.int32)
        result = self.alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        self.assertEqual(result[0].item(), 6)

    def test_new_page_allocated(self):
        seq_lens = torch.tensor([1], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([1], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)
        result = self.alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        self.assertEqual(result[0].item(), 0)

    def test_mixed_batch(self):
        seq_lens = torch.tensor([1, 2], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([1, 2], dtype=torch.int32)
        last_loc = torch.tensor([-1, 5], dtype=torch.int32)
        result = self.alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        self.assertEqual(result[0].item(), 0)
        self.assertEqual(result[1].item(), 6)

    def test_free_pages_consumed(self):
        self.alloc.free_pages = torch.arange(10, dtype=torch.int64)
        seq_lens = torch.tensor([1, 129], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([1, 129], dtype=torch.int32)
        last_loc = torch.tensor([-1, 127], dtype=torch.int32)
        self.alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        self.assertEqual(len(self.alloc.free_pages), 8)

    def test_returns_none_when_exhausted(self):
        self.alloc.free_pages = torch.arange(0, dtype=torch.int64)
        seq_lens = torch.tensor([1], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([1], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)
        result = self.alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        self.assertIsNone(result)

    def test_returns_int32(self):
        seq_lens = torch.tensor([1], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([1], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)
        result = self.alloc.alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        self.assertEqual(result.dtype, torch.int32)

    # --- Free tests ---

    def test_empty_free_index_noop(self):
        free_pages_before = len(self.alloc.free_pages)
        self.alloc.free(torch.empty(0, dtype=torch.int64))
        self.assertEqual(len(self.alloc.free_pages), free_pages_before)

    def test_free_adds_page_indices(self):
        self.alloc.free_pages = torch.arange(5, dtype=torch.int64)
        free_index = torch.tensor([128, 129, 256], dtype=torch.int64)
        self.alloc.free(free_index)
        self.assertEqual(len(self.alloc.free_pages), 7)

    def test_free_deduplicates_pages(self):
        self.alloc.free_pages = torch.arange(5, dtype=torch.int64)
        free_index = torch.tensor([128, 129, 130], dtype=torch.int64)
        self.alloc.free(free_index)
        self.assertEqual(len(self.alloc.free_pages), 6)

    def test_free_with_need_sort(self):
        self.alloc.free_pages = torch.arange(5, dtype=torch.int64)
        self.alloc.need_sort = True
        free_index = torch.tensor([128], dtype=torch.int64)
        self.alloc.free(free_index)
        self.assertEqual(len(self.alloc.release_pages), 1)
        self.assertEqual(self.alloc.release_pages[0].item(), 1)
        self.assertEqual(len(self.alloc.free_pages), 5)


if __name__ == "__main__":
    unittest.main()
