"""Regression tests for paged allocator OOM gating."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE = 4


class _UnexpectedKernelLaunch:
    def __getitem__(self, _):
        raise AssertionError("the allocator must reject OOM before launching Triton")


def _make_allocator() -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        size=PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )


class TestPagedAllocatorOOMGate(CustomTestCase):
    def test_extend_rejects_insufficient_pages_before_kernel_launch(self):
        """An oversized prefill returns OOM without reading past the free-page list."""
        allocator = _make_allocator()
        prefix_lens = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([2 * PAGE_SIZE], dtype=torch.int64)

        with patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel",
            _UnexpectedKernelLaunch(),
        ):
            result = allocator.alloc_extend(
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens,
                last_loc=torch.tensor([-1], dtype=torch.int64),
                extend_num_tokens=2 * PAGE_SIZE,
            )

        self.assertIsNone(result)

    def test_decode_rejects_insufficient_pages_before_kernel_launch(self):
        """A page-wrapping decode returns OOM without reading past the free-page list."""
        allocator = _make_allocator()
        allocated = allocator.alloc(PAGE_SIZE)
        self.assertIsNotNone(allocated)
        seq_lens = torch.tensor([PAGE_SIZE + 1], dtype=torch.int64)

        with patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel",
            _UnexpectedKernelLaunch(),
        ):
            result = allocator.alloc_decode(
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens,
                last_loc=allocated[-1:],
            )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
