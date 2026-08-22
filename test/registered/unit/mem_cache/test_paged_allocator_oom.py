"""Regression tests for paged allocator out-of-memory handling."""

from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_allocator(*, need_sort=False, size=8):
    return PagedTokenToKVPoolAllocator(
        size=size,
        page_size=4,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


@pytest.mark.parametrize("num_new_pages", [None, 3])
def test_extend_checks_capacity_before_launching_kernel(num_new_pages):
    allocator = _make_allocator()
    free_pages_before = allocator.free_pages.clone()
    prefix_lens = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([12], dtype=torch.int64)
    last_loc = torch.tensor([-1], dtype=torch.int64)

    with patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel") as kernel:
        result = allocator.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            extend_num_tokens=12,
            num_new_pages=num_new_pages,
        )

    assert result is None
    assert torch.equal(allocator.free_pages, free_pages_before)
    kernel.__getitem__.assert_not_called()


def test_decode_checks_capacity_before_launching_kernel():
    allocator = _make_allocator()
    free_pages_before = allocator.free_pages.clone()
    seq_lens = torch.tensor([5, 9, 13], dtype=torch.int64)
    last_loc = torch.tensor([3, 7, 11], dtype=torch.int64)

    with patch("sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel") as kernel:
        result = allocator.alloc_decode(seq_lens, seq_lens, last_loc)

    assert result is None
    assert torch.equal(allocator.free_pages, free_pages_before)
    kernel.__getitem__.assert_not_called()


def test_extend_launches_when_capacity_is_sufficient():
    allocator = _make_allocator()
    prefix_lens = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([8], dtype=torch.int64)
    last_loc = torch.tensor([-1], dtype=torch.int64)

    with patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel") as kernel:
        result = allocator.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            extend_num_tokens=8,
        )

    assert result is not None
    assert result.shape == (8,)
    assert len(allocator.free_pages) == 0
    kernel.__getitem__.assert_called_once_with((1,))


def test_decode_launches_when_capacity_is_sufficient():
    allocator = _make_allocator()
    seq_lens = torch.tensor([5], dtype=torch.int64)
    last_loc = torch.tensor([3], dtype=torch.int64)

    with patch("sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel") as kernel:
        result = allocator.alloc_decode(seq_lens, seq_lens, last_loc)

    assert result is not None
    assert result.shape == (1,)
    assert len(allocator.free_pages) == 1
    kernel.__getitem__.assert_called_once_with((1,))


def test_capacity_check_happens_after_release_pages_are_merged():
    allocator = _make_allocator(need_sort=True, size=12)
    allocated = allocator.alloc(8)
    allocator.free(allocated[:4])

    prefix_lens = torch.tensor([0], dtype=torch.int64)
    seq_lens = torch.tensor([8], dtype=torch.int64)
    last_loc = torch.tensor([-1], dtype=torch.int64)

    with patch("sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel") as kernel:
        result = allocator.alloc_extend(
            prefix_lens,
            prefix_lens,
            seq_lens,
            seq_lens,
            last_loc,
            extend_num_tokens=8,
        )

    assert result is not None
    assert len(allocator.free_pages) == 0
    kernel.__getitem__.assert_called_once_with((1,))
