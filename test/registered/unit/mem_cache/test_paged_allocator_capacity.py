import unittest
from unittest.mock import patch

import torch

import sglang.srt.mem_cache.allocator.paged as paged
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_allocator(*, free_pages, release_pages=(), need_sort=False):
    allocator = object.__new__(PagedTokenToKVPoolAllocator)
    allocator.page_size = 8
    allocator.device = "cpu"
    allocator.debug_mode = False
    allocator.need_sort = need_sort
    allocator.free_pages = torch.tensor(free_pages, dtype=torch.int64)
    allocator.release_pages = torch.tensor(release_pages, dtype=torch.int64)
    return allocator


class TestPagedAllocatorCapacity(CustomTestCase):
    def test_alloc_extend_does_not_launch_kernel_when_capacity_is_insufficient(self):
        allocator = make_allocator(free_pages=[7])
        free_pages = allocator.free_pages.clone()

        with patch.object(paged, "alloc_extend_kernel") as kernel:
            result = allocator.alloc_extend(
                prefix_lens=torch.tensor([8, 8]),
                prefix_lens_cpu=torch.tensor([8, 8]),
                seq_lens=torch.tensor([9, 9]),
                seq_lens_cpu=torch.tensor([9, 9]),
                last_loc=torch.tensor([7, 7]),
                extend_num_tokens=2,
            )

        self.assertIsNone(result)
        kernel.__getitem__.assert_not_called()
        self.assertTrue(torch.equal(allocator.free_pages, free_pages))

    def test_alloc_decode_does_not_launch_kernel_when_capacity_is_insufficient(self):
        allocator = make_allocator(free_pages=[7])
        free_pages = allocator.free_pages.clone()

        with patch.object(paged, "alloc_decode_kernel") as kernel:
            result = allocator.alloc_decode(
                seq_lens=torch.tensor([9, 9]),
                seq_lens_cpu=torch.tensor([9, 9]),
                last_loc=torch.tensor([7, 7]),
            )

        self.assertIsNone(result)
        kernel.__getitem__.assert_not_called()
        self.assertTrue(torch.equal(allocator.free_pages, free_pages))

    def test_alloc_extend_does_not_merge_released_pages_on_oom(self):
        allocator = make_allocator(free_pages=[7], release_pages=[3], need_sort=True)
        free_pages = allocator.free_pages.clone()
        release_pages = allocator.release_pages.clone()

        with patch.object(paged, "alloc_extend_kernel") as kernel:
            result = allocator.alloc_extend(
                prefix_lens=torch.tensor([8, 8, 8]),
                prefix_lens_cpu=torch.tensor([8, 8, 8]),
                seq_lens=torch.tensor([9, 9, 9]),
                seq_lens_cpu=torch.tensor([9, 9, 9]),
                last_loc=torch.tensor([7, 7, 7]),
                extend_num_tokens=3,
            )

        self.assertIsNone(result)
        kernel.__getitem__.assert_not_called()
        self.assertTrue(torch.equal(allocator.free_pages, free_pages))
        self.assertTrue(torch.equal(allocator.release_pages, release_pages))

    def test_alloc_decode_does_not_merge_released_pages_on_oom(self):
        allocator = make_allocator(free_pages=[7], release_pages=[3], need_sort=True)
        free_pages = allocator.free_pages.clone()
        release_pages = allocator.release_pages.clone()

        with patch.object(paged, "alloc_decode_kernel") as kernel:
            result = allocator.alloc_decode(
                seq_lens=torch.tensor([9, 9, 9]),
                seq_lens_cpu=torch.tensor([9, 9, 9]),
                last_loc=torch.tensor([7, 7, 7]),
            )

        self.assertIsNone(result)
        kernel.__getitem__.assert_not_called()
        self.assertTrue(torch.equal(allocator.free_pages, free_pages))
        self.assertTrue(torch.equal(allocator.release_pages, release_pages))

    def test_alloc_extend_launches_at_exact_capacity(self):
        allocator = make_allocator(free_pages=[9, 3, 7])

        with patch.object(paged, "alloc_extend_kernel") as kernel:
            result = allocator.alloc_extend(
                prefix_lens=torch.tensor([8, 8]),
                prefix_lens_cpu=torch.tensor([8, 8]),
                seq_lens=torch.tensor([9, 9]),
                seq_lens_cpu=torch.tensor([9, 9]),
                last_loc=torch.tensor([7, 7]),
                extend_num_tokens=2,
            )

        self.assertIsNotNone(result)
        kernel.__getitem__.assert_called_once_with((2,))
        launched_free_pages = kernel.__getitem__.return_value.call_args.args[3]
        self.assertTrue(torch.equal(launched_free_pages, torch.tensor([9, 3, 7])))
        self.assertTrue(torch.equal(allocator.free_pages, torch.tensor([7])))

    def test_alloc_decode_launches_at_exact_capacity(self):
        allocator = make_allocator(free_pages=[9, 3])

        with patch.object(paged, "alloc_decode_kernel") as kernel:
            result = allocator.alloc_decode(
                seq_lens=torch.tensor([9, 9]),
                seq_lens_cpu=torch.tensor([9, 9]),
                last_loc=torch.tensor([7, 7]),
            )

        self.assertIsNotNone(result)
        kernel.__getitem__.assert_called_once_with((2,))
        launched_free_pages = kernel.__getitem__.return_value.call_args.args[2]
        self.assertTrue(torch.equal(launched_free_pages, torch.tensor([9, 3])))
        self.assertEqual(len(allocator.free_pages), 0)

    def test_allocations_launch_when_no_new_page_is_required(self):
        extend_allocator = make_allocator(free_pages=[])
        decode_allocator = make_allocator(free_pages=[])

        with patch.object(paged, "alloc_extend_kernel") as extend_kernel:
            extend_result = extend_allocator.alloc_extend(
                prefix_lens=torch.tensor([5]),
                prefix_lens_cpu=torch.tensor([5]),
                seq_lens=torch.tensor([6]),
                seq_lens_cpu=torch.tensor([6]),
                last_loc=torch.tensor([4]),
                extend_num_tokens=1,
            )

        with patch.object(paged, "alloc_decode_kernel") as decode_kernel:
            decode_result = decode_allocator.alloc_decode(
                seq_lens=torch.tensor([2]),
                seq_lens_cpu=torch.tensor([2]),
                last_loc=torch.tensor([0]),
            )

        self.assertIsNotNone(extend_result)
        self.assertIsNotNone(decode_result)
        extend_kernel.__getitem__.assert_called_once_with((1,))
        decode_kernel.__getitem__.assert_called_once_with((1,))

    def test_alloc_extend_merges_released_pages_before_capacity_check(self):
        allocator = make_allocator(free_pages=[7], release_pages=[3], need_sort=True)

        with patch.object(paged, "alloc_extend_kernel") as kernel:
            result = allocator.alloc_extend(
                prefix_lens=torch.tensor([8, 8]),
                prefix_lens_cpu=torch.tensor([8, 8]),
                seq_lens=torch.tensor([9, 9]),
                seq_lens_cpu=torch.tensor([9, 9]),
                last_loc=torch.tensor([7, 7]),
                extend_num_tokens=2,
            )

        self.assertIsNotNone(result)
        launched_free_pages = kernel.__getitem__.return_value.call_args.args[3]
        self.assertTrue(torch.equal(launched_free_pages, torch.tensor([3, 7])))
        self.assertEqual(len(allocator.free_pages), 0)
        self.assertEqual(len(allocator.release_pages), 0)


if __name__ == "__main__":
    unittest.main()
