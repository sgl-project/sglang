"""MPS pages must survive partial extension, exhaustion and grouped reuse."""

import unittest

import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=10, suite="stage-a-unit-test-mps")


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires Torch MPS")
class TestMPSPagedAllocator(CustomTestCase):
    def test_partial_pages_exhaustion_and_reuse(self):
        from sglang.srt.hardware_backend.mps.allocator import (
            MPSPagedTokenToKVPoolAllocator,
        )

        for page_size in (16, 32, 64):
            for need_sort in (False, True):
                with self.subTest(page_size=page_size, need_sort=need_sort):
                    p = page_size
                    allocator = MPSPagedTokenToKVPoolAllocator(
                        size=4 * p,
                        page_size=p,
                        dtype=torch.float32,
                        device="mps",
                        kvcache=None,
                        need_sort=need_sort,
                    )
                    allocator.free_pages = torch.tensor([3, 1, 4, 2], device="mps")
                    prefix = torch.tensor([0, 0])
                    lengths = torch.tensor([p - 1, p + 1])
                    slots = allocator.alloc_extend(
                        prefix.to("mps"),
                        prefix,
                        lengths.to("mps"),
                        lengths,
                        torch.tensor([-1, -1], device="mps"),
                        2 * p,
                    )
                    self.assertEqual(
                        slots.cpu().tolist(),
                        [*range(3 * p, 4 * p - 1), *range(p, 2 * p), 4 * p],
                    )
                    self.assertEqual(allocator.available_size(), p)
                    last = slots[torch.tensor([p - 2, 2 * p - 1], device="mps")]
                    lengths = lengths + 1
                    inside = allocator.alloc_decode(
                        lengths.to("mps"), lengths, last.int()
                    )
                    self.assertEqual(inside.dtype, torch.int64)
                    self.assertEqual(inside.cpu().tolist(), [4 * p - 1, 4 * p + 1])
                    self.assertEqual(allocator.available_size(), p)
                    lengths = lengths + 1
                    boundary = allocator.alloc_decode(
                        lengths.to("mps"), lengths, inside.int()
                    )
                    self.assertEqual(boundary.dtype, torch.int64)
                    self.assertEqual(boundary.cpu().tolist(), [2 * p, 4 * p + 2])
                    self.assertEqual(allocator.available_size(), 0)
                    prefix = lengths
                    lengths = lengths + torch.tensor([3, 0])
                    partial = allocator.alloc_extend(
                        prefix.to("mps"),
                        prefix,
                        lengths.to("mps"),
                        lengths,
                        boundary,
                        3,
                    )
                    self.assertEqual(
                        partial.cpu().tolist(), list(range(2 * p + 1, 2 * p + 4))
                    )
                    full = torch.tensor([2 * p + 1, p + 3])
                    self.assertIsNone(
                        allocator.alloc_extend(
                            lengths.to("mps"),
                            lengths,
                            full.to("mps"),
                            full,
                            torch.stack((partial[-1], boundary[-1])),
                            int((full - lengths).sum()),
                        )
                    )
                    self.assertIsNone(
                        allocator.alloc_decode(full.to("mps"), full, boundary)
                    )
                    self.assertEqual(allocator.available_size(), 0)
                    allocator.free_group_begin()
                    for freed in (slots, inside, boundary, partial):
                        allocator.free(freed)
                    allocator.free_group_end()
                    self.assertEqual(allocator.available_size(), 4 * p)
                    reused = allocator.alloc(4 * p)
                    self.assertEqual(
                        reused.cpu().sort().values.tolist(), list(range(p, 5 * p))
                    )
                    allocator.free_segment(reused[: 2 * p], start_pos=0)
                    self.assertEqual(allocator.available_size(), 2 * p)
                    again = allocator.alloc(2 * p)
                    self.assertEqual(
                        again.cpu().sort().values.tolist(),
                        reused[: 2 * p].cpu().sort().values.tolist(),
                    )


if __name__ == "__main__":
    unittest.main()
