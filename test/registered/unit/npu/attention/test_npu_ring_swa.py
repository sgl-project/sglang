"""Per-request prompt+ring KV regions for --uocr-ring-swa."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.ring_swa import (
    RingRegionAllocator,
    attn_len,
    extend_locs,
    ring_write_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE = 128
WINDOW = 128


class TestRingRegionAllocator(CustomTestCase):
    def _allocator(self, pages: int = 64) -> RingRegionAllocator:
        return RingRegionAllocator(page_size=PAGE, window=WINDOW, total_pages=pages)

    def test_region_is_sized_by_the_prompt_not_the_context(self):
        """The ring holds the only generated KV, so R + window is the need."""
        regions = self._allocator()
        self.assertEqual(regions.pages_for(1000), 9)  # ceil(1128 / 128)
        self.assertEqual(regions.pages_for(1517), 13)  # ceil(1645 / 128)
        self.assertEqual(regions.pages_for(0), 1)  # window only

    def test_runs_are_page_aligned_and_packed(self):
        """First fit: the second run starts right after the first one."""
        regions = self._allocator()
        first = regions.alloc(1000)  # 9 pages
        second = regions.alloc(1000)
        self.assertEqual(first, 0)
        self.assertEqual(second, 9 * PAGE)
        self.assertEqual(regions.available_tokens(), (64 - 18) * PAGE)

    def test_freed_run_is_merged_and_reusable(self):
        """Release returns the pages to the interval list and merges neighbours."""
        regions = self._allocator()
        first = regions.alloc(2000)  # 17 pages
        second = regions.alloc(2000)
        regions.free_run(first)
        regions.free_run(second)
        self.assertEqual(regions.free, [(0, 64)])
        self.assertEqual(regions.available_tokens(), 64 * PAGE)
        self.assertEqual(regions.alloc(5000), 0)  # 41 pages, fits again

    def test_out_of_regions_reports_none(self):
        """A request that cannot fit gets None; the caller raises prefill OOM."""
        regions = self._allocator(pages=8)
        self.assertIsNotNone(regions.alloc(500))  # 5 pages
        self.assertIsNone(regions.alloc(500))  # only 3 pages left

    def test_releasing_the_same_run_twice_is_a_no_op(self):
        """A double release must not hand the same pages out twice."""
        regions = self._allocator()
        base = regions.alloc(1000)
        regions.free_run(base)
        regions.free_run(base)
        self.assertEqual(regions.available_tokens(), 64 * PAGE)
        self.assertEqual(regions.free, [(0, 64)])


class TestRingSlots(CustomTestCase):
    def test_extend_locs_stay_linear_inside_each_region(self):
        """Prefill writes base + position for every request in the batch."""
        locs = extend_locs(
            bases=torch.tensor([0, 4 * PAGE]),
            prefix_lens=torch.tensor([0, 5]),
            seq_lens=torch.tensor([3, 7]),
            page_size=PAGE,
            device="cpu",
        )
        self.assertEqual(locs.tolist(), [0, 1, 2, 4 * PAGE + 5, 4 * PAGE + 6])

    def test_write_slots_wrap_in_place_after_the_window(self):
        """Token t lands on base + R + (t mod W); nothing is ever moved."""
        region_base = 2 * PAGE
        positions = region_base + torch.arange(10, 20)  # t = 0..9, prompt of 10
        slots = ring_write_slots(
            cache_loc=positions,
            region_base=torch.tensor(region_base),
            prompt_lens=torch.tensor([10]),
            window=4,
        )
        self.assertEqual(
            slots.tolist(),
            [
                region_base + 10,
                region_base + 11,
                region_base + 12,
                region_base + 13,
                region_base + 10,
                region_base + 11,
                region_base + 12,
                region_base + 13,
                region_base + 10,
                region_base + 11,
            ],
        )

    def test_attn_len_saturates_at_prompt_plus_window(self):
        """Attention never reads past one window of generated tokens."""
        got = attn_len(
            prompt_lens=torch.tensor([10, 10]),
            seq_lens=torch.tensor([12, 500]),
            window=4,
        )
        self.assertEqual(got.tolist(), [12, 14])


if __name__ == "__main__":
    unittest.main()
