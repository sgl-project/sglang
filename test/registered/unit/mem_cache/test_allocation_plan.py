from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from typing import Any, List, Optional

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.allocation import (  # noqa: E402
    _decode_write_positions,
    _plan_decode_alloc,
    _plan_extend_alloc,
)


def _make_req(allocated_old: Optional[int]) -> Any:
    kv = (
        None
        if allocated_old is None
        else types.SimpleNamespace(kv_allocated_len=allocated_old)
    )
    return types.SimpleNamespace(kv=kv)


def _make_reqs(allocated_olds: List[Optional[int]]) -> List[Any]:
    return [_make_req(allocated_old) for allocated_old in allocated_olds]


def _plan_extend(
    *, reqs: List[Any], prefix_lens: List[int], seq_lens: List[int], page_size: int
) -> Any:
    return _plan_extend_alloc(
        reqs=reqs,
        prefix_lens_cpu=torch.tensor(prefix_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        page_size=page_size,
    )


class TestPlanExtendAlloc(CustomTestCase):
    def test_fresh_req_allocates_from_prefix_len_to_the_ceiling_of_seq_len(self):
        """A never-allocated request starts at its radix prefix and rounds the tail page up."""
        plan = _plan_extend(
            reqs=_make_reqs([None]),
            prefix_lens=[8],
            seq_lens=[21],
            page_size=4,
        )

        self.assertEqual(plan.alloc_starts_cpu.tolist(), [8])
        self.assertEqual(plan.alloc_ends_cpu.tolist(), [24])
        self.assertEqual(plan.need_size, 16)

    def test_chunked_continuation_starts_at_the_watermark_not_the_prefix(self):
        """Restarting at prefix_len would re-allocate pages the previous chunk still owns."""
        plan = _plan_extend(
            reqs=_make_reqs([12]),
            prefix_lens=[8],
            seq_lens=[21],
            page_size=4,
        )

        self.assertEqual(plan.alloc_starts_cpu.tolist(), [12])
        self.assertEqual(plan.alloc_ends_cpu.tolist(), [24])
        self.assertEqual(plan.need_size, 12)

    def test_watermark_never_shrinks_after_speculative_over_allocation(self):
        """A target of plain ceil(seq_len) would drop the spec-reserved tail out of tracking."""
        plan = _plan_extend(
            reqs=_make_reqs([64]),
            prefix_lens=[8],
            seq_lens=[21],
            page_size=4,
        )

        self.assertEqual(plan.alloc_starts_cpu.tolist(), [64])
        self.assertEqual(plan.alloc_ends_cpu.tolist(), [64])
        self.assertEqual(plan.need_size, 0)

    def test_every_per_req_need_is_a_whole_number_of_pages(self):
        """alloc() lays pages out back to back, so one ragged need shifts every later req off-page."""
        plan = _plan_extend(
            reqs=_make_reqs([None, 12, 64, 4]),
            prefix_lens=[8, 8, 8, 4],
            seq_lens=[21, 21, 21, 5],
            page_size=4,
        )

        needs = (plan.alloc_ends_cpu - plan.alloc_starts_cpu).tolist()
        self.assertEqual(needs, [16, 12, 0, 4])
        for index, need in enumerate(needs):
            with self.subTest(req=index):
                self.assertEqual(need % 4, 0)

    def test_unaligned_prefix_len_is_rejected(self):
        """The radix cache promises page-aligned prefixes; a ragged one breaks every downstream slice."""
        with self.assertRaises(AssertionError):
            _plan_extend(
                reqs=_make_reqs([None]),
                prefix_lens=[6],
                seq_lens=[21],
                page_size=4,
            )

    def test_page_size_one_allocates_exactly_the_extend_range(self):
        """Page size 1 must reproduce the unpaged token-slot semantics exactly."""
        plan = _plan_extend(
            reqs=_make_reqs([None, 8]),
            prefix_lens=[8, 8],
            seq_lens=[21, 21],
            page_size=1,
        )

        self.assertEqual(plan.alloc_starts_cpu.tolist(), [8, 8])
        self.assertEqual(plan.alloc_ends_cpu.tolist(), [21, 21])
        self.assertEqual(plan.need_size, 26)


class TestPlanDecodeAlloc(CustomTestCase):
    def test_step_inside_the_current_page_allocates_nothing(self):
        """page_size - 1 of every page_size decode steps must not touch the allocator at all."""
        plan = _plan_decode_alloc(
            reqs=_make_reqs([12]),
            locs_cpu=torch.tensor([9], dtype=torch.int64),
            token_per_req=1,
            page_size=4,
        )

        self.assertEqual(plan.alloc_starts_cpu.tolist(), [12])
        self.assertEqual(plan.alloc_ends_cpu.tolist(), [12])
        self.assertEqual(plan.need_size, 0)

    def test_step_crossing_the_page_boundary_allocates_exactly_one_page(self):
        """The step that fills the last slot of a page must open the next one, and only one."""
        plan = _plan_decode_alloc(
            reqs=_make_reqs([12]),
            locs_cpu=torch.tensor([12], dtype=torch.int64),
            token_per_req=1,
            page_size=4,
        )

        self.assertEqual(plan.alloc_starts_cpu.tolist(), [12])
        self.assertEqual(plan.alloc_ends_cpu.tolist(), [16])
        self.assertEqual(plan.need_size, 4)

    def test_token_per_req_larger_than_a_page_allocates_every_page_it_spans(self):
        """A spec verify block wider than a page must not stop at the first new page."""
        plan = _plan_decode_alloc(
            reqs=_make_reqs([12]),
            locs_cpu=torch.tensor([12], dtype=torch.int64),
            token_per_req=9,
            page_size=4,
        )

        self.assertEqual(plan.alloc_ends_cpu.tolist(), [24])
        self.assertEqual(plan.need_size, 12)

    def test_watermark_never_shrinks_below_a_reserved_allocation(self):
        """Decode following a spec reservation must keep the reserved tail, not free it silently."""
        plan = _plan_decode_alloc(
            reqs=_make_reqs([64]),
            locs_cpu=torch.tensor([12], dtype=torch.int64),
            token_per_req=1,
            page_size=4,
        )

        self.assertEqual(plan.alloc_ends_cpu.tolist(), [64])
        self.assertEqual(plan.need_size, 0)

    def test_unaligned_watermark_is_rejected(self):
        """Every write domain starts at the watermark; a ragged one puts blocks across two pages."""
        with self.assertRaises(AssertionError):
            _plan_decode_alloc(
                reqs=_make_reqs([13]),
                locs_cpu=torch.tensor([13], dtype=torch.int64),
                token_per_req=1,
                page_size=4,
            )

    def test_mixed_batch_needs_are_each_a_whole_number_of_pages(self):
        """One ragged need would shift every later request's slice of new_pages off its page."""
        plan = _plan_decode_alloc(
            reqs=_make_reqs([12, 16, 64]),
            locs_cpu=torch.tensor([9, 16, 12], dtype=torch.int64),
            token_per_req=1,
            page_size=4,
        )

        needs = (plan.alloc_ends_cpu - plan.alloc_starts_cpu).tolist()
        self.assertEqual(needs, [0, 4, 0])
        self.assertEqual(plan.need_size, 4)


class TestDecodeWritePositions(CustomTestCase):
    def test_encoder_decoder_positions_include_the_encoder_offset(self):
        """The paged path used to size against seq_lens while writing at encoder_lens + seq_lens."""
        batch = types.SimpleNamespace(
            model_config=types.SimpleNamespace(is_encoder_decoder=True),
            encoder_lens_cpu=[6, 0],
            encoder_lens=torch.tensor([6, 0], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([9, 9], dtype=torch.int64),
            seq_lens=torch.tensor([9, 9], dtype=torch.int64),
        )

        locs_cpu, locs_device = _decode_write_positions(batch)

        self.assertEqual(locs_cpu.tolist(), [15, 9])
        self.assertEqual(locs_device.tolist(), [15, 9])

    def test_decoder_only_positions_are_the_sequence_lengths(self):
        """A decoder-only model must not pay for the encoder-decoder host tensor build."""
        batch = types.SimpleNamespace(
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            seq_lens_cpu=torch.tensor([9, 4], dtype=torch.int64),
            seq_lens=torch.tensor([9, 4], dtype=torch.int64),
        )

        locs_cpu, locs_device = _decode_write_positions(batch)

        self.assertEqual(locs_cpu.tolist(), [9, 4])
        self.assertEqual(locs_device.tolist(), [9, 4])


if __name__ == "__main__":
    unittest.main()
