import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    TboForwardBatchPreparer,
    _compute_cp_aligned_extend_split_plan,
)
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTboCpAlignedSplit(CustomTestCase):
    def test_round_robin_split_commutes_with_mhc_repeat(self):
        embedding = torch.arange(32 * 7).reshape(32, 7)
        hc_mult = 4

        for cp_rank in range(4):
            strategy = SimpleNamespace(cp_size=4, cp_rank=cp_rank)
            shard = lambda x: InterleaveCPStrategy._interleave_shard(strategy, x)
            repeat_then_split = shard(embedding.unsqueeze(1).repeat(1, hc_mult, 1))
            split_then_repeat = shard(embedding).unsqueeze(1).repeat(1, hc_mult, 1)

            self.assertTrue(torch.equal(repeat_then_split, split_then_repeat))
            self.assertEqual(split_then_repeat.shape, (8, hc_mult, 7))
            self.assertTrue(split_then_repeat.is_contiguous())
            self.assertEqual(split_then_repeat.stride(), repeat_then_split.stride())

    def test_single_sequence_uses_aligned_half(self):
        self.assertEqual(
            _compute_cp_aligned_extend_split_plan([221], 224, 4),
            (0, 112, True),
        )
        self.assertEqual(
            _compute_cp_aligned_extend_split_plan([200], 200, 4),
            (0, 100, True),
        )

    def test_multi_sequence_can_split_inside_a_sequence(self):
        self.assertEqual(
            _compute_cp_aligned_extend_split_plan([214, 200], 416, 4),
            (0, 208, True),
        )
        self.assertEqual(
            _compute_cp_aligned_extend_split_plan([191, 192], 384, 4),
            (1, 192, True),
        )

    def test_sequence_boundary_is_preserved_when_aligned(self):
        self.assertEqual(
            _compute_cp_aligned_extend_split_plan([100, 100], 200, 4),
            (1, 100, False),
        )

    def test_odd_number_of_cp_blocks_stays_aligned(self):
        plan = _compute_cp_aligned_extend_split_plan([205, 207], 412, 4)
        self.assertEqual(plan, (0, 204, True))
        self.assertEqual(plan[1] % 4, 0)
        self.assertEqual((412 - plan[1]) % 4, 0)

    def test_degenerate_parent_falls_back(self):
        self.assertIsNone(_compute_cp_aligned_extend_split_plan([4], 4, 4))
        self.assertIsNone(_compute_cp_aligned_extend_split_plan([], 0, 4))

    def test_invalid_padding_falls_back(self):
        self.assertIsNone(_compute_cp_aligned_extend_split_plan([214, 200], 414, 4))
        self.assertIsNone(_compute_cp_aligned_extend_split_plan([214, 200], 400, 4))

    def test_cp_alignment_can_override_generic_sequence_split(self):
        batch = SimpleNamespace(
            tbo_split_seq_index=0,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[221],
            spec_info=None,
            input_ids=range(224),
        )
        parallel = SimpleNamespace(attn_dp_size=1, attn_tp_size=1, attn_cp_size=4)
        backend = SimpleNamespace(is_none=lambda: True)
        module = "sglang.srt.batch_overlap.two_batch_overlap"
        cp_utils = "sglang.srt.layers.attention.dsa.utils"

        with (
            patch.object(
                TboForwardBatchPreparer,
                "_compute_split_token_index",
                return_value=0,
            ),
            patch(f"{module}._is_two_chunk_split_enabled", return_value=False),
            patch(f"{module}.get_parallel", return_value=parallel),
            patch(f"{module}.get_moe_a2a_backend", return_value=backend),
            patch(
                f"{cp_utils}.is_dsa_prefill_cp_interleave",
                return_value=True,
            ),
        ):
            self.assertEqual(
                TboForwardBatchPreparer._compute_split_plan(batch),
                (0, 112, True),
            )


if __name__ == "__main__":
    unittest.main()
