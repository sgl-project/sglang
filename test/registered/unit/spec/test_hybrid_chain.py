import math
import unittest

import torch

from sglang.srt.speculative.hybrid_chain import (
    HybridChainState,
    parse_position_thresholds,
    splice_hybrid_chain,
)
from sglang.srt.speculative.hybrid_ragged import (
    build_hybrid_ragged_tier_grid,
    compute_hybrid_verify_lens,
    plan_hybrid_ragged_tier,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridThresholds(CustomTestCase):
    def test_parse_per_verify_column(self):
        self.assertEqual(
            parse_position_thresholds("off,off,0.40,0.55", 0.4, 3),
            [0.0, 0.4, 0.55],
        )
        self.assertEqual(parse_position_thresholds("", 0.4, 3), [0.0, 0.4, 0.4])

    def test_rejects_thresholds_for_non_draft_columns(self):
        for value in ("0.1,off,0.4,0.5", "off,0.1,0.4,0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_position_thresholds(value, 0.4, 3)

    def test_rejects_wrong_width_and_range(self):
        for value in ("off,off,0.4", "off,off,0.4,1.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_position_thresholds(value, 0.4, 3)


class TestHybridChainSplice(CustomTestCase):
    def setUp(self):
        self.mtp = torch.tensor(
            [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]
        )
        self.retrieval = torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [20, 21, 99, 98, 97, 96],
                [30, 77, 76, 75, 74, 73],
                [40, 0, 0, 0, 0, 0],
            ]
        )
        self.retrieval_lens = torch.tensor([5, 5, 5, 0])
        self.step_logprobs = torch.log(
            torch.tensor([[0.9, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])
        )
        self.thresholds = torch.log(torch.tensor([0.4, 0.55]))

    def test_splices_only_after_prefix_agreement(self):
        spliced, kept, agreed, grafted = splice_hybrid_chain(
            mtp_draft_tokens=self.mtp,
            step_logprobs=self.step_logprobs,
            retrieval_chains=self.retrieval,
            retrieval_lens=self.retrieval_lens,
            tail_log_thresholds=self.thresholds,
            verify_width=6,
        )

        self.assertEqual(kept.tolist(), [3, 1, 2, 3])
        self.assertEqual(agreed.tolist(), [3, 1, 0, 0])
        self.assertEqual(grafted.tolist(), [True, True, False, False])
        self.assertEqual(
            spliced.tolist(),
            [
                [11, 12, 13, 14, 15],
                [21, 99, 98, 97, 96],
                [31, 32, 33, 0, 0],
                [41, 42, 43, 0, 0],
            ],
        )

    def test_failed_graft_preserves_full_mtp_chain(self):
        spliced, kept, _, grafted = splice_hybrid_chain(
            mtp_draft_tokens=self.mtp,
            step_logprobs=self.step_logprobs,
            retrieval_chains=None,
            retrieval_lens=None,
            tail_log_thresholds=self.thresholds,
            verify_width=6,
        )

        self.assertFalse(bool(grafted.any()))
        self.assertEqual(kept.tolist(), [3, 1, 2, 3])
        torch.testing.assert_close(spliced[:, :3], self.mtp)

    def test_token_zero_does_not_count_as_retrieval_padding(self):
        spliced, _, agreed, grafted = splice_hybrid_chain(
            mtp_draft_tokens=torch.tensor([[0]]),
            step_logprobs=None,
            retrieval_chains=torch.tensor([[7, 0, 0]]),
            retrieval_lens=torch.tensor([0]),
            tail_log_thresholds=torch.empty(0),
            verify_width=3,
        )

        self.assertEqual(agreed.item(), 0)
        self.assertFalse(grafted.item())
        self.assertEqual(spliced.tolist(), [[0, 0]])


class TestHybridRaggedLengths(CustomTestCase):
    def test_lengths_cover_mtp_floor_and_retrieval_tail(self):
        widths = compute_hybrid_verify_lens(
            num_keep_drafts=torch.tensor([3, 3, 3, 3]),
            graft_ok=torch.tensor([False, True, True, False]),
            num_retrieval_tokens=torch.tensor([3, 8, 4, 2]),
            num_steps=3,
            verify_width=9,
        )

        self.assertEqual(widths.dtype, torch.int32)
        self.assertEqual(widths.tolist(), [4, 9, 5, 4])

    def test_extension_ema_counts_only_retrieval_extensions(self):
        state = HybridChainState(
            thresholds=[0.0, 0.4, 0.55],
            tau_min=0.1,
            dynamic_tau=True,
            device="cpu",
        )
        state.last_num_keep_drafts = torch.tensor([1, 2, 1, 2])
        state.last_graft_ok = torch.tensor([True, True, False, False])
        state.update_extension_ema(
            num_correct_drafts=torch.tensor([2, 2, 3, 4]), keep_on_device=False
        )

        self.assertTrue(math.isclose(state.extension_ema, 0.0125))


class TestHybridRaggedTiers(CustomTestCase):
    def test_grid_tracks_each_batch_floor_and_ceiling(self):
        tiers = build_hybrid_ragged_tier_grid(
            capture_bs=[1, 2],
            verify_width=9,
            floor=4,
            tier_step=5,
        )

        self.assertEqual(tiers, [4, 8, 9, 13, 18])

    def test_plan_spends_slack_without_exceeding_request_width(self):
        plan = plan_hybrid_ragged_tier(
            accept_verify_lens=[4, 5],
            verify_width=9,
            tier_num_tokens=14,
        )

        self.assertEqual(plan.accept_verify_lens, [4, 5])
        self.assertEqual(plan.forward_verify_lens, [9, 5])
        self.assertEqual(sum(plan.forward_verify_lens), 14)


if __name__ == "__main__":
    unittest.main()
