"""The scheduler's penalty accounting for chain speculative decoding."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.test_utils import CustomTestCase


def _batch(mode, *, tracks=True, required=True, **fields):
    orchestrator = MagicMock(is_required=required)
    return SimpleNamespace(
        forward_mode=mode,
        spec_algorithm=SimpleNamespace(
            tracks_speculative_penalty_history=lambda: tracks
        ),
        sampling_info=SimpleNamespace(penalizer_orchestrator=orchestrator),
        **fields,
    )


class TestSpeculativePenaltyAccounting(CustomTestCase):
    def test_only_tracking_decode_and_mixed_batches_with_penalties_count(self):
        select = Scheduler._speculative_penalty_orchestrator
        for mode in (ForwardMode.DECODE, ForwardMode.MIXED):
            batch = _batch(mode)
            self.assertIs(select(batch), batch.sampling_info.penalizer_orchestrator)
        for batch in (
            _batch(ForwardMode.EXTEND),
            _batch(ForwardMode.IDLE),
            _batch(ForwardMode.DECODE, tracks=False),
            _batch(ForwardMode.DECODE, required=False),
        ):
            self.assertIsNone(select(batch))
        self.assertIsNone(
            select(SimpleNamespace(sampling_info=None, forward_mode=ForwardMode.DECODE))
        )

    def test_decode_counts_the_anchors(self):
        bonus = torch.tensor([4, 9])
        batch = _batch(
            ForwardMode.DECODE, spec_info=SimpleNamespace(bonus_tokens=bonus)
        )
        penalties = batch.sampling_info.penalizer_orchestrator
        Scheduler._cumulate_speculative_anchors(batch, penalties)
        (tokens, rows), _ = penalties.cumulate_output_tokens.call_args
        self.assertTrue(torch.equal(tokens, bonus))
        self.assertEqual(rows.tolist(), [0, 1])

    def test_mixed_batch_counts_only_the_running_rows(self):
        """Running requests follow the prefill rows as one-token rows whose input
        is their anchor; prompt tokens are not output."""
        batch = _batch(
            ForwardMode.MIXED,
            reqs=[object()] * 3,
            decoding_reqs=[object()] * 2,
            input_ids=torch.tensor([11, 12, 13, 5, 6]),
        )
        penalties = batch.sampling_info.penalizer_orchestrator
        Scheduler._cumulate_speculative_anchors(batch, penalties)
        (tokens, lens), _ = penalties.cumulate_padded_output_tokens.call_args
        self.assertEqual(lens.tolist(), [0, 1, 1])
        self.assertEqual(tokens[1:, 0].tolist(), [5, 6])

    def test_verify_counts_the_accepted_drafts_and_marks_the_commit(self):
        """Rows of the result's output stride, as the result processor commits
        them; rows past the batch are padding."""
        penalties = MagicMock()
        tokens = torch.tensor([7, 8, 3, -1, 2, 0, 0, 0, 4, 4, 4, 4])
        result = GenerationBatchResult(
            next_token_ids=tokens,
            accept_lens=torch.tensor([3, 1, 4], dtype=torch.int32),
            speculative_num_draft_tokens=4,
        )
        Scheduler._cumulate_speculative_drafts(penalties, result, 2)
        (counted, lens), _ = penalties.cumulate_padded_output_tokens.call_args
        self.assertEqual(counted.tolist(), [[7, 8, 3, -1], [2, 0, 0, 0]])
        # The bonus is the next step's anchor.
        self.assertEqual(lens.tolist(), [2, 0])
        penalties.record_commits.assert_called_once_with()
        result.num_non_draft_tokens_per_req = 2
        with self.assertRaises(AssertionError):
            Scheduler._cumulate_speculative_drafts(penalties, result, 2)

    def test_extend_result_only_marks_the_commit(self):
        penalties = MagicMock()
        result = GenerationBatchResult(next_token_ids=torch.tensor([1, 2]))
        Scheduler._cumulate_speculative_drafts(penalties, result, 2)
        penalties.cumulate_padded_output_tokens.assert_not_called()
        penalties.cumulate_output_tokens.assert_not_called()
        penalties.record_commits.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
