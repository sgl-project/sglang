"""CPU-only tests for speculative verify penalties."""

from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.draft_penalties import apply_draft_token_penalties
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _sampling_info(**kwargs):
    defaults = dict(
        draft_presence_penalties=None,
        draft_frequency_penalties=None,
        draft_repetition_penalties=None,
        draft_cumulated_presence_penalties=None,
        draft_cumulated_repetition_penalties=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestDraftTokenPenalties(CustomTestCase):
    def test_presence_penalty_counts_each_prior_draft_token_once(self):
        candidates = torch.tensor([[1, 1, 2]])
        logits = torch.zeros(1, 3, 5)
        sampling_info = _sampling_info(
            draft_presence_penalties=torch.tensor([[1.5]]),
            draft_cumulated_presence_penalties=torch.zeros(1, 5),
        )

        apply_draft_token_penalties(logits, candidates, sampling_info)

        self.assertEqual(logits[0, 0].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(logits[0, 1].tolist(), [0, -1.5, 0, 0, 0])
        self.assertEqual(logits[0, 2].tolist(), [0, -1.5, 0, 0, 0])

    def test_frequency_penalty_counts_duplicate_draft_tokens(self):
        candidates = torch.tensor([[1, 1, 2]])
        logits = torch.zeros(1, 3, 5)
        sampling_info = _sampling_info(
            draft_frequency_penalties=torch.tensor([[0.5]]),
        )

        apply_draft_token_penalties(logits, candidates, sampling_info)

        self.assertEqual(logits[0, 1].tolist(), [0, -0.5, 0, 0, 0])
        self.assertEqual(logits[0, 2].tolist(), [0, -1.0, 0, 0, 0])

    def test_batch_rows_use_their_own_draft_tokens(self):
        candidates = torch.tensor([[1, 2], [3, 4]])
        logits = torch.zeros(2, 2, 5)
        sampling_info = _sampling_info(
            draft_presence_penalties=torch.tensor([[1.5], [2.0]]),
            draft_cumulated_presence_penalties=torch.zeros(2, 5),
        )

        apply_draft_token_penalties(logits, candidates, sampling_info)

        self.assertEqual(logits[0, 1].tolist(), [0, -1.5, 0, 0, 0])
        self.assertEqual(logits[1, 1].tolist(), [0, 0, 0, -2.0, 0])

    def test_repetition_penalty_is_sign_aware_and_deduplicated(self):
        candidates = torch.tensor([[1, 1, 2]])
        logits = torch.zeros(1, 3, 5)
        logits[0, 1, 1] = 2.0
        logits[0, 2, 1] = -2.0
        sampling_info = _sampling_info(
            draft_repetition_penalties=torch.tensor([[2.0]]),
            draft_cumulated_repetition_penalties=torch.ones(1, 5),
        )

        apply_draft_token_penalties(logits, candidates, sampling_info)

        self.assertAlmostEqual(logits[0, 1, 1].item(), 1.0)
        self.assertAlmostEqual(logits[0, 2, 1].item(), -4.0)

    def test_already_penalized_tokens_are_not_penalized_again(self):
        candidates = torch.tensor([[1, 2, 3]])
        logits = torch.zeros(1, 3, 5)
        base_presence = torch.zeros(1, 5)
        base_presence[0, 1] = 1.5
        base_presence[0, 2] = 1.5
        base_repetition = torch.ones(1, 5)
        base_repetition[0, 1] = 2.0
        base_repetition[0, 2] = 2.0
        sampling_info = _sampling_info(
            draft_presence_penalties=torch.tensor([[1.5]]),
            draft_repetition_penalties=torch.tensor([[2.0]]),
            draft_cumulated_presence_penalties=base_presence,
            draft_cumulated_repetition_penalties=base_repetition,
        )

        apply_draft_token_penalties(logits, candidates, sampling_info)

        self.assertEqual(logits[0, 1, 1].item(), 0.0)
        self.assertEqual(logits[0, 1, 2].item(), 0.0)
        self.assertEqual(logits[0, 2, 1].item(), 0.0)
        self.assertEqual(logits[0, 2, 2].item(), 0.0)


class TestPenaltyCumulation(CustomTestCase):
    def test_spec_step_cumulates_all_accepted_tokens(self):
        calls = []

        class FakeOrchestrator:
            def cumulate_output_tokens(self, output_ids, row_indices=None):
                calls.append((output_ids.tolist(), row_indices.tolist()))

        batch = object.__new__(ScheduleBatch)
        batch.device = "cpu"
        batch.reqs = [
            SimpleNamespace(
                output_ids=[10, 11],
                origin_input_ids=[1, 2, 3],
                penalty_cumulated_output_len=None,
            ),
            SimpleNamespace(
                output_ids=[],
                origin_input_ids=[4, 5],
                penalty_cumulated_output_len=None,
            ),
        ]
        batch.sampling_info = SimpleNamespace(penalizer_orchestrator=FakeOrchestrator())

        batch.cumulate_penalty_output_tokens()

        self.assertEqual(calls, [([10, 5], [0, 1]), ([11], [0])])

    def test_later_step_only_cumulates_new_tokens(self):
        calls = []

        class FakeOrchestrator:
            def cumulate_output_tokens(self, output_ids, row_indices=None):
                calls.append((output_ids.tolist(), row_indices.tolist()))

        batch = object.__new__(ScheduleBatch)
        batch.device = "cpu"
        batch.reqs = [
            SimpleNamespace(
                output_ids=[10, 11, 12],
                origin_input_ids=[1, 2, 3],
                penalty_cumulated_output_len=2,
            ),
            SimpleNamespace(
                output_ids=[7],
                origin_input_ids=[4, 5],
                penalty_cumulated_output_len=0,
            ),
        ]
        batch.sampling_info = SimpleNamespace(penalizer_orchestrator=FakeOrchestrator())

        batch.cumulate_penalty_output_tokens()

        self.assertEqual(calls, [([12, 7], [0, 1])])

    def test_external_cumulation_skips_stale_output_ids(self):
        calls = []

        class FakeOrchestrator:
            def cumulate_output_tokens(self, output_ids, row_indices=None):
                calls.append((output_ids.tolist(), row_indices.tolist()))

        batch = object.__new__(ScheduleBatch)
        batch.device = "cpu"
        batch.reqs = [
            SimpleNamespace(
                output_ids=[10, 11],
                origin_input_ids=[1, 2, 3],
                penalty_cumulated_output_len=0,
                penalty_external_cumulation=True,
            )
        ]
        batch.sampling_info = SimpleNamespace(penalizer_orchestrator=FakeOrchestrator())

        batch.cumulate_penalty_output_tokens()

        self.assertEqual(calls, [])


class TestAcceptedTokenCumulation(CustomTestCase):
    def test_accept_lens_mask_each_verify_position(self):
        calls = []

        class FakeOrchestrator:
            is_required = True

            def cumulate_output_tokens_masked(self, output_ids, mask):
                calls.append((output_ids.tolist(), mask.tolist()))

        sampling_info = object.__new__(SamplingBatchInfo)
        sampling_info.penalizer_orchestrator = FakeOrchestrator()

        sampling_info.cumulate_accepted_output_tokens(
            torch.tensor([[10, 11, 12], [20, 21, 22]]),
            torch.tensor([2, 1]),
        )

        self.assertEqual(
            calls,
            [
                ([10, 20], [True, True]),
                ([11, 21], [True, False]),
                ([12, 22], [False, False]),
            ],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
