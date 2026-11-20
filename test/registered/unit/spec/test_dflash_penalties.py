"""Sampling penalties under DFLASH-family speculative decoding — no server, no model."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.penaltylib.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
)
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 8
BLOCK = 3
BS = 2


class _ForwardSamplingInfo(SimpleNamespace):
    """The forward-only copy of SamplingBatchInfo seen by the verify step."""

    def __init__(self, **fields):
        values = dict(
            has_custom_logit_processor=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            penalizer_orchestrator=None,
            grammar_mask=None,
            logit_bias=None,
        )
        values.update(fields)
        super().__init__(**values)

    def __len__(self):
        return BS


class _BanTokens:
    """Grammar mask stand-in: the same tokens are banned for every request."""

    def __init__(self, token_ids):
        self.token_ids = token_ids

    def apply(self, logits):
        logits[:, self.token_ids] = float("-inf")


def _make_req(freq=0.0, repetition=1.0, output_ids=()):
    req = MagicMock()
    req.sampling_params.frequency_penalty = freq
    req.sampling_params.repetition_penalty = repetition
    req.output_ids = array("q", output_ids)
    req.penalizer_cumulated_len = 0
    return req


def _orchestrator(reqs):
    batch = MagicMock()
    batch.reqs = reqs
    batch.device = "cpu"
    return BatchedPenalizerOrchestrator(
        VOCAB_SIZE, batch, {BatchedFrequencyPenalizer, BatchedRepetitionPenalizer}
    )


def _verify_logits():
    # Mixed signs, so the scaling penalty divides some logits and multiplies others.
    return torch.randn(
        BS * BLOCK, VOCAB_SIZE, generator=torch.Generator().manual_seed(0)
    )


def _penalized(logits, additive, scaling):
    """Reference: apply request i's penalties to each row of its verify block."""
    rows = logits.clone().view(BS, BLOCK, VOCAB_SIZE) + additive[:, None, :]
    rows = torch.where(rows < 0, rows * scaling[:, None, :], rows / scaling[:, None, :])
    return rows.view(BS * BLOCK, VOCAB_SIZE)


class TestDFlashVerifyPenalties(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.additive = torch.zeros(BS, VOCAB_SIZE)
        self.additive[0, 1], self.additive[1, 4] = -2.0, -3.0
        self.scaling = torch.ones(BS, VOCAB_SIZE)
        self.scaling[0, 2], self.scaling[1, 5] = 1.5, 2.0

    def test_accumulated_penalties_reach_every_row_of_the_block(self):
        logits = _verify_logits()
        expected = _penalized(logits, self.additive, self.scaling)
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=_ForwardSamplingInfo(
                acc_additive_penalties=self.additive,
                acc_scaling_penalties=self.scaling,
            ),
            draft_token_num=BLOCK,
        )
        torch.testing.assert_close(logits, expected)

    def test_scaling_penalty_is_kept_under_a_grammar_mask(self):
        logits = _verify_logits()
        expected = _penalized(logits, self.additive, self.scaling)
        expected[:, 7] = float("-inf")
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=_ForwardSamplingInfo(
                acc_additive_penalties=self.additive,
                acc_scaling_penalties=self.scaling,
                grammar_mask=_BanTokens([7]),
            ),
            draft_token_num=BLOCK,
        )
        torch.testing.assert_close(logits, expected)

    def test_live_orchestrator_is_used_without_accumulated_buffers(self):
        orch = _orchestrator([_make_req(freq=0.5), _make_req(repetition=2.0)])
        orch.cumulate_output_tokens(torch.tensor([3, 6]))
        logits = _verify_logits()
        expected = logits.clone().view(BS, BLOCK, VOCAB_SIZE)
        for position in range(BLOCK):
            rows = expected[:, position, :].clone()
            orch.apply(rows)
            expected[:, position, :] = rows
        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=_ForwardSamplingInfo(penalizer_orchestrator=orch),
            draft_token_num=BLOCK,
        )
        torch.testing.assert_close(logits, expected.view(BS * BLOCK, VOCAB_SIZE))

    def test_dspark_fast_path_respects_accumulated_penalties(self):
        self.assertTrue(verify_logits_adjustments_are_noop(_ForwardSamplingInfo()))
        self.assertFalse(
            verify_logits_adjustments_are_noop(
                _ForwardSamplingInfo(acc_additive_penalties=self.additive)
            )
        )
        self.assertFalse(
            verify_logits_adjustments_are_noop(
                _ForwardSamplingInfo(acc_scaling_penalties=self.scaling)
            )
        )


class TestCommittedOutputTokens(CustomTestCase):
    def _batch(self, reqs):
        orch = _orchestrator(reqs)
        batch = SimpleNamespace(
            reqs=reqs,
            device="cpu",
            sampling_info=SimpleNamespace(penalizer_orchestrator=orch),
        )
        return batch, orch.penalizers[BatchedFrequencyPenalizer]

    def test_each_committed_token_is_fed_once(self):
        reqs = [
            _make_req(freq=1.0, output_ids=[5, 5, 2]),
            _make_req(freq=1.0, output_ids=[3]),
        ]
        batch, freq = self._batch(reqs)

        ScheduleBatch.cumulate_penalty_committed_output_tokens(batch)
        counts = freq.cumulated_frequency_penalties
        self.assertEqual(counts[0, [5, 2]].tolist(), [2.0, 1.0])
        self.assertEqual(counts[1, 3].item(), 1.0)
        self.assertEqual(counts[1, 0].item(), 0.0)  # padding id
        self.assertEqual([r.penalizer_cumulated_len for r in reqs], [3, 1])

        # Next step: row 0 commits nothing, row 1 commits a block of four.
        reqs[1].output_ids.extend([3, 4, 4, 4])
        ScheduleBatch.cumulate_penalty_committed_output_tokens(batch)
        self.assertEqual(counts[0, [5, 2]].tolist(), [2.0, 1.0])
        self.assertEqual(counts[1, [3, 4]].tolist(), [2.0, 3.0])
        self.assertEqual([r.penalizer_cumulated_len for r in reqs], [3, 5])

        # Nothing new anywhere: no change.
        ScheduleBatch.cumulate_penalty_committed_output_tokens(batch)
        self.assertEqual(counts.sum().item(), 8.0)

    def test_dflash_decode_prep_feeds_committed_tokens_when_penalized(self):
        exec_ns = SimpleNamespace(
            mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=False)
        )
        for is_dflash_family, is_required, fed in (
            (True, True, True),
            (True, False, False),
        ):
            batch = MagicMock()
            batch.spec_algorithm.is_dflash_family.return_value = is_dflash_family
            batch.sampling_info.penalizer_orchestrator.is_required = is_required
            with patch.object(spec_utils, "get_exec", return_value=exec_ns):
                spec_utils.spec_prepare_for_decode(batch)
            self.assertEqual(batch.cumulate_penalty_committed_output_tokens.called, fed)
            batch.spec_info.prepare_for_decode.assert_called_once_with(batch)


if __name__ == "__main__":
    unittest.main()
