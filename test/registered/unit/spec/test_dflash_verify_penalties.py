"""DFLASH and DSPARK must honor repetition/frequency/presence penalties and
min_tokens, which they silently ignored (sgl-project/sglang#46752)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.sampling.penaltylib.frequency_penalty import (  # noqa: E402
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (  # noqa: E402
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import (  # noqa: E402
    BatchedRepetitionPenalizer,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo  # noqa: E402
from sglang.srt.speculative.dflash_utils import (  # noqa: E402
    apply_dflash_verify_logits_adjustments,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_verify import (  # noqa: E402
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_utils import spec_prepare_for_decode  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB_SIZE = 8
DRAFT_TOKEN_NUM = 3
REPETITION_PENALTIES = (2.0, 1.0)
FREQUENCY_PENALTIES = (0.5, 1.5)
LAST_OUTPUT_TOKENS = (5, 1)


def _make_req(repetition_penalty, frequency_penalty, last_output_token):
    req = MagicMock()
    req.sampling_params.repetition_penalty = repetition_penalty
    req.sampling_params.frequency_penalty = frequency_penalty
    req.output_ids = [last_output_token]
    req.origin_input_ids = [0]
    return req


def _make_penalized_batch():
    reqs = [
        _make_req(*params)
        for params in zip(REPETITION_PENALTIES, FREQUENCY_PENALTIES, LAST_OUTPUT_TOKENS)
    ]
    batch = ScheduleBatch(reqs=reqs)
    batch.device = "cpu"
    orchestrator = BatchedPenalizerOrchestrator(
        VOCAB_SIZE,
        batch,
        {BatchedRepetitionPenalizer, BatchedFrequencyPenalizer},
    )
    bs = len(reqs)
    batch.sampling_info = SamplingBatchInfo(
        temperatures=torch.ones(bs, 1),
        top_ps=torch.ones(bs),
        top_ks=torch.ones(bs, dtype=torch.int32),
        min_ps=torch.zeros(bs),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB_SIZE,
        penalizer_orchestrator=orchestrator,
        device="cpu",
    )
    return batch


def _verify_logits(bs):
    generator = torch.Generator().manual_seed(0)
    return torch.randn(bs * DRAFT_TOKEN_NUM, VOCAB_SIZE, generator=generator)


class TestDFlashVerifyAppliesPenalties(CustomTestCase):
    def _assert_matches_spec_penalty_semantics(self, batch, sampling_info):
        orchestrator = batch.sampling_info.penalizer_orchestrator
        orchestrator.cumulate_output_tokens(torch.tensor(LAST_OUTPUT_TOKENS))
        logits = _verify_logits(len(batch.reqs))
        expected = logits.clone()
        orchestrator.apply(expected, repeat=DRAFT_TOKEN_NUM)
        self.assertFalse(torch.equal(expected, logits))

        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=sampling_info(batch.sampling_info),
            draft_token_num=DRAFT_TOKEN_NUM,
        )

        torch.testing.assert_close(logits, expected)

    def test_scheduler_handoff_penalties_reach_every_verify_row(self):
        """Every verify row gets its request's penalties from the
        copy_for_forward() sampling info the scheduler hands the worker."""
        self._assert_matches_spec_penalty_semantics(
            _make_penalized_batch(), lambda info: info.copy_for_forward()
        )

    def test_live_penalizer_keeps_multiplicative_penalties(self):
        """repetition_penalty survives when a live penalizer reaches verify."""
        self._assert_matches_spec_penalty_semantics(
            _make_penalized_batch(), lambda info: info
        )


class TestDFlashFastPathsSeePenalties(CustomTestCase):
    def test_noop_check_sees_folded_penalties(self):
        penalized = _make_penalized_batch().sampling_info.copy_for_forward()
        self.assertFalse(verify_logits_adjustments_are_noop(penalized))

        plain = _make_penalized_batch().sampling_info
        plain.penalizer_orchestrator.is_required = False
        self.assertTrue(verify_logits_adjustments_are_noop(plain.copy_for_forward()))

    def test_greedy_argmax_skip_refused_with_folded_penalties(self):
        """A penalized greedy batch must not take the argmax before penalties."""
        penalized = _make_penalized_batch().sampling_info.copy_for_forward()
        worker = SimpleNamespace(_selector_sample=None)
        batch = SimpleNamespace(has_grammar=False, return_logprob=False)

        self.assertFalse(
            DFlashWorkerV2._can_skip_verify_logits(worker, batch, penalized)
        )


class TestDFlashDecodePrepCumulatesPenalties(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            mamba_radix_cache_strategy="no_buffer"
        )
        override.install()
        self.addCleanup(override.restore)

    def _frequency_state(self, batch):
        penalizers = batch.sampling_info.penalizer_orchestrator.penalizers
        return penalizers[BatchedFrequencyPenalizer].cumulated_frequency_penalties

    def _expected_frequency_state(self, steps):
        expected = torch.zeros(len(LAST_OUTPUT_TOKENS), VOCAB_SIZE)
        for row, (token, penalty) in enumerate(
            zip(LAST_OUTPUT_TOKENS, FREQUENCY_PENALTIES)
        ):
            expected[row, token] = penalty * steps
        return expected

    def test_dflash_family_cumulates_once_per_decode_step(self):
        batch = _make_penalized_batch()
        batch.spec_algorithm = SimpleNamespace(
            is_dflash_family=lambda: True, is_uno=lambda: False
        )
        batch.spec_info = SimpleNamespace(prepare_for_decode=MagicMock())

        for _ in range(2):
            spec_prepare_for_decode(batch)

        torch.testing.assert_close(
            self._frequency_state(batch), self._expected_frequency_state(steps=2)
        )
        self.assertEqual(batch.spec_info.prepare_for_decode.call_count, 2)

    def test_eagle_prep_stays_the_only_cumulation_site_for_eagle(self):
        """EAGLE penalty counters advance once per step, not twice."""
        batch = _make_penalized_batch()
        batch.spec_algorithm = SimpleNamespace(
            is_dflash_family=lambda: False, is_uno=lambda: False
        )

        with patch(
            "sglang.srt.speculative.eagle_utils.eagle_prepare_for_decode"
        ) as eagle_prepare:
            spec_prepare_for_decode(batch)

        eagle_prepare.assert_called_once_with(batch)
        torch.testing.assert_close(
            self._frequency_state(batch), self._expected_frequency_state(steps=0)
        )


if __name__ == "__main__":
    unittest.main()
