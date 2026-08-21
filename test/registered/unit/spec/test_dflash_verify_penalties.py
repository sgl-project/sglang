"""Regression tests for issue #33493.

1. DFlash/DSpark verify must read the `acc_additive_penalties` field that
   `SamplingBatchInfo.copy_for_forward()` actually produces (overlap mode),
   not the stale `acc_linear_penalties` (the field's pre-#21258 name,
   removed in that PR).
2. Once those penalties reach the verify logits, the `min_new_tokens`
   stop-token mask has to lift. The dflash family never reaches the
   seen-token accounting in `ScheduleBatch.prepare_for_decode`, and it
   commits a variable-length accepted run per step, so its counter must
   track the tokens each request actually committed -- not the number of
   verify rounds.

No server, no model loading -- CPU only.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import dataclasses
import types
import unittest
from typing import Optional

import torch

from sglang.srt.managers import schedule_batch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
from sglang.srt.sampling.penaltylib.min_new_tokens import BatchedMinNewTokensPenalizer
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer
from sglang.srt.sampling.penaltylib.repetition_penalty import BatchedRepetitionPenalizer
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)
from sglang.test.test_utils import CustomTestCase

BS = 2
VOCAB = 8
DRAFT_TOKENS = 3
EOS = 0
MIN_NEW_TOKENS = 3


class _OverlapSamplingInfo:
    """Mimics the object handed to spec verify in overlap scheduling:
    `copy_for_forward()` folds the penalizer into `acc_additive_penalties`
    and strips `penalizer_orchestrator`."""

    def __init__(self, acc_additive_penalties):
        self.acc_additive_penalties = acc_additive_penalties
        self.acc_scaling_penalties = None
        self.penalizer_orchestrator = None
        self.has_custom_logit_processor = False
        self.grammar_mask = None
        self.logit_bias = None
        self.is_all_greedy = True

    def __len__(self):
        return BS


def _min_new_tokens_style_penalty():
    # min_new_tokens penalizer suppresses stop tokens with large negative bias.
    penalties = torch.zeros((BS, VOCAB), dtype=torch.float32)
    penalties[:, 0] = torch.finfo(torch.float32).min  # token 0 = EOS
    return penalties


class TestDFlashVerifyAdditivePenalties(CustomTestCase):
    def test_overlap_additive_penalties_are_applied(self):
        """In overlap mode the accumulated additive penalties (e.g.
        min_new_tokens EOS suppression) must reach the verify logits."""
        info = _OverlapSamplingInfo(_min_new_tokens_style_penalty())
        logits = torch.zeros((BS * DRAFT_TOKENS, VOCAB), dtype=torch.float32)

        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits,
            sampling_info=info,
            draft_token_num=DRAFT_TOKENS,
        )

        self.assertTrue(
            (logits[:, 0] < -1e30).all(),
            f"EOS suppression from acc_additive_penalties was not applied "
            f"to verify logits; got column 0 = {logits[:, 0].tolist()}",
        )
        # Other vocab entries untouched.
        self.assertTrue((logits[:, 1:] == 0).all())

    def test_noop_gate_detects_additive_penalties(self):
        """DSpark's greedy fold fast path must NOT be taken when additive
        penalties are pending."""
        info = _OverlapSamplingInfo(_min_new_tokens_style_penalty())
        self.assertFalse(
            verify_logits_adjustments_are_noop(info),
            "verify_logits_adjustments_are_noop wrongly reported noop while "
            "acc_additive_penalties is set (issue #33493)",
        )

    def test_noop_gate_true_when_clean(self):
        info = _OverlapSamplingInfo(None)
        self.assertTrue(verify_logits_adjustments_are_noop(info))

    def test_sampling_batch_info_field_contract(self):
        """The field speculative code must consume is `acc_additive_penalties`;
        the pre-#21258 name `acc_linear_penalties` is no longer a field, so
        `getattr(sampling_info, "acc_linear_penalties", None)` silently
        yields None."""
        fields = {f.name for f in dataclasses.fields(SamplingBatchInfo)}
        self.assertIn("acc_additive_penalties", fields)
        self.assertNotIn("acc_linear_penalties", fields)


class _FakeSamplingParams:
    def __init__(self, min_new_tokens: int, repetition_penalty: float = 1.5):
        self.min_new_tokens = min_new_tokens
        self.stop_token_ids: Optional[set] = None
        self.frequency_penalty = 0.5
        self.presence_penalty = 0.5
        self.repetition_penalty = repetition_penalty


class _FakeTokenizer:
    eos_token_id = EOS
    additional_stop_token_ids: Optional[set] = None


class _FakeReq:
    """Only the attributes the penalizers and the dflash decode hook read."""

    def __init__(self, min_new_tokens: int):
        self.sampling_params = _FakeSamplingParams(min_new_tokens)
        self.tokenizer = _FakeTokenizer()
        self.origin_input_ids = [7]
        self.output_ids = []


class _FakeDFlashBatch:
    """Only the attributes `spec_prepare_for_decode` touches on the dflash
    branch. The seen-token hook is the real ScheduleBatch method, so the
    counting itself is not faked."""

    sync_min_new_tokens_output_counts = ScheduleBatch.sync_min_new_tokens_output_counts
    cumulate_penalty_output_tokens = ScheduleBatch.cumulate_penalty_output_tokens

    def __init__(self, orchestrator):
        self.device = "cpu"
        self.reqs = list(orchestrator.reqs())
        self.sampling_info = types.SimpleNamespace(penalizer_orchestrator=orchestrator)
        self.spec_algorithm = types.SimpleNamespace(is_dflash_family=lambda: True)
        self.spec_info = types.SimpleNamespace(prepare_for_decode=lambda batch: None)


class _FakeReqHolder:
    """The orchestrator weakrefs its batch, so it cannot be a SimpleNamespace."""

    def __init__(self, reqs):
        self.reqs = reqs
        self.device = "cpu"


def _make_orchestrator(
    min_new_tokens: int, penalizers=None
) -> BatchedPenalizerOrchestrator:
    reqs = [_FakeReq(min_new_tokens) for _ in range(BS)]
    batch = _FakeReqHolder(reqs)
    orch = BatchedPenalizerOrchestrator(
        vocab_size=VOCAB,
        batch=batch,
        penalizers=penalizers or {BatchedMinNewTokensPenalizer},
    )
    # The orchestrator holds only a weakref to the batch.
    orch._keepalive_batch = batch
    return orch


def _fold_additive(orch) -> torch.Tensor:
    """What SamplingBatchInfo.update_penalties() hands to verify."""
    acc = torch.zeros((BS, VOCAB), dtype=torch.float32)
    orch.accumulate_additive_penalties(acc)
    return acc


def _counts(penalizer) -> list:
    return penalizer.len_output_tokens.flatten().tolist()


def _commit(batch, accept_lens):
    """Emulate BatchResultProcessor.process_batch_result_decode extending
    req.output_ids with the run _resolve_spec_v2_tokens settled."""
    for req, n in zip(batch.reqs, accept_lens):
        req.output_ids.extend([11] * n)


class TestDFlashMinNewTokensCounting(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._saved_get_server_args = spec_utils.get_server_args
        spec_utils.get_server_args = lambda: types.SimpleNamespace(
            enable_mamba_extra_buffer_lazy=lambda: False
        )

    def tearDown(self):
        spec_utils.get_server_args = self._saved_get_server_args
        super().tearDown()

    def test_ragged_accept_lengths_are_counted_exactly(self):
        """The counter must track committed tokens, not verify rounds: two
        rounds accepting [3, 1] then [2, 4] give [5, 5], not [2, 2]."""
        orch = _make_orchestrator(MIN_NEW_TOKENS)
        batch = _FakeDFlashBatch(orch)
        penalizer = orch.penalizers[BatchedMinNewTokensPenalizer]

        self.assertEqual(_counts(penalizer), [0, 0])

        _commit(batch, [3, 1])
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(
            _counts(penalizer),
            [3, 1],
            "min_new_tokens counter did not track the per-request accepted "
            "run length after round 1 (accept_lens [3, 1])",
        )

        _commit(batch, [2, 4])
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(
            _counts(penalizer),
            [5, 5],
            "min_new_tokens counter did not track the per-request accepted "
            "run length after round 2 (accept_lens [2, 4])",
        )

    def test_mask_lifts_exactly_at_min_new_tokens(self):
        """min_new_tokens=5: the mask must still be on at 4 committed tokens
        and off at 5."""
        orch = _make_orchestrator(5)
        batch = _FakeDFlashBatch(orch)
        penalizer = orch.penalizers[BatchedMinNewTokensPenalizer]

        self.assertEqual(_fold_additive(orch)[0, EOS].item(), float("-inf"))

        _commit(batch, [4, 4])
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(_counts(penalizer), [4, 4])
        acc = _fold_additive(orch)
        self.assertEqual(
            acc[:, EOS].tolist(),
            [float("-inf")] * BS,
            "EOS suppression lifted before min_new_tokens tokens were committed",
        )

        _commit(batch, [1, 1])
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(_counts(penalizer), [5, 5])
        acc = _fold_additive(orch)
        self.assertEqual(
            acc[:, EOS].tolist(),
            [0.0] * BS,
            f"min_new_tokens stop-token suppression never lifted after 5 "
            f"committed tokens; acc_additive_penalties = {acc.tolist()}",
        )
        self.assertTrue((acc == 0).all())

    def test_prefill_first_token_is_counted_once(self):
        """The single token prefill emits is counted exactly once, however many
        decode steps run before and after it is committed."""
        orch = _make_orchestrator(MIN_NEW_TOKENS)
        batch = _FakeDFlashBatch(orch)
        penalizer = orch.penalizers[BatchedMinNewTokensPenalizer]

        # In overlap the first decode step is prepared before the prefill
        # result has been processed: nothing is committed yet.
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(_counts(penalizer), [0, 0])

        _commit(batch, [1, 1])  # process_batch_result_prefill
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(_counts(penalizer), [1, 1])

        # Re-running the hook without new commits must not advance it again.
        spec_utils.spec_prepare_for_decode(batch)
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(
            _counts(penalizer),
            [1, 1],
            "the prefill token was counted more than once",
        )

    def test_grammar_truncated_run_counts_only_retained_tokens(self):
        """_resolve_spec_v2_tokens replaces the accepted run with the
        grammar-retained prefix before it is appended, so only the retained
        tokens are counted."""
        orch = _make_orchestrator(MIN_NEW_TOKENS)
        batch = _FakeDFlashBatch(orch)
        penalizer = orch.penalizers[BatchedMinNewTokensPenalizer]

        # accept_lens said [4, 4]; the grammar FSM terminated after 2 tokens on
        # row 0, so only 2 are committed there.
        _commit(batch, [2, 4])
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(
            _counts(penalizer),
            [2, 4],
            "grammar-truncated request counted the untruncated accept_len",
        )

    def test_uncommitted_request_is_not_counted(self):
        """A retracted or already-finished request commits nothing this step
        (process_batch_result_decode `continue`s past it), so its counter must
        not move."""
        orch = _make_orchestrator(MIN_NEW_TOKENS)
        batch = _FakeDFlashBatch(orch)
        penalizer = orch.penalizers[BatchedMinNewTokensPenalizer]

        _commit(batch, [2, 2])
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(_counts(penalizer), [2, 2])

        # Row 1 is retracted/finished: nothing appended for it.
        _commit(batch, [3, 0])
        spec_utils.spec_prepare_for_decode(batch)
        self.assertEqual(
            _counts(penalizer),
            [5, 2],
            "a request that committed no tokens had its counter advanced",
        )

    def test_other_penalty_ledgers_are_untouched(self):
        """The dflash hook must not advance the repetition / frequency /
        presence ledgers -- those need the accepted token ids, and are tracked
        separately by #28180."""
        orch = _make_orchestrator(
            MIN_NEW_TOKENS,
            penalizers={
                BatchedMinNewTokensPenalizer,
                BatchedFrequencyPenalizer,
                BatchedPresencePenalizer,
                BatchedRepetitionPenalizer,
            },
        )
        batch = _FakeDFlashBatch(orch)
        watched = {
            BatchedFrequencyPenalizer: "cumulated_frequency_penalties",
            BatchedPresencePenalizer: "cumulated_presence_penalties",
            BatchedRepetitionPenalizer: "cumulated_repetition_penalties",
        }
        before = {
            cls: getattr(orch.penalizers[cls], attr).clone()
            for cls, attr in watched.items()
        }

        _commit(batch, [3, 1])
        spec_utils.spec_prepare_for_decode(batch)

        self.assertEqual(_counts(orch.penalizers[BatchedMinNewTokensPenalizer]), [3, 1])
        for cls, attr in watched.items():
            self.assertTrue(
                torch.equal(getattr(orch.penalizers[cls], attr), before[cls]),
                f"{cls.__name__}.{attr} was modified by the dflash "
                f"min_new_tokens hook (F2 ledger is out of scope)",
            )

    def test_hot_path_skipped_when_only_other_penalizers_are_required(self):
        """min_new_tokens == 0 with repetition/frequency/presence active: the
        orchestrator is required as a whole, but the min_new_tokens penalizer
        was never prepared, so the hook must return before it builds the counts
        list, the pinned host tensor or the H2D copy."""
        orch = _make_orchestrator(
            0,
            penalizers={
                BatchedMinNewTokensPenalizer,
                BatchedFrequencyPenalizer,
                BatchedPresencePenalizer,
                BatchedRepetitionPenalizer,
            },
        )
        batch = _FakeDFlashBatch(orch)
        self.assertTrue(orch.is_required, "test setup: other penalizers active")
        self.assertFalse(orch.penalizers[BatchedMinNewTokensPenalizer].is_prepared())

        watched = {
            BatchedFrequencyPenalizer: "cumulated_frequency_penalties",
            BatchedPresencePenalizer: "cumulated_presence_penalties",
            BatchedRepetitionPenalizer: "cumulated_repetition_penalties",
        }
        before = {
            cls: getattr(orch.penalizers[cls], attr).clone()
            for cls, attr in watched.items()
        }

        # is_pin_memory_available is evaluated while building the host tensor,
        # so observing it is equivalent to observing the allocation.
        allocations = []
        setter_calls = []
        saved_pin = schedule_batch.is_pin_memory_available
        saved_setter = type(orch).set_min_new_tokens_output_counts
        schedule_batch.is_pin_memory_available = lambda device: (
            allocations.append(device) or False
        )
        type(orch).set_min_new_tokens_output_counts = (
            lambda self, counts: setter_calls.append(counts)
        )
        try:
            _commit(batch, [3, 1])
            spec_utils.spec_prepare_for_decode(batch)
        finally:
            schedule_batch.is_pin_memory_available = saved_pin
            type(orch).set_min_new_tokens_output_counts = saved_setter

        self.assertEqual(
            allocations,
            [],
            "the dflash hook built a min_new_tokens counts tensor for a request "
            "that does not use min_new_tokens",
        )
        self.assertEqual(
            setter_calls,
            [],
            "the dflash hook synced the min_new_tokens counter for a request "
            "that does not use min_new_tokens",
        )
        for cls, attr in watched.items():
            self.assertTrue(
                torch.equal(getattr(orch.penalizers[cls], attr), before[cls]),
                f"{cls.__name__}.{attr} was modified by the dflash "
                f"min_new_tokens hook",
            )

    def test_hot_path_runs_when_min_new_tokens_is_active(self):
        """Control for the test above: with min_new_tokens > 0 the same
        instrumentation must observe exactly one allocation and one sync."""
        orch = _make_orchestrator(MIN_NEW_TOKENS)
        batch = _FakeDFlashBatch(orch)
        allocations = []
        saved_pin = schedule_batch.is_pin_memory_available
        schedule_batch.is_pin_memory_available = lambda device: (
            allocations.append(device) or False
        )
        try:
            _commit(batch, [3, 1])
            spec_utils.spec_prepare_for_decode(batch)
        finally:
            schedule_batch.is_pin_memory_available = saved_pin

        self.assertEqual(len(allocations), 1)
        self.assertEqual(_counts(orch.penalizers[BatchedMinNewTokensPenalizer]), [3, 1])

    def test_generic_cumulate_still_advances_by_one(self):
        """Normal decode and EAGLE keep the one-token-per-step accounting."""
        orch = _make_orchestrator(MIN_NEW_TOKENS)
        penalizer = orch.penalizers[BatchedMinNewTokensPenalizer]
        ids = torch.tensor([11] * BS, dtype=torch.int64)
        orch.cumulate_output_tokens(ids)
        orch.cumulate_output_tokens(ids)
        self.assertEqual(_counts(penalizer), [2, 2])

    def test_no_counting_when_no_penalizer_is_required(self):
        """min_new_tokens == 0 -> nothing is prepared, the hook is a no-op."""
        orch = _make_orchestrator(0)
        batch = _FakeDFlashBatch(orch)
        self.assertFalse(orch.is_required)
        spec_utils.spec_prepare_for_decode(batch)
        self.assertFalse(orch.penalizers[BatchedMinNewTokensPenalizer].is_prepared())


if __name__ == "__main__":
    unittest.main()
