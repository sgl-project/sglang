"""Unit tests for srt/sampling/penaltylib/ — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.sampling.penaltylib.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.min_new_tokens import (
    BatchedMinNewTokensPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.presence_penalty import (
    BatchedPresencePenalizer,
)
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 32
DEVICE = "cpu"


# Helpers: mock Req and ScheduleBatch
def _make_req(freq=0.0, presence=0.0, min_tokens=0, stop_ids=None, eos_id=2):
    """Create a mock request with sampling params."""
    req = MagicMock()
    req.sampling_params.frequency_penalty = freq
    req.sampling_params.presence_penalty = presence
    req.sampling_params.min_new_tokens = min_tokens
    req.sampling_params.stop_token_ids = stop_ids
    req.tokenizer.additional_stop_token_ids = None
    req.tokenizer.eos_token_id = eos_id
    return req


def _make_batch(reqs):
    """Create a mock ScheduleBatch.
    Note: orchestrator accesses batch.reqs as an attribute (not a method call)."""
    batch = MagicMock()
    batch.reqs = reqs
    batch.device = DEVICE
    return batch


# BatchedPenalizerOrchestrator
class TestBatchedPenalizerOrchestrator(CustomTestCase):

    def test_init_detects_required_penalizers(self):
        """Test that orchestrator marks is_required=True when any request has nonzero penalty."""
        reqs = [_make_req(freq=1.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertTrue(orch.is_required)

    def test_init_not_required_when_no_penalties(self):
        """Test that orchestrator marks is_required=False when all penalties are zero."""
        reqs = [_make_req()]  # all defaults (0.0)
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertFalse(orch.is_required)

    def test_batch_property_via_weakref(self):
        """Test that batch property returns the original batch via weakref."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch, set())
        self.assertIs(orch.batch, batch)

    def test_batch_setter_none(self):
        """Test that setting batch to None breaks the weakref cleanly."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch, set())
        orch.batch = None
        self.assertIsNone(orch.batch)

    def test_batch_setter_new_batch(self):
        """Test that batch can be reassigned to a different ScheduleBatch."""
        reqs = [_make_req()]
        batch1 = _make_batch(reqs)
        batch2 = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(VOCAB_SIZE, batch1, set())
        orch.batch = batch2
        self.assertIs(orch.batch, batch2)

    def test_context_manager_releases(self):
        """Test that exiting the context manager releases all penalizers."""
        reqs = [_make_req(freq=1.0)]
        batch = _make_batch(reqs)
        with BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        ) as orch:
            self.assertTrue(orch.is_required)
        self.assertFalse(orch.is_required)
        self.assertEqual(len(orch.penalizers), 0)

    def test_filter_empty_indices_releases(self):
        """Test that filtering with no indices left fully releases the orchestrator."""
        reqs = [_make_req(freq=1.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        orch.filter(torch.tensor([], dtype=torch.long))
        self.assertFalse(orch.is_required)

    def test_filter_not_required_is_noop(self):
        """Test that filter on a not-required orchestrator does nothing."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertFalse(orch.is_required)
        orch.filter(torch.tensor([0]))  # should not raise

    def test_merge_both_not_required_is_noop(self):
        """Test that merging two not-required orchestrators stays not-required."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch1 = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        orch2 = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        orch1.merge(orch2)  # should not raise
        self.assertFalse(orch1.is_required)


# BatchedFrequencyPenalizer
class TestBatchedFrequencyPenalizer(CustomTestCase):

    def _setup(self, freq_values):
        reqs = [_make_req(freq=f) for f in freq_values]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        pen = orch.penalizers[BatchedFrequencyPenalizer]
        return orch, pen

    def test_is_required_with_nonzero_penalty(self):
        """Test that nonzero frequency_penalty makes the penalizer required."""
        _, pen = self._setup([1.5])
        self.assertTrue(pen.is_required())

    def test_is_not_required_with_zero_penalty(self):
        """Test that zero frequency_penalty makes the penalizer not required."""
        _, pen = self._setup([0.0])
        self.assertFalse(pen.is_required())

    def test_cumulate_and_apply(self):
        """Test that cumulating a token applies frequency penalty to its logit."""
        orch, pen = self._setup([2.0])
        output_ids = torch.tensor([5])
        pen.cumulate_output_tokens(output_ids)

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertAlmostEqual(logits[0, 5].item(), -2.0, places=5)
        # Other tokens unaffected
        self.assertAlmostEqual(logits[0, 0].item(), 0.0, places=5)

    def test_cumulate_twice_doubles_penalty(self):
        """Test that frequency penalty scales linearly with occurrence count."""
        orch, pen = self._setup([1.0])
        pen.cumulate_output_tokens(torch.tensor([3]))
        pen.cumulate_output_tokens(torch.tensor([3]))

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertAlmostEqual(logits[0, 3].item(), -2.0, places=5)

    def test_filter_keeps_subset(self):
        """Test that filter retains only the selected batch indices."""
        orch, pen = self._setup([1.0, 2.0])
        keep = torch.tensor([1])
        pen.filter(keep)
        self.assertEqual(pen.frequency_penalties.shape[0], 1)
        self.assertAlmostEqual(pen.frequency_penalties[0, 0].item(), 2.0, places=5)

    def test_merge_concatenates(self):
        """Test that merge concatenates penalty tensors from two penalizers."""
        _, pen1 = self._setup([1.0])
        _, pen2 = self._setup([2.0])
        pen1.merge(pen2)
        self.assertEqual(pen1.frequency_penalties.shape[0], 2)

    def test_teardown_cleans_attributes(self):
        """Test that teardown deletes internal tensors and resets prepared state."""
        _, pen = self._setup([1.0])
        pen.teardown()
        self.assertFalse(hasattr(pen, "frequency_penalties"))
        self.assertFalse(hasattr(pen, "cumulated_frequency_penalties"))
        self.assertFalse(pen.is_prepared())

    def test_cumulate_when_not_prepared_is_noop(self):
        """Test that cumulate before prepare does not crash."""
        _, pen = self._setup([0.0])
        # pen is not prepared (is_required=False)
        pen.cumulate_output_tokens(torch.tensor([1]))  # should not raise

    def test_apply_when_not_prepared_is_noop(self):
        """Test that apply on an unprepared penalizer leaves logits unchanged."""
        _, pen = self._setup([0.0])
        logits = torch.zeros(1, VOCAB_SIZE)
        original = logits.clone()
        pen.apply(logits)
        self.assertTrue(torch.equal(logits, original))


# BatchedPresencePenalizer
class TestBatchedPresencePenalizer(CustomTestCase):

    def _setup(self, presence_values):
        reqs = [_make_req(presence=p) for p in presence_values]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedPresencePenalizer}
        )
        pen = orch.penalizers[BatchedPresencePenalizer]
        return orch, pen

    def test_is_required_with_nonzero_penalty(self):
        """Test that nonzero presence_penalty makes the penalizer required."""
        _, pen = self._setup([0.5])
        self.assertTrue(pen.is_required())

    def test_presence_penalty_does_not_scale(self):
        """Test that presence penalty is flat (same value regardless of count)."""
        orch, pen = self._setup([1.0])
        pen.cumulate_output_tokens(torch.tensor([7]))
        pen.cumulate_output_tokens(torch.tensor([7]))  # same token again

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        # scatter_ overwrites (not adds), so penalty should be 1.0, not 2.0
        self.assertAlmostEqual(logits[0, 7].item(), -1.0, places=5)

    def test_filter_keeps_subset(self):
        """Test that filter retains the first request's presence penalty."""
        orch, pen = self._setup([1.0, 2.0])
        keep = torch.tensor([0])
        pen.filter(keep)
        self.assertEqual(pen.presence_penalties.shape[0], 1)
        self.assertAlmostEqual(pen.presence_penalties[0, 0].item(), 1.0, places=5)

    def test_merge_concatenates(self):
        """Test that merge concatenates presence penalty tensors."""
        _, pen1 = self._setup([1.0])
        _, pen2 = self._setup([2.0])
        pen1.merge(pen2)
        self.assertEqual(pen1.presence_penalties.shape[0], 2)

    def test_teardown_cleans_attributes(self):
        """Test that teardown removes the presence_penalties tensor."""
        _, pen = self._setup([1.0])
        pen.teardown()
        self.assertFalse(hasattr(pen, "presence_penalties"))


# BatchedMinNewTokensPenalizer
class TestBatchedMinNewTokensPenalizer(CustomTestCase):

    def _setup(self, configs):
        """configs: list of (min_tokens, stop_ids, eos_id)."""
        reqs = [_make_req(min_tokens=c[0], stop_ids=c[1], eos_id=c[2]) for c in configs]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedMinNewTokensPenalizer}
        )
        pen = orch.penalizers[BatchedMinNewTokensPenalizer]
        return orch, pen

    def test_is_required_with_positive_min_tokens(self):
        """Test that positive min_new_tokens makes the penalizer required."""
        _, pen = self._setup([(5, None, 2)])
        self.assertTrue(pen.is_required())

    def test_is_not_required_with_zero_min_tokens(self):
        """Test that min_new_tokens=0 makes the penalizer not required."""
        _, pen = self._setup([(0, None, 2)])
        self.assertFalse(pen.is_required())

    def test_blocks_eos_before_min_tokens(self):
        """Test that EOS token is blocked before min_new_tokens is reached."""
        orch, pen = self._setup([(3, None, 2)])
        # Before any output: len=0 < min=3 → block EOS (token 2)
        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertTrue(torch.isinf(logits[0, 2]) and logits[0, 2] < 0)
        # Non-stop tokens should be fine
        self.assertEqual(logits[0, 0].item(), 0.0)

    def test_allows_eos_after_min_tokens(self):
        """Test that EOS is allowed after generating min_new_tokens."""
        orch, pen = self._setup([(2, None, 2)])
        # Generate 2 tokens
        pen.cumulate_output_tokens(torch.tensor([10]))
        pen.cumulate_output_tokens(torch.tensor([11]))
        # Now len=2 >= min=2 → EOS should NOT be blocked
        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        self.assertEqual(logits[0, 2].item(), 0.0)

    def test_blocks_custom_stop_tokens(self):
        """Test that custom stop_token_ids are also blocked before min_new_tokens."""
        orch, pen = self._setup([(3, {5, 10}, 2)])
        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        # EOS (2), stop token 5, stop token 10 should all be blocked
        self.assertTrue(torch.isinf(logits[0, 2]) and logits[0, 2] < 0)
        self.assertTrue(torch.isinf(logits[0, 5]) and logits[0, 5] < 0)
        self.assertTrue(torch.isinf(logits[0, 10]) and logits[0, 10] < 0)

    def test_blocks_additional_stop_tokens(self):
        """Test that tokenizer's additional_stop_token_ids are also blocked."""
        req = _make_req(min_tokens=3, stop_ids=None, eos_id=2)
        req.tokenizer.additional_stop_token_ids = {7, 8}
        batch = _make_batch([req])
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedMinNewTokensPenalizer}
        )
        pen = orch.penalizers[BatchedMinNewTokensPenalizer]

        logits = torch.zeros(1, VOCAB_SIZE)
        pen.apply(logits)
        # EOS (2) + additional stops (7, 8) should all be blocked
        for tok in [2, 7, 8]:
            self.assertTrue(
                torch.isinf(logits[0, tok]) and logits[0, tok] < 0,
                f"token {tok} should be blocked before min_new_tokens",
            )
        # Non-stop tokens should be fine
        self.assertEqual(logits[0, 0].item(), 0.0)

    def test_filter_keeps_subset(self):
        """Test that filter keeps the second request (min_tokens=5) and drops the first."""
        orch, pen = self._setup([(3, None, 2), (5, None, 2)])
        keep = torch.tensor([1])
        pen.filter(keep)
        self.assertEqual(pen.min_new_tokens.shape[0], 1)
        self.assertEqual(pen.min_new_tokens[0, 0].item(), 5)

    def test_merge_concatenates(self):
        """Test that merge combines min_new_tokens tensors from two penalizers."""
        _, pen1 = self._setup([(3, None, 2)])
        _, pen2 = self._setup([(5, None, 2)])
        pen1.merge(pen2)
        self.assertEqual(pen1.min_new_tokens.shape[0], 2)

    def test_teardown_cleans_attributes(self):
        """Test that teardown removes min_new_tokens, stop_token_penalties, and len_output_tokens."""
        _, pen = self._setup([(3, None, 2)])
        pen.teardown()
        self.assertFalse(hasattr(pen, "min_new_tokens"))
        self.assertFalse(hasattr(pen, "stop_token_penalties"))
        self.assertFalse(hasattr(pen, "len_output_tokens"))


# _BatchedPenalizer base class edge cases
class TestBatchedPenalizerBase(CustomTestCase):

    def test_filter_when_not_prepared_is_noop(self):
        """Test that filter on an unprepared penalizer does not crash."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        pen = orch.penalizers[BatchedFrequencyPenalizer]
        # pen is not prepared (frequency_penalty=0 → not required)
        pen.filter(torch.tensor([0]))  # should not raise

    def test_merge_prepares_both_if_needed(self):
        """Test that merge prepares unprepared side before concatenating."""
        reqs_a = [_make_req(freq=0.0)]  # not required
        reqs_b = [_make_req(freq=1.0)]  # required
        batch_a = _make_batch(reqs_a)
        batch_b = _make_batch(reqs_b)
        orch_a = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch_a, {BatchedFrequencyPenalizer}
        )
        orch_b = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch_b, {BatchedFrequencyPenalizer}
        )
        pen_a = orch_a.penalizers[BatchedFrequencyPenalizer]
        pen_b = orch_b.penalizers[BatchedFrequencyPenalizer]
        self.assertFalse(pen_a.is_prepared())
        self.assertTrue(pen_b.is_prepared())
        # Merge should prepare pen_a first
        pen_a.merge(pen_b)
        self.assertTrue(pen_a.is_prepared())
        self.assertEqual(pen_a.frequency_penalties.shape[0], 2)

    def test_merge_both_unprepared_is_noop(self):
        """Test that merging two unprepared penalizers keeps them unprepared."""
        reqs = [_make_req()]
        batch = _make_batch(reqs)
        orch1 = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        orch2 = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        pen1 = orch1.penalizers[BatchedFrequencyPenalizer]
        pen2 = orch2.penalizers[BatchedFrequencyPenalizer]
        pen1.merge(pen2)  # both not prepared → noop
        self.assertFalse(pen1.is_prepared())

    def test_prepare_is_idempotent(self):
        """Test that calling prepare() multiple times does not crash."""
        reqs = [_make_req(freq=1.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        pen = orch.penalizers[BatchedFrequencyPenalizer]
        self.assertTrue(pen.is_prepared())
        # Calling prepare again should not crash or reinitialize
        pen.prepare()
        self.assertTrue(pen.is_prepared())


# Orchestrator with multiple penalizer types
class TestOrchestratorMultiplePenalizers(CustomTestCase):

    def test_all_three_penalizers(self):
        """Test orchestrator managing frequency, presence, and min_new_tokens together."""
        reqs = [_make_req(freq=1.0, presence=0.5, min_tokens=2, eos_id=2)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE,
            batch,
            {
                BatchedFrequencyPenalizer,
                BatchedPresencePenalizer,
                BatchedMinNewTokensPenalizer,
            },
        )
        self.assertTrue(orch.is_required)

        # Cumulate one token
        output_ids = torch.tensor([5])
        orch.cumulate_output_tokens(output_ids)

        # Apply all penalties
        logits = torch.zeros(1, VOCAB_SIZE)
        orch.apply(logits)

        # Token 5: freq_penalty=1.0 (cumulated once) + pres_penalty=0.5
        self.assertAlmostEqual(logits[0, 5].item(), -1.5, places=4)
        # EOS (token 2): blocked by min_new_tokens (len=1 < min=2)
        self.assertTrue(torch.isinf(logits[0, 2]) and logits[0, 2] < 0)

    def test_filter_with_penalizer_no_longer_required(self):
        """Test that penalizer is torn down when no longer required after filter."""
        reqs = [_make_req(freq=0.0), _make_req(freq=1.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertTrue(orch.is_required)

        # Keep only the request with freq=0 (index 0)
        batch.reqs = [reqs[0]]
        orch.filter(torch.tensor([0]))

        pen = orch.penalizers[BatchedFrequencyPenalizer]
        # After filter, only req with freq=0 remains → penalizer not required
        self.assertFalse(pen.is_required())

    def test_filter_keeps_required_penalizer(self):
        """Test that filter keeps penalizer active when still required."""
        reqs = [_make_req(freq=1.0), _make_req(freq=2.0)]
        batch = _make_batch(reqs)
        orch = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch, {BatchedFrequencyPenalizer}
        )
        self.assertTrue(orch.is_required)

        batch.reqs = [reqs[1]]
        orch.filter(torch.tensor([1]))
        self.assertTrue(orch.is_required)

    def test_merge_one_required(self):
        """Test that merge marks orchestrator as required when one side is."""
        reqs_a = [_make_req(freq=0.0)]
        reqs_b = [_make_req(freq=1.0)]
        batch_a = _make_batch(reqs_a)
        batch_b = _make_batch(reqs_b)
        orch_a = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch_a, {BatchedFrequencyPenalizer}
        )
        orch_b = BatchedPenalizerOrchestrator(
            VOCAB_SIZE, batch_b, {BatchedFrequencyPenalizer}
        )
        self.assertFalse(orch_a.is_required)
        self.assertTrue(orch_b.is_required)

        orch_a.merge(orch_b)
        self.assertTrue(orch_a.is_required)
        pen = orch_a.penalizers[BatchedFrequencyPenalizer]
        self.assertEqual(pen.frequency_penalties.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
