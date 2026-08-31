"""Unit tests for srt/sampling/sampling_batch_info.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=26, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="base-c-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.sampling.sampling_batch_info import (
    SamplingBatchInfo,
    merge_bias_tensor,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 32
DEVICE = "cpu"


# Helper: construct a minimal SamplingBatchInfo
def _make_info(batch_size=2, **overrides):
    """Create a SamplingBatchInfo with sane defaults for testing."""
    defaults = dict(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.ones(batch_size),
        top_ks=torch.full((batch_size,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        penalizer_orchestrator=MagicMock(is_required=False),
    )
    defaults.update(overrides)
    return SamplingBatchInfo(**defaults)


def _serial_batched_fill(entries, vocab_mask):
    for entry in entries:
        entry.grammar.fill_vocab_mask(vocab_mask, entry.row)


class TestMergeBiasTensor(CustomTestCase):

    def test_both_none_returns_none(self):
        """Test that merging two None tensors returns None."""
        result = merge_bias_tensor(None, None, 2, 3, DEVICE, 0.0)
        self.assertIsNone(result)

    def test_both_present_concatenates(self):
        """Test that two present tensors are concatenated along batch dim."""
        lhs = torch.ones(2, VOCAB_SIZE)
        rhs = torch.zeros(3, VOCAB_SIZE)
        result = merge_bias_tensor(lhs, rhs, 2, 3, DEVICE, 0.0)
        self.assertEqual(result.shape, (5, VOCAB_SIZE))
        self.assertEqual(result[0, 0].item(), 1.0)
        self.assertEqual(result[3, 0].item(), 0.0)

    def test_lhs_none_fills_default(self):
        """Test that missing lhs is filled with default value before concatenation."""
        rhs = torch.ones(3, VOCAB_SIZE)
        result = merge_bias_tensor(None, rhs, 2, 3, DEVICE, 0.0)
        self.assertEqual(result.shape, (5, VOCAB_SIZE))
        # First 2 rows filled with default (0.0)
        self.assertEqual(result[0, 0].item(), 0.0)
        # Last 3 rows from rhs
        self.assertEqual(result[2, 0].item(), 1.0)

    def test_rhs_none_fills_default(self):
        """Test that missing rhs is filled with default value before concatenation."""
        lhs = torch.ones(2, VOCAB_SIZE)
        result = merge_bias_tensor(lhs, None, 2, 3, DEVICE, 0.0)
        self.assertEqual(result.shape, (5, VOCAB_SIZE))
        self.assertEqual(result[0, 0].item(), 1.0)
        # Last 3 rows filled with default (0.0)
        self.assertEqual(result[3, 0].item(), 0.0)

    def test_custom_default_value(self):
        """Test that a custom default (-1.0) fills the missing lhs rows."""
        rhs = torch.ones(1, VOCAB_SIZE)
        result = merge_bias_tensor(None, rhs, 2, 1, DEVICE, -1.0)
        self.assertEqual(result[0, 0].item(), -1.0)
        self.assertEqual(result[1, 0].item(), -1.0)
        self.assertEqual(result[2, 0].item(), 1.0)


# SamplingBatchInfo.__len__
class TestSamplingBatchInfoLen(CustomTestCase):

    def test_len_matches_batch_size(self):
        """Test that __len__ returns batch size (number of temperature rows)."""
        info = _make_info(batch_size=5)
        self.assertEqual(len(info), 5)


class TestMergeCustomLogitProcessor(CustomTestCase):

    def test_both_none_returns_none(self):
        """Test that merging two None processor dicts returns None."""
        result = SamplingBatchInfo.merge_custom_logit_processor(
            None, None, 2, 3, DEVICE
        )
        self.assertIsNone(result)

    def test_same_key_merges_masks(self):
        """Test that same processor key concatenates the boolean masks."""
        proc = MagicMock()
        lhs = {42: (proc, torch.tensor([True, False]))}
        rhs = {42: (proc, torch.tensor([False, True, True]))}
        result = SamplingBatchInfo.merge_custom_logit_processor(lhs, rhs, 2, 3, DEVICE)
        self.assertIn(42, result)
        self.assertEqual(result[42][1].shape[0], 5)
        self.assertTrue(result[42][1][0].item())  # from lhs
        self.assertFalse(result[42][1][1].item())  # from lhs
        self.assertTrue(result[42][1][3].item())  # from rhs

    def test_disjoint_keys(self):
        """Test that disjoint processor keys are merged with zero-filled padding."""
        proc_a = MagicMock()
        proc_b = MagicMock()
        lhs = {1: (proc_a, torch.tensor([True, False]))}
        rhs = {2: (proc_b, torch.tensor([True]))}
        result = SamplingBatchInfo.merge_custom_logit_processor(lhs, rhs, 2, 1, DEVICE)
        # Key 1: lhs mask [True, False] + zero-filled rhs [False]
        self.assertEqual(result[1][1].shape[0], 3)
        self.assertTrue(result[1][1][0].item())
        self.assertFalse(result[1][1][2].item())
        # Key 2: zero-filled lhs [False, False] + rhs mask [True]
        self.assertEqual(result[2][1].shape[0], 3)
        self.assertFalse(result[2][1][0].item())
        self.assertTrue(result[2][1][2].item())

    def test_lhs_none_rhs_present(self):
        """Test that None lhs is treated as empty dict and rhs mask is padded."""
        proc = MagicMock()
        rhs = {10: (proc, torch.tensor([True]))}
        result = SamplingBatchInfo.merge_custom_logit_processor(None, rhs, 2, 1, DEVICE)
        self.assertIn(10, result)
        self.assertEqual(result[10][1].shape[0], 3)


# apply_logits_bias
class TestApplyLogitsBias(CustomTestCase):

    def test_applies_additive_penalties(self):
        """Test that pre-accumulated additive penalties are added to logits."""
        info = _make_info(batch_size=1)
        info.acc_additive_penalties = torch.tensor([[-1.0] * VOCAB_SIZE])
        logits = torch.zeros(1, VOCAB_SIZE)
        info.apply_logits_bias(logits)
        self.assertAlmostEqual(logits[0, 0].item(), -1.0, places=5)

    def test_applies_logit_bias(self):
        """Test that per-token logit_bias is added to logits."""
        info = _make_info(batch_size=1)
        bias = torch.zeros(1, VOCAB_SIZE)
        bias[0, 5] = 10.0
        info.logit_bias = bias
        logits = torch.zeros(1, VOCAB_SIZE)
        info.apply_logits_bias(logits)
        self.assertAlmostEqual(logits[0, 5].item(), 10.0, places=5)
        self.assertAlmostEqual(logits[0, 0].item(), 0.0, places=5)

    def test_applies_vocab_mask(self):
        """Test that a grammar_mask gets applied to the logits."""
        info = _make_info(batch_size=1)
        grammar = MagicMock()
        info.grammar_mask = GrammarMask(grammar, torch.ones(1, VOCAB_SIZE))
        logits = torch.zeros(1, VOCAB_SIZE)
        info.apply_logits_bias(logits)
        grammar.apply_vocab_mask.assert_called_once()

    def test_applies_penalizer_orchestrator(self):
        """Test that a required orchestrator's apply() is called on logits."""
        orch = MagicMock(is_required=True)
        info = _make_info(batch_size=1, penalizer_orchestrator=orch)
        logits = torch.zeros(1, VOCAB_SIZE)
        info.apply_logits_bias(logits)
        orch.apply.assert_called_once_with(logits)

    def test_no_bias_no_change(self):
        """Test that logits stay unchanged when no bias sources are set."""
        info = _make_info(batch_size=1)
        info.acc_additive_penalties = None
        info.logit_bias = None
        info.grammar_mask = None
        logits = torch.zeros(1, VOCAB_SIZE)
        original = logits.clone()
        info.apply_logits_bias(logits)
        self.assertTrue(torch.equal(logits, original))

    def test_apply_logits_bias_without_penalizer_orchestrator(self):
        info = _make_info(batch_size=1, penalizer_orchestrator=None)
        logits = torch.zeros(1, VOCAB_SIZE)

        info.apply_logits_bias(logits)

        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))

    def test_observer_sees_production_constraint_boundary(self):
        events = []

        class Observer:
            def before_grammar(self, logits, sampling_info):
                events.append(("before", logits.clone()))
                return object()

        grammar = MagicMock()
        grammar.apply_vocab_mask.side_effect = lambda logits, vocab_mask: logits.fill_(
            -4.0
        )
        info = _make_info(batch_size=1)
        info.acc_additive_penalties = torch.ones(1, VOCAB_SIZE)
        info.grammar_mask = GrammarMask(grammar, torch.ones(1, VOCAB_SIZE))
        info.logit_bias = torch.full((1, VOCAB_SIZE), 2.0)
        logits = torch.zeros(1, VOCAB_SIZE)

        state = info.apply_logits_bias_with_observer(logits, observer=Observer())

        self.assertIsNotNone(state)
        self.assertTrue(torch.equal(events[0][1], torch.ones_like(logits)))
        self.assertEqual(len(events), 1)
        self.assertTrue(torch.equal(logits, torch.full_like(logits, -2.0)))
        grammar.apply_vocab_mask.assert_called_once()

    def test_observer_path_preserves_production_logit_transforms(self):
        class Observer:
            def before_grammar(self, logits, sampling_info):
                return object()

        def make_info():
            grammar = MagicMock()
            grammar.apply_vocab_mask.side_effect = (
                lambda logits, vocab_mask: logits.add_(vocab_mask)
            )
            info = _make_info(batch_size=1)
            info.acc_additive_penalties = torch.linspace(
                -0.5, 0.5, VOCAB_SIZE
            ).unsqueeze(0)
            info.acc_scaling_penalties = torch.linspace(1.0, 1.5, VOCAB_SIZE).unsqueeze(
                0
            )
            info.grammar_mask = GrammarMask(
                grammar,
                torch.linspace(-2.0, 0.0, VOCAB_SIZE).unsqueeze(0),
            )
            info.logit_bias = torch.linspace(0.0, 1.0, VOCAB_SIZE).unsqueeze(0)
            return info

        ordinary = make_info()
        observed = make_info()
        ordinary_logits = torch.linspace(-3.0, 3.0, VOCAB_SIZE).unsqueeze(0)
        observed_logits = ordinary_logits.clone()

        ordinary.apply_logits_bias(ordinary_logits)
        observed.apply_logits_bias_with_observer(
            observed_logits,
            observer=Observer(),
        )

        self.assertTrue(torch.equal(observed_logits, ordinary_logits))


# update_penalties
class TestUpdatePenalties(CustomTestCase):

    def test_required_creates_penalties_tensor(self):
        """Test that update_penalties allocates a zero tensor and calls orchestrator methods."""
        orch = MagicMock(is_required=True)
        orch.accumulate_scaling_penalties.return_value = None
        info = _make_info(batch_size=2, penalizer_orchestrator=orch)
        info.update_penalties()
        self.assertIsNotNone(info.acc_additive_penalties)
        self.assertEqual(info.acc_additive_penalties.shape, (2, VOCAB_SIZE))
        orch.accumulate_additive_penalties.assert_called_once_with(
            info.acc_additive_penalties
        )
        orch.accumulate_scaling_penalties.assert_called_once()

    def test_not_required_sets_none(self):
        """Test that update_penalties sets acc_additive_penalties to None when not required."""
        orch = MagicMock(is_required=False)
        info = _make_info(batch_size=2, penalizer_orchestrator=orch)
        info.update_penalties()
        self.assertIsNone(info.acc_additive_penalties)


# update_regex_vocab_mask
class TestUpdateRegexVocabMask(CustomTestCase):

    def test_no_grammars_clears_mask(self):
        """Test that None grammars clears the grammar_mask."""
        info = _make_info(batch_size=1)
        info.grammars = None
        info.update_regex_vocab_mask()
        self.assertIsNone(info.grammar_mask)

    def test_empty_grammars_clears_mask(self):
        """Test that empty grammars list clears the grammar_mask."""
        info = _make_info(batch_size=1)
        info.grammars = []
        info.update_regex_vocab_mask()
        self.assertIsNone(info.grammar_mask)

    def test_with_grammars_allocates_and_fills(self):
        """Test that an active grammar gets allocate, fill, and move called."""
        grammar = MagicMock()
        grammar.finished = False
        grammar.is_terminated.return_value = False
        grammar.fill_vocab_mask_batched.side_effect = _serial_batched_fill
        grammar.allocate_vocab_mask.return_value = torch.zeros(1, VOCAB_SIZE)
        grammar.move_vocab_mask.return_value = torch.zeros(1, VOCAB_SIZE)
        info = _make_info(batch_size=1)
        info.grammars = [grammar]
        info.update_regex_vocab_mask()
        grammar.allocate_vocab_mask.assert_called_once()
        grammar.fill_vocab_mask_batched.assert_called_once()
        grammar.fill_vocab_mask.assert_called_once()
        grammar.move_vocab_mask.assert_called_once()
        self.assertIs(info.grammar_mask.grammar, grammar)

    def test_mixed_grammars_only_active_fills(self):
        """Test that finished, terminated, and None grammars are skipped."""
        active = MagicMock()
        active.finished = False
        active.is_terminated.return_value = False
        active.fill_vocab_mask_batched.side_effect = _serial_batched_fill
        active.allocate_vocab_mask.return_value = torch.zeros(3, VOCAB_SIZE)
        active.move_vocab_mask.return_value = torch.zeros(3, VOCAB_SIZE)

        finished = MagicMock()
        finished.finished = True

        terminated = MagicMock()
        terminated.finished = False
        terminated.is_terminated.return_value = True

        info = _make_info(batch_size=3)
        info.grammars = [active, finished, terminated]
        info.update_regex_vocab_mask()

        active.fill_vocab_mask_batched.assert_called_once()
        active.fill_vocab_mask.assert_called_once()
        finished.fill_vocab_mask.assert_not_called()
        terminated.fill_vocab_mask.assert_not_called()

    def test_batched_fill_skips_serial_loop(self):
        """A backend with a batched kernel must not also run the serial fill."""
        active = MagicMock()
        active.finished = False
        active.is_terminated.return_value = False
        active.allocate_vocab_mask.return_value = torch.zeros(2, VOCAB_SIZE)
        active.move_vocab_mask.return_value = torch.zeros(2, VOCAB_SIZE)

        other = MagicMock()
        other.finished = False
        other.is_terminated.return_value = False

        info = _make_info(batch_size=2)
        info.grammars = [active, other]
        info.update_regex_vocab_mask()

        # The batched call happened with both active rows; serial fill skipped.
        active.fill_vocab_mask_batched.assert_called_once()
        entries, _mask = active.fill_vocab_mask_batched.call_args.args
        self.assertEqual([e.row for e in entries], [0, 1])
        active.fill_vocab_mask.assert_not_called()
        other.fill_vocab_mask.assert_not_called()


# filter_batch
class TestFilterBatch(CustomTestCase):

    def test_filter_keeps_correct_indices(self):
        """Test that filter retains rows at indices 0 and 2, dropping index 1."""
        info = _make_info(batch_size=3)
        info.temperatures = torch.tensor([[1.0], [2.0], [3.0]])
        info.top_ps = torch.tensor([0.9, 0.8, 0.7])
        info.top_ks = torch.tensor([10, 20, 30], dtype=torch.int32)
        info.min_ps = torch.tensor([0.0, 0.1, 0.2])
        info.logit_bias = torch.ones(3, VOCAB_SIZE)
        keep = torch.tensor([0, 2])
        info.filter_batch([0, 2], keep)
        self.assertEqual(len(info), 2)
        self.assertAlmostEqual(info.temperatures[0, 0].item(), 1.0)
        self.assertAlmostEqual(info.temperatures[1, 0].item(), 3.0)
        self.assertAlmostEqual(info.top_ps[1].item(), 0.7)
        # logit_bias should also be filtered
        self.assertEqual(info.logit_bias.shape, (2, VOCAB_SIZE))

    def test_filter_with_custom_logit_processor(self):
        """Test that filter updates both custom_params list and processor mask."""
        proc = MagicMock()
        info = _make_info(batch_size=3)
        info.has_custom_logit_processor = True
        info.custom_logit_processor = {42: (proc, torch.tensor([True, False, True]))}
        info.custom_params = [{"a": 1}, {"b": 2}, {"c": 3}]
        keep = torch.tensor([0, 2])
        info.filter_batch([0, 2], keep)
        self.assertEqual(info.custom_params, [{"a": 1}, {"c": 3}])
        mask = info.custom_logit_processor[42][1]
        self.assertEqual(mask.shape[0], 2)

    def test_filter_removes_all_custom_processors(self):
        """Test cleanup when filter removes all requests using a processor."""
        proc = MagicMock()
        info = _make_info(batch_size=3)
        info.has_custom_logit_processor = True
        info.custom_logit_processor = {42: (proc, torch.tensor([False, True, False]))}
        info.custom_params = [None, {"x": 1}, None]
        # Keep only index 0 and 2 — processor 42's mask becomes [False, False]
        keep = torch.tensor([0, 2])
        info.filter_batch([0, 2], keep)
        self.assertFalse(info.has_custom_logit_processor)
        self.assertIsNone(info.custom_logit_processor)

    def test_filter_with_none_sampling_seed(self):
        """Test that filter preserves None sampling_seed without error."""
        info = _make_info(batch_size=3)
        info.sampling_seed = None
        keep = torch.tensor([1])
        info.filter_batch([1], keep)
        self.assertIsNone(info.sampling_seed)


# merge_batch
class TestMergeBatch(CustomTestCase):

    def test_merge_concatenates_tensors(self):
        """Test that merge concatenates temperature tensors from both batches."""
        info1 = _make_info(batch_size=2)
        info1.temperatures = torch.tensor([[1.0], [2.0]])
        info2 = _make_info(batch_size=1)
        info2.temperatures = torch.tensor([[3.0]])
        info1.merge_batch(info2)
        self.assertEqual(len(info1), 3)
        self.assertAlmostEqual(info1.temperatures[2, 0].item(), 3.0)

    def test_merge_combines_flags(self):
        """Test that merge ANDs is_all_greedy and ORs need_*_sampling flags."""
        info1 = _make_info(
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
        )
        info2 = _make_info(
            is_all_greedy=False,
            need_top_p_sampling=True,
            need_top_k_sampling=True,
            need_min_p_sampling=True,
        )
        info1.merge_batch(info2)
        self.assertFalse(info1.is_all_greedy)  # AND semantics
        self.assertTrue(info1.need_top_p_sampling)  # OR semantics
        self.assertTrue(info1.need_top_k_sampling)  # OR semantics
        self.assertTrue(info1.need_min_p_sampling)  # OR semantics

    def test_merge_with_logit_bias(self):
        """Test that merge pads missing logit_bias with zeros before concatenation."""
        info1 = _make_info(batch_size=1)
        info1.logit_bias = torch.ones(1, VOCAB_SIZE)
        info2 = _make_info(batch_size=1)
        info2.logit_bias = None
        info1.merge_batch(info2)
        self.assertEqual(info1.logit_bias.shape, (2, VOCAB_SIZE))

    def test_merge_with_custom_logit_processor(self):
        """Test that merge combines processors when only one side has them."""
        proc = MagicMock()
        info1 = _make_info(batch_size=1)
        info1.has_custom_logit_processor = True
        info1.custom_logit_processor = {1: (proc, torch.tensor([True]))}
        info1.custom_params = [{"a": 1}]
        info2 = _make_info(batch_size=1)
        info2.has_custom_logit_processor = False
        info2.custom_logit_processor = None
        info2.custom_params = None
        info1.merge_batch(info2)
        self.assertTrue(info1.has_custom_logit_processor)
        self.assertEqual(len(info1.custom_params), 2)

    def test_merge_with_none_sampling_seed(self):
        """Test that merge preserves None when both sampling_seeds are None."""
        info1 = _make_info(batch_size=1)
        info1.sampling_seed = None
        info2 = _make_info(batch_size=1)
        info2.sampling_seed = None
        info1.merge_batch(info2)
        self.assertIsNone(info1.sampling_seed)

    def test_merge_with_both_sampling_seeds(self):
        """Test that merge concatenates both sampling_seed tensors."""
        info1 = _make_info(batch_size=2)
        info1.sampling_seed = torch.tensor([10, 20], dtype=torch.int64)
        info2 = _make_info(batch_size=1)
        info2.sampling_seed = torch.tensor([30], dtype=torch.int64)
        info1.merge_batch(info2)
        self.assertEqual(info1.sampling_seed.shape[0], 3)
        self.assertEqual(info1.sampling_seed[0].item(), 10)
        self.assertEqual(info1.sampling_seed[1].item(), 20)
        self.assertEqual(info1.sampling_seed[2].item(), 30)


# copy_for_forward
class TestCopyForForward(CustomTestCase):

    def test_returns_copy_without_orchestrator(self):
        """Test that copy_for_forward returns a copy with orchestrator set to None."""
        orch = MagicMock(is_required=False)
        info = _make_info(batch_size=1, penalizer_orchestrator=orch)
        copied = info.copy_for_forward()
        self.assertIsNone(copied.penalizer_orchestrator)
        # Original should still have orchestrator
        self.assertIsNotNone(info.penalizer_orchestrator)


# from_schedule_batch
class TestFromScheduleBatch(CustomTestCase):

    def setUp(self):
        super().setUp()
        # from_schedule_batch reads these two flags from the exec bag; give
        # each test a mutable stand-in so it does not depend on a published
        # (or leaked) process context.
        self._exec_ns = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
            features=SimpleNamespace(enable_custom_logit_processor=False),
        )
        exec_patch = patch(
            "sglang.srt.sampling.sampling_batch_info.get_exec",
            return_value=self._exec_ns,
        )
        exec_patch.start()
        self.addCleanup(exec_patch.stop)

    def _make_req(
        self,
        temp=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        freq=0.0,
        presence=0.0,
        min_tokens=0,
        logit_bias=None,
        seed=None,
        stop_ids=None,
        eos_id=2,
    ):
        req = MagicMock()
        req.sampling_params.temperature = temp
        req.sampling_params.top_p = top_p
        req.sampling_params.top_k = top_k
        req.sampling_params.min_p = min_p
        req.sampling_params.frequency_penalty = freq
        req.sampling_params.presence_penalty = presence
        req.sampling_params.min_new_tokens = min_tokens
        req.sampling_params.logit_bias = logit_bias
        req.sampling_params.sampling_seed = seed
        req.sampling_params.stop_token_ids = stop_ids
        req.sampling_params.custom_params = None
        req.custom_logit_processor = None
        req.tokenizer.additional_stop_token_ids = None
        req.tokenizer.eos_token_id = eos_id
        return req

    def test_basic_construction(self):
        """Test that from_schedule_batch correctly extracts sampling params from requests."""

        reqs = [self._make_req(temp=0.8, top_p=0.9, top_k=50, min_p=0.1)]
        batch = MagicMock()
        batch.reqs = reqs
        batch.device = DEVICE

        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assertEqual(len(info), 1)
        self.assertAlmostEqual(info.temperatures[0, 0].item(), 0.8, places=5)
        self.assertAlmostEqual(info.top_ps[0].item(), 0.9, places=5)
        self.assertEqual(info.top_ks[0].item(), 50)

    def test_greedy_detection(self):
        """Test that top_k=1 sets is_all_greedy=True."""

        reqs = [self._make_req(top_k=1)]
        batch = MagicMock()
        batch.reqs = reqs
        batch.device = DEVICE
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assertTrue(info.is_all_greedy)

    def test_logit_bias_construction(self):
        """Test that logit_bias dict is converted to a tensor with correct values."""

        reqs = [self._make_req(logit_bias={"5": 2.0, "10": -1.0})]
        batch = MagicMock()
        batch.reqs = reqs
        batch.device = DEVICE
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assertIsNotNone(info.logit_bias)
        self.assertAlmostEqual(info.logit_bias[0, 5].item(), 2.0)
        self.assertAlmostEqual(info.logit_bias[0, 10].item(), -1.0)
        self.assertAlmostEqual(info.logit_bias[0, 0].item(), 0.0)

    def test_deterministic_seed(self):
        """Test that explicit seed=123 is kept and missing seed defaults to 42."""
        self._exec_ns.deterministic.enable_deterministic_inference = True

        reqs = [self._make_req(seed=123), self._make_req(seed=None)]
        batch = MagicMock()
        batch.reqs = reqs
        batch.device = DEVICE
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assertIsNotNone(info.sampling_seed)
        self.assertEqual(info.sampling_seed[0].item(), 123)
        self.assertEqual(info.sampling_seed[1].item(), 42)  # default

    def test_from_schedule_batch_sampling_flags(self):
        """Test that sampling flags (need_top_p/top_k/min_p) are set correctly."""

        reqs = [self._make_req(top_p=0.9, top_k=50, min_p=0.1)]
        batch = MagicMock()
        batch.reqs = reqs
        batch.device = DEVICE
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assertTrue(info.need_top_p_sampling)  # 0.9 != 1.0
        self.assertTrue(info.need_top_k_sampling)  # 50 != TOP_K_ALL
        self.assertTrue(info.need_min_p_sampling)  # 0.1 > 0
        self.assertFalse(info.is_all_greedy)  # top_k=50 > 1

    def test_no_logit_bias_when_all_none(self):
        """Test that logit_bias stays None when no request has logit_bias set."""

        reqs = [self._make_req(), self._make_req()]
        batch = MagicMock()
        batch.reqs = reqs
        batch.device = DEVICE
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assertIsNone(info.logit_bias)

    def test_custom_logit_processor_merging(self):
        """Test deserialization and merging of custom logit processors."""
        from sglang.srt.sampling.custom_logit_processor import (
            DisallowedTokensLogitsProcessor,
        )

        self._exec_ns.features.enable_custom_logit_processor = True

        proc_str = DisallowedTokensLogitsProcessor.to_str()
        req1 = self._make_req()
        req1.custom_logit_processor = proc_str
        req1.sampling_params.custom_params = {"token_ids": [1]}
        req2 = self._make_req()
        req2.custom_logit_processor = None  # no processor
        req2.sampling_params.custom_params = None

        batch = MagicMock()
        batch.reqs = [req1, req2]
        batch.device = DEVICE
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)

        self.assertTrue(info.has_custom_logit_processor)
        self.assertIsNotNone(info.custom_logit_processor)
        self.assertEqual(len(info.custom_logit_processor), 1)
        # Check the mask: req1 has processor (True), req2 doesn't (False)
        key = list(info.custom_logit_processor.keys())[0]
        proc, mask = info.custom_logit_processor[key]
        self.assertIsInstance(proc, DisallowedTokensLogitsProcessor)
        self.assertTrue(mask[0].item())
        self.assertFalse(mask[1].item())
        # custom_params should be collected for all reqs
        self.assertEqual(len(info.custom_params), 2)


if __name__ == "__main__":
    unittest.main()
