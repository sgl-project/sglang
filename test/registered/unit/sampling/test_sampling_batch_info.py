"""Unit tests for srt/sampling/sampling_batch_info.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=26, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="stage-b-test-cpu-intel")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
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
    )
    defaults.update(overrides)
    return SamplingBatchInfo(**defaults)


def _penalties(additive=None, scaling=None):
    return SimpleNamespace(logit_penalties=lambda: (additive, scaling))


def _serial_batched_fill(entries, vocab_mask):
    for entry in entries:
        entry.grammar.fill_vocab_mask(vocab_mask, entry.row)


# apply_logits_bias
class TestApplyLogitsBias(CustomTestCase):
    def test_applies_additive_penalties(self):
        """Test that pre-accumulated additive penalties are added to logits."""
        info = _make_info(
            batch_size=1, penalties=_penalties(torch.tensor([[-1.0] * VOCAB_SIZE]))
        )
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

    def test_no_bias_no_change(self):
        """Test that logits stay unchanged when no bias sources are set."""
        info = _make_info(batch_size=1)
        info.logit_bias = None
        info.grammar_mask = None
        logits = torch.zeros(1, VOCAB_SIZE)
        original = logits.clone()
        info.apply_logits_bias(logits)
        self.assertTrue(torch.equal(logits, original))

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
        info = _make_info(batch_size=1, penalties=_penalties(torch.ones(1, VOCAB_SIZE)))
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
            grammar.apply_vocab_mask.side_effect = lambda logits, vocab_mask: (
                logits.add_(vocab_mask)
            )
            info = _make_info(
                batch_size=1,
                penalties=_penalties(
                    torch.linspace(-0.5, 0.5, VOCAB_SIZE).unsqueeze(0),
                    torch.linspace(1.0, 1.5, VOCAB_SIZE).unsqueeze(0),
                ),
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


# build
class TestBuild(CustomTestCase):
    def setUp(self):
        super().setUp()
        # build reads these two flags from the exec bag; give
        # each test a mutable stand-in so it does not depend on a published
        # (or leaked) process context.
        self._exec_ns = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
            features=SimpleNamespace(
                enable_custom_logit_processor=False, enable_watermark=False
            ),
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
        req.sampling_params.repetition_penalty = 1.0
        req.sampling_params.min_new_tokens = min_tokens
        req.sampling_params.logit_bias = logit_bias
        req.sampling_params.sampling_seed = seed
        req.sampling_params.stop_token_ids = stop_ids
        req.sampling_params.custom_params = None
        req.custom_logit_processor = None
        req.grammar = None
        req.return_sampling_mask = False
        req.tokenizer.additional_stop_token_ids = None
        req.tokenizer.eos_token_id = eos_id
        return req

    def test_basic_construction(self):
        """Test that build correctly extracts sampling params from requests."""

        reqs = [self._make_req(temp=0.8, top_p=0.9, top_k=50, min_p=0.1)]

        info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE)
        self.assertEqual(len(info), 1)
        self.assertAlmostEqual(info.temperatures[0, 0].item(), 0.8, places=5)
        self.assertAlmostEqual(info.top_ps[0].item(), 0.9, places=5)
        self.assertEqual(info.top_ks[0].item(), 50)

    def test_greedy_detection(self):
        """Test that top_k=1 sets is_all_greedy=True."""

        reqs = [self._make_req(top_k=1)]
        info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE)
        self.assertTrue(info.is_all_greedy)

    def test_logit_bias_construction(self):
        """Test that logit_bias dict is converted to a tensor with correct values."""

        reqs = [self._make_req(logit_bias={"5": 2.0, "10": -1.0})]
        info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE)
        self.assertIsNotNone(info.logit_bias)
        self.assertAlmostEqual(info.logit_bias[0, 5].item(), 2.0)
        self.assertAlmostEqual(info.logit_bias[0, 10].item(), -1.0)
        self.assertAlmostEqual(info.logit_bias[0, 0].item(), 0.0)

    def test_deterministic_seed(self):
        """Test that explicit seed=123 is kept and missing seed defaults to 42."""
        self._exec_ns.deterministic.enable_deterministic_inference = True

        reqs = [self._make_req(seed=123), self._make_req(seed=None)]
        info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE)
        self.assertIsNotNone(info.sampling_seed)
        self.assertEqual(info.sampling_seed[0].item(), 123)
        self.assertEqual(info.sampling_seed[1].item(), 42)  # default

    def test_build_sampling_flags(self):
        """Test that sampling flags (need_top_p/top_k/min_p) are set correctly."""

        reqs = [self._make_req(top_p=0.9, top_k=50, min_p=0.1)]
        info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE)
        self.assertTrue(info.need_top_p_sampling)  # 0.9 != 1.0
        self.assertTrue(info.need_top_k_sampling)  # 50 != TOP_K_ALL
        self.assertTrue(info.need_min_p_sampling)  # 0.1 > 0
        self.assertFalse(info.is_all_greedy)  # top_k=50 > 1

    def test_npu_top_k_top_p_eligibility_uses_request_params(self):
        cases = (
            ((1, 1024), True),
            ((1, 1025), False),
            ((4, TOP_K_ALL), False),
        )
        for top_ks, expected in cases:
            with self.subTest(top_ks=top_ks):
                reqs = [self._make_req(top_k=top_k) for top_k in top_ks]

                info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE)

                self.assertEqual(info.npu_top_k_top_p_eligible, expected)

    def test_max_top_k_uses_request_params(self):
        cases = (
            ((20, 50), 50),
            ((1, 20), 20),
            ((20, TOP_K_ALL), None),
        )
        for top_ks, expected in cases:
            with self.subTest(top_ks=top_ks):
                reqs = [self._make_req(top_k=top_k) for top_k in top_ks]

                info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE)

                self.assertEqual(info.max_top_k, expected)

    def test_no_logit_bias_when_all_none(self):
        """Test that logit_bias stays None when no request has logit_bias set."""

        reqs = [self._make_req(), self._make_req()]
        info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE)
        self.assertIsNone(info.logit_bias)

    def test_custom_logit_processor_merging(self):
        """Test deserialization and merging of custom logit processors."""
        self._exec_ns.features.enable_custom_logit_processor = True

        proc_str = DisallowedTokensLogitsProcessor.to_str()
        req1 = self._make_req()
        req1.custom_logit_processor = proc_str
        req1.sampling_params.custom_params = {"token_ids": [1]}
        req2 = self._make_req()
        req2.custom_logit_processor = None  # no processor
        req2.sampling_params.custom_params = None

        reqs = [req1, req2]
        info = SamplingBatchInfo.build(reqs, VOCAB_SIZE, DEVICE)

        self.assertTrue(info.has_custom_logit_processor)
        self.assertIsNotNone(info.custom_logit_processor)
        self.assertEqual(len(info.custom_logit_processor), 1)
        key = list(info.custom_logit_processor.keys())[0]
        entry = info.custom_logit_processor[key]
        self.assertIsInstance(entry.processor, DisallowedTokensLogitsProcessor)
        self.assertEqual(entry.rows, [0])
        self.assertEqual(entry.indices.tolist(), entry.rows)
        self.assertEqual(entry.indices.dtype, torch.long)
        # custom_params should be collected for all reqs
        self.assertEqual(len(info.custom_params), 2)


if __name__ == "__main__":
    unittest.main()
