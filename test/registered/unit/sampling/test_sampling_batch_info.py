"""Unit tests for srt/sampling/sampling_batch_info.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=26, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="extra-test-cpu-intel")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.constrained.base_grammar_backend import GrammarMask
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.sampling.custom_logit_processor import (
    UOCR_PAGE_GUARD_PROCESSOR,
    DisallowedTokensLogitsProcessor,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_pipeline import (
    SamplingPipeline,
    _adjust_before_watermark,
)
from sglang.srt.sampling.watermarking.interface import SampleStep
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 32
DEVICE = "cpu"


# Helper: construct a minimal SamplingBatchInfo
def _tables_step(additive, scaling=None):
    """A step whose penalty tables are given."""
    return SimpleNamespace(logit_penalties=lambda: (additive, scaling))


def _make_info(**overrides):
    """Create a SamplingBatchInfo with sane defaults for testing."""
    return SamplingBatchInfo(vocab_size=VOCAB_SIZE, device=DEVICE, **overrides)


def _reqs(n):
    """n requests behind a merge_batch side; only their count matters here."""
    return [SimpleNamespace(filler_seed=0)] * n


def _serial_batched_fill(entries, vocab_mask):
    for entry in entries:
        entry.grammar.fill_vocab_mask(vocab_mask, entry.row)


class TestSamplingMaskBatchIndices(CustomTestCase):
    def test_filter_and_merge_preserve_support_logprob_modes(self):
        info = _make_info(
            return_sampling_masks=[True, True, False],
            return_sampling_support_logprobs=[False, True, False],
            sampling_mask_batch_indices=torch.tensor([0, 1]),
        )
        info.filter_batch([1, 2], torch.tensor([1, 2]))
        self.assertEqual(info.return_sampling_support_logprobs, [True, False])
        self.assertEqual(info.sampling_support_logprobs_capture_indices.tolist(), [0])

        other = _make_info(
            return_sampling_masks=[True, True],
            return_sampling_support_logprobs=[False, True],
            sampling_mask_batch_indices=torch.tensor([0, 1]),
        )
        info.merge_batch(other, reqs=_reqs(2), other_reqs=_reqs(2))
        self.assertEqual(
            info.return_sampling_support_logprobs,
            [True, False, False, True],
        )
        self.assertEqual(
            info.sampling_support_logprobs_capture_indices.tolist(), [0, 2]
        )

    def test_filter_removes_last_opted_in_row_then_merge_restores_capture(self):
        info = _make_info(
            return_sampling_masks=[False, True],
            sampling_mask_batch_indices=torch.tensor([1]),
        )
        info.filter_batch([0], torch.tensor([0]))
        self.assertIsNone(info.sampling_mask_batch_indices)
        other = _make_info(
            return_sampling_masks=[True],
            sampling_mask_batch_indices=torch.tensor([0]),
        )
        info.merge_batch(other, reqs=_reqs(1), other_reqs=_reqs(1))
        self.assertEqual(info.return_sampling_masks, [False, True])
        self.assertEqual(info.sampling_mask_batch_indices.tolist(), [1])

    def test_filter_rebuilds_row_indices(self):
        info = _make_info(
            return_sampling_masks=[False, True, False, True],
            sampling_mask_batch_indices=torch.tensor([1, 3]),
        )

        info.filter_batch([1, 2, 3], torch.tensor([1, 2, 3]))

        self.assertEqual(info.return_sampling_masks, [True, False, True])
        self.assertEqual(info.sampling_mask_batch_indices.tolist(), [0, 2])

    def test_merge_offsets_rhs_row_indices(self):
        lhs = _make_info(
            return_sampling_masks=[False, True],
            sampling_mask_batch_indices=torch.tensor([1]),
        )
        rhs = _make_info(
            return_sampling_masks=[True, False, True],
            sampling_mask_batch_indices=torch.tensor([0, 2]),
        )

        lhs.merge_batch(rhs, reqs=_reqs(2), other_reqs=_reqs(3))

        self.assertEqual(lhs.return_sampling_masks, [False, True, True, False, True])
        self.assertEqual(lhs.sampling_mask_batch_indices.tolist(), [1, 2, 4])


def _preset_grammar_mask(info, grammar_mask):
    """Make the grammar slot build `grammar_mask`."""
    info.update_regex_vocab_mask = lambda: setattr(info, "grammar_mask", grammar_mask)


# SamplingPipeline
class TestSamplingPipelineOrder(CustomTestCase):
    def test_sample_runs_every_stage_in_one_fixed_order(self):
        """The stages run in non-speculative sampling's order; the custom
        processor and sanitize also process the watermark's logprob logits."""
        events = []
        info = _make_info()
        _preset_grammar_mask(
            info, SimpleNamespace(apply=lambda logits: events.append("grammar"))
        )
        info.step_penalties = SimpleNamespace(
            logit_penalties=lambda: events.append("penalty") or (None, None),
            commit=lambda token_ids: events.append("penalty commit"),
        )
        logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
        logprob_logits = torch.zeros(1, 4)

        def named(logits):
            return "logits" if logits is logits_output.next_token_logits else "logprob"

        class Watermark:
            def apply(self, logits, *, keep_logits):
                events.append("watermark")
                return SampleStep(sampling_info=info, logprob_logits=logprob_logits)

            def commit(self, token_ids):
                events.append("watermark commit")

        def sampler(*args, logprob_logits):
            events.append("sampler")
            return torch.tensor([0])

        pipeline = SamplingPipeline(
            sampler=sampler,
            watermark=SimpleNamespace(bind_sample=lambda forward_batch: Watermark()),
        )
        forward_batch = SimpleNamespace(
            sampling_info=info,
            return_logprob=True,
            top_logprobs_nums=[0],
            token_ids_logprobs=[None],
            positions=torch.tensor([0]),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
        )
        info.has_custom_logit_processor = True
        module = "sglang.srt.sampling.sampling_pipeline"
        with (
            patch(
                f"{module}.apply_logit_bias",
                side_effect=lambda *args, **kwargs: events.append("logit_bias"),
            ),
            patch(
                f"{module}.apply_custom_logit_processor",
                side_effect=lambda logits, info: events.append(
                    f"custom {named(logits)}"
                ),
            ),
            patch(
                f"{module}.sanitize_nan_logits",
                side_effect=lambda logits, msg: events.append(
                    f"sanitize {named(logits)}"
                ),
            ),
        ):
            pipeline.sample(
                logits_output, forward_batch, observer=None, graph_token_ids=None
            )

        self.assertEqual(
            events,
            [
                "penalty",
                "grammar",
                "logit_bias",
                "watermark",
                "custom logits",
                "sanitize logits",
                "custom logprob",
                "sanitize logprob",
                "sampler",
                "watermark commit",
                "penalty commit",
            ],
        )

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
        info = _make_info()
        info.step_penalties = _tables_step(torch.ones(1, VOCAB_SIZE))
        _preset_grammar_mask(info, GrammarMask(grammar, torch.ones(1, VOCAB_SIZE)))
        info.logit_bias = torch.full((1, VOCAB_SIZE), 2.0)
        logits = torch.zeros(1, VOCAB_SIZE)

        state = _adjust_before_watermark(logits, info, observer=Observer())

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
            info = _make_info()
            info.step_penalties = _tables_step(
                torch.linspace(-0.5, 0.5, VOCAB_SIZE).unsqueeze(0),
                torch.linspace(1.0, 1.5, VOCAB_SIZE).unsqueeze(0),
            )
            _preset_grammar_mask(
                info,
                GrammarMask(
                    grammar, torch.linspace(-2.0, 0.0, VOCAB_SIZE).unsqueeze(0)
                ),
            )
            info.logit_bias = torch.linspace(0.0, 1.0, VOCAB_SIZE).unsqueeze(0)
            return info

        ordinary = make_info()
        observed = make_info()
        ordinary_logits = torch.linspace(-3.0, 3.0, VOCAB_SIZE).unsqueeze(0)
        observed_logits = ordinary_logits.clone()

        _adjust_before_watermark(ordinary_logits, ordinary, observer=None)
        _adjust_before_watermark(observed_logits, observed, observer=Observer())

        self.assertTrue(torch.equal(observed_logits, ordinary_logits))


# update_regex_vocab_mask
class TestUpdateRegexVocabMask(CustomTestCase):
    def test_no_grammars_clears_mask(self):
        """Test that None grammars clears the grammar_mask."""
        info = _make_info()
        info.grammars = None
        info.update_regex_vocab_mask()
        self.assertIsNone(info.grammar_mask)

    def test_empty_grammars_clears_mask(self):
        """Test that empty grammars list clears the grammar_mask."""
        info = _make_info()
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
        info = _make_info()
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

        info = _make_info()
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

        info = _make_info()
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
    def test_seeds_follow_rows_through_merge_and_filter(self):
        """A merged-in unseeded side takes its requests' filler seeds; once the last
        seeded row leaves, the batch returns to the unseeded samplers."""
        info = _make_info()
        seeded = _make_info(
            sampling_seed=torch.tensor([123], dtype=torch.int64),
            rows_with_seed=[True],
        )
        info.merge_batch(
            seeded,
            reqs=[SimpleNamespace(filler_seed=456)],
            other_reqs=[SimpleNamespace(filler_seed=789)],
        )
        self.assertEqual(info.sampling_seed.tolist(), [456, 123])
        self.assertEqual(info.rows_with_seed, [False, True])

        info.filter_batch([0], torch.tensor([0]))
        self.assertIsNone(info.sampling_seed)
        self.assertIsNone(info.rows_with_seed)

    def test_filter_with_none_sampling_seed(self):
        """Test that filter preserves None sampling_seed without error."""
        info = _make_info()
        info.sampling_seed = None
        keep = torch.tensor([1])
        info.filter_batch([1], keep)
        self.assertIsNone(info.sampling_seed)


# merge_batch
class TestMergeBatch(CustomTestCase):
    def test_merge_with_none_sampling_seed(self):
        """Test that merge preserves None when both sampling_seeds are None."""
        info1 = _make_info()
        info1.sampling_seed = None
        info2 = _make_info()
        info2.sampling_seed = None
        info1.merge_batch(info2, reqs=_reqs(1), other_reqs=_reqs(1))
        self.assertIsNone(info1.sampling_seed)

    def test_merge_with_both_sampling_seeds(self):
        """Test that merge concatenates both sampling_seed tensors."""
        info1 = _make_info()
        info1.sampling_seed = torch.tensor([10, 20], dtype=torch.int64)
        info2 = _make_info()
        info2.sampling_seed = torch.tensor([30], dtype=torch.int64)
        info1.merge_batch(info2, reqs=_reqs(2), other_reqs=_reqs(1))
        self.assertEqual(info1.sampling_seed.shape[0], 3)
        self.assertEqual(info1.sampling_seed[0].item(), 10)
        self.assertEqual(info1.sampling_seed[1].item(), 20)
        self.assertEqual(info1.sampling_seed[2].item(), 30)


# copy_for_forward
class TestCopyForForward(CustomTestCase):
    def test_returns_copy(self):
        """Both copy implementations preserve every field, including fields
        declared with init=False, and leave the step's tables to sampling."""
        for fast_path in (False, True):
            info = _make_info()
            step = MagicMock()
            info.step_penalties = step
            with (
                self.subTest(fast_path=fast_path),
                patch(
                    "sglang.srt.sampling.sampling_batch_info._fast_copy_enabled",
                    return_value=fast_path,
                ),
            ):
                copied = info.copy_for_forward()
            step.logit_penalties.assert_not_called()
            for name, value in info.__dict__.items():
                self.assertIs(copied.__dict__[name], value, name)


# from_schedule_batch and set_plugin_state
class TestFromScheduleBatch(CustomTestCase):
    def setUp(self):
        super().setUp()
        # These read two flags from the exec bag; give
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
        self._spec_ns = SimpleNamespace(speculative_algorithm=None)
        spec_patch = patch(
            "sglang.srt.sampling.sampling_batch_info.get_spec",
            return_value=self._spec_ns,
        )
        spec_patch.start()
        self.addCleanup(spec_patch.stop)

    def _make_req(
        self,
        top_k=-1,
        freq=0.0,
        presence=0.0,
        min_tokens=0,
        seed=None,
        stop_ids=None,
        eos_id=2,
        filler_seed=0,
    ):
        req = MagicMock()
        req.sampling_params.top_k = top_k
        req.sampling_params.frequency_penalty = freq
        req.sampling_params.presence_penalty = presence
        req.sampling_params.min_new_tokens = min_tokens
        req.sampling_params.sampling_seed = seed
        req.sampling_params.logit_bias = None
        req.sampling_params.stop_token_ids = stop_ids
        req.sampling_params.custom_params = None
        req.custom_logit_processor = None
        req.tokenizer.additional_stop_token_ids = None
        req.tokenizer.eos_token_id = eos_id
        req.filler_seed = filler_seed
        return req

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

    def test_seed_without_deterministic_inference(self):
        """A non-greedy request's seed applies and the other rows take their filler
        seeds; a greedy seed, or speculative decoding, leaves the batch unseeded."""
        batch = MagicMock()
        batch.device = DEVICE
        batch.reqs = [
            self._make_req(seed=123, top_k=50),
            self._make_req(seed=None, filler_seed=456),
        ]
        info = SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE)
        self.assertEqual(info.sampling_seed.tolist(), [123, 456])
        self.assertEqual(info.rows_with_seed, [True, False])

        batch.reqs = [self._make_req(seed=123, top_k=1)]
        self.assertIsNone(
            SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE).sampling_seed
        )

        self._spec_ns.speculative_algorithm = "EAGLE"
        batch.reqs = [self._make_req(seed=123, top_k=50)]
        self.assertIsNone(
            SamplingBatchInfo.from_schedule_batch(batch, VOCAB_SIZE).sampling_seed
        )

    def test_builtin_page_guard_reaches_logits_with_external_processors_disabled(self):
        from array import array

        from sglang.srt.sampling.custom_logit_processor import (
            apply_custom_logit_processor,
        )

        req = self._make_req()
        req.origin_input_ids = array("q", [7, 20])
        req.output_ids = array("q", [7, 10, 11, 12, 13])
        req.uocr_doc = None
        req.page_guard_state = None
        req.custom_logit_processor = UOCR_PAGE_GUARD_PROCESSOR
        req.sampling_params.custom_params = dict(
            __req__=req,
            page_token_id=7,
            eos_token_id=1,
            max_page_tokens=3,
        )
        info = _make_info()
        info.set_plugin_state([req])
        self.assertTrue(info.has_custom_logit_processor)
        logits = torch.zeros(1, VOCAB_SIZE)
        apply_custom_logit_processor(logits, info)
        self.assertEqual(torch.isfinite(logits[0]).nonzero().flatten().tolist(), [1])
        self.assertFalse(self._exec_ns.features.enable_custom_logit_processor)

    def test_external_processor_still_requires_the_enable_flag(self):
        req = self._make_req()
        req.custom_logit_processor = DisallowedTokensLogitsProcessor.to_str()
        info = _make_info()
        info.set_plugin_state([req])
        self.assertFalse(info.has_custom_logit_processor)
        self.assertIsNone(info.custom_logit_processor)

    def test_custom_logit_processor_grouping(self):
        """Test deserialization and grouping of custom logit processors."""
        self._exec_ns.features.enable_custom_logit_processor = True

        proc_str = DisallowedTokensLogitsProcessor.to_str()
        req1 = self._make_req()
        req1.custom_logit_processor = proc_str
        req1.sampling_params.custom_params = {"token_ids": [1]}
        req2 = self._make_req()
        req2.custom_logit_processor = None  # no processor
        req2.sampling_params.custom_params = None

        info = _make_info()
        info.set_plugin_state([req1, req2])

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
