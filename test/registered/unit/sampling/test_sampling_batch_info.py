"""Unit tests for srt/sampling/sampling_batch_info.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=26, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="extra-test-cpu-intel")

import dataclasses
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.sampling.custom_logit_processor import (
    UOCR_PAGE_GUARD_PROCESSOR,
    DisallowedTokensLogitsProcessor,
)
from sglang.srt.sampling.grammar_mask import build_grammar_mask
from sglang.srt.sampling.penalties import StepPenalties
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_pipeline import (
    SamplingPipeline,
    VerifyRoute,
    VerifyStep,
    _adjust_before_watermark,
)
from sglang.srt.sampling.watermarking.interface import SampleWatermarkOutput
from sglang.test.test_utils import CustomTestCase

VOCAB_SIZE = 32
DEVICE = "cpu"


# Helper: construct a minimal SamplingBatchInfo
def _tables_step(additive, scaling=None):
    """A step whose penalty tables are given."""
    step = SimpleNamespace(logit_penalties=lambda: (additive, scaling))
    step.apply = partial(StepPenalties.apply, step)
    return step


def _make_info(**overrides):
    """Create a SamplingBatchInfo with sane defaults for testing."""
    return SamplingBatchInfo(vocab_size=VOCAB_SIZE, device=DEVICE, **overrides)


def _serial_batched_fill(entries, vocab_mask):
    for entry in entries:
        entry.grammar.fill_vocab_mask(vocab_mask, entry.row)


def _grammar(vocab_mask, apply_vocab_mask):
    """An active grammar whose step mask is `vocab_mask`."""
    grammar = MagicMock(finished=False)
    grammar.is_terminated.return_value = False
    grammar.allocate_vocab_mask.return_value = vocab_mask
    grammar.move_vocab_mask.side_effect = lambda mask, device: mask
    grammar.apply_vocab_mask.side_effect = apply_vocab_mask
    return grammar


# SamplingPipeline
class TestSamplingPipelineOrder(CustomTestCase):
    def test_sample_runs_every_stage_in_one_fixed_order(self):
        """The stages run in non-speculative sampling's order; the custom
        processor and sanitize also process the watermark's logprob logits."""
        events = []
        info = _make_info()
        info.grammars = [
            _grammar(
                torch.zeros(1, 4),
                lambda logits, vocab_mask: events.append("grammar"),
            )
        ]
        info.step_penalties = SimpleNamespace(
            apply=lambda logits: events.append("penalty"),
            commit=lambda token_ids: events.append("penalty commit"),
        )
        logits_output = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 4))
        logprob_logits = torch.zeros(1, 4)

        def named(logits):
            return "logits" if logits is logits_output.next_token_logits else "logprob"

        class Watermark:
            def apply(self, logits, *, keep_logits):
                events.append("watermark")
                return SampleWatermarkOutput(
                    sampling_info=info, logprob_logits=logprob_logits
                )

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
                side_effect=lambda logits, info, **kwargs: events.append(
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
                "custom logprob",
                "sanitize logits",
                "sanitize logprob",
                "sampler",
                "watermark commit",
                "penalty commit",
            ],
        )

    def test_verify_runs_the_watermark_before_the_custom_processor(self):
        """Verify runs sampling's order; the custom processor also processes the
        unwatermarked logits the watermark keeps for the logprobs."""
        events = []
        logits = torch.zeros(2, 4)
        watermarked = torch.zeros(2, 4)

        def apply_watermark(logits, *, sampling_info, **kwargs):
            events.append("watermark")
            return watermarked, sampling_info

        step = VerifyStep(
            batch=SimpleNamespace(
                sampling_info=_make_info(has_custom_logit_processor=True),
                req_pool_indices=None,
                return_logprob=False,
            ),
            route=VerifyRoute(adjusts_logits=True, need_truncated_sampling=False),
            watermark=SimpleNamespace(apply=apply_watermark),
            watermark_full_mask=True,
        )

        module = "sglang.srt.sampling.sampling_pipeline"
        with (
            patch(
                f"{module}.apply_logit_bias",
                side_effect=lambda *args, **kwargs: events.append("logit_bias"),
            ),
            patch(
                f"{module}.apply_custom_logit_processor",
                side_effect=lambda t, info, **kwargs: events.append(
                    "custom " + ("logits" if t is logits else "watermarked")
                ),
            ),
        ):
            verify_logits, _ = step.adjust(logits, torch.zeros(1, 2, dtype=torch.long))

        self.assertIs(verify_logits, watermarked)
        self.assertEqual(
            events, ["logit_bias", "watermark", "custom watermarked", "custom logits"]
        )

    def test_observer_sees_production_constraint_boundary(self):
        events = []

        class Observer:
            def before_grammar(self, logits, sampling_info):
                events.append(("before", logits.clone()))
                return object()

        grammar = _grammar(
            torch.ones(1, VOCAB_SIZE),
            lambda logits, vocab_mask: logits.fill_(-4.0),
        )
        info = _make_info()
        info.step_penalties = _tables_step(torch.ones(1, VOCAB_SIZE))
        info.grammars = [grammar]
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
            grammar = _grammar(
                torch.linspace(-2.0, 0.0, VOCAB_SIZE).unsqueeze(0),
                lambda logits, vocab_mask: logits.add_(vocab_mask),
            )
            info = _make_info()
            info.step_penalties = _tables_step(
                torch.linspace(-0.5, 0.5, VOCAB_SIZE).unsqueeze(0),
                torch.linspace(1.0, 1.5, VOCAB_SIZE).unsqueeze(0),
            )
            info.grammars = [grammar]
            info.logit_bias = torch.linspace(0.0, 1.0, VOCAB_SIZE).unsqueeze(0)
            return info

        ordinary = make_info()
        observed = make_info()
        ordinary_logits = torch.linspace(-3.0, 3.0, VOCAB_SIZE).unsqueeze(0)
        observed_logits = ordinary_logits.clone()

        _adjust_before_watermark(ordinary_logits, ordinary, observer=None)
        _adjust_before_watermark(observed_logits, observed, observer=Observer())

        self.assertTrue(torch.equal(observed_logits, ordinary_logits))


# build_grammar_mask
class TestBuildGrammarMask(CustomTestCase):
    def test_no_grammars_builds_no_mask(self):
        for grammars in (None, []):
            self.assertIsNone(
                build_grammar_mask(grammars, vocab_size=VOCAB_SIZE, device=DEVICE)
            )

    def test_with_grammars_allocates_and_fills(self):
        """Test that an active grammar gets allocate, fill, and move called."""
        grammar = MagicMock()
        grammar.finished = False
        grammar.is_terminated.return_value = False
        grammar.fill_vocab_mask_batched.side_effect = _serial_batched_fill
        grammar.allocate_vocab_mask.return_value = torch.zeros(1, VOCAB_SIZE)
        grammar.move_vocab_mask.return_value = torch.zeros(1, VOCAB_SIZE)
        grammar_mask = build_grammar_mask(
            [grammar], vocab_size=VOCAB_SIZE, device=DEVICE
        )
        grammar.allocate_vocab_mask.assert_called_once()
        grammar.fill_vocab_mask_batched.assert_called_once()
        grammar.fill_vocab_mask.assert_called_once()
        grammar.move_vocab_mask.assert_called_once()
        self.assertIs(grammar_mask.grammar, grammar)

    def test_mixed_grammars_only_active_fills(self):
        """Test that finished, terminated, and None grammars are skipped."""
        active = MagicMock()
        active.finished = False
        active.is_terminated.return_value = False
        active.fill_vocab_mask_batched.side_effect = _serial_batched_fill
        active.allocate_vocab_mask.return_value = torch.zeros(4, VOCAB_SIZE)
        active.move_vocab_mask.return_value = torch.zeros(4, VOCAB_SIZE)

        finished = MagicMock()
        finished.finished = True

        terminated = MagicMock()
        terminated.finished = False
        terminated.is_terminated.return_value = True

        build_grammar_mask(
            [None, active, finished, terminated], vocab_size=VOCAB_SIZE, device=DEVICE
        )

        active.fill_vocab_mask_batched.assert_called_once()
        active.fill_vocab_mask.assert_called_once_with(ANY, 1)
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

        build_grammar_mask([active, other], vocab_size=VOCAB_SIZE, device=DEVICE)

        # The batched call happened with both active rows; serial fill skipped.
        active.fill_vocab_mask_batched.assert_called_once()
        entries, _mask = active.fill_vocab_mask_batched.call_args.args
        self.assertEqual([e.row for e in entries], [0, 1])
        active.fill_vocab_mask.assert_not_called()
        other.fill_vocab_mask.assert_not_called()


# copy_for_forward
class TestCopyForForward(CustomTestCase):
    def test_copy_matches_dataclasses_replace(self):
        """The one-dict-update copy holds exactly what dataclasses.replace holds,
        and leaves the step's penalty tables to sampling."""
        info = _make_info()
        step = MagicMock()
        info.step_penalties = step
        copied = info.copy_for_forward()
        replaced = dataclasses.replace(info)
        step.logit_penalties.assert_not_called()
        self.assertIs(type(copied), SamplingBatchInfo)
        self.assertIsNot(copied, info)
        self.assertEqual(copied.__dict__.keys(), replaced.__dict__.keys())
        for name, value in replaced.__dict__.items():
            self.assertIs(copied.__dict__[name], value, name)
        # The copy skips __init__: a __post_init__ or an init=False field would
        # make it differ from dataclasses.replace.
        self.assertNotIn("__post_init__", dir(SamplingBatchInfo))
        self.assertTrue(all(f.init for f in dataclasses.fields(SamplingBatchInfo)))


class TestStepBuilds(CustomTestCase):
    def setUp(self):
        super().setUp()
        # These read a flag from the exec bag; give each test a mutable
        # stand-in so it does not depend on a published (or leaked) process
        # context.
        self._exec_ns = SimpleNamespace(
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
        top_k=-1,
        freq=0.0,
        presence=0.0,
        min_tokens=0,
        stop_ids=None,
        eos_id=2,
        mask_mode=None,
    ):
        req = MagicMock()
        req.sampling_params.top_k = top_k
        req.sampling_params.frequency_penalty = freq
        req.sampling_params.presence_penalty = presence
        req.sampling_params.min_new_tokens = min_tokens
        req.sampling_params.logit_bias = None
        req.sampling_params.stop_token_ids = stop_ids
        req.sampling_params.custom_params = None
        req.custom_logit_processor = None
        req.grammar = None
        req.return_sampling_mask = mask_mode is not None
        req.sampling_logprobs_mode = mask_mode
        req.tokenizer.additional_stop_token_ids = None
        req.tokenizer.eos_token_id = eos_id
        return req

    def test_mask_rows(self):
        """Mask rows index the batch; support rows index the mask rows."""
        reqs = [
            self._make_req(top_k=5),
            self._make_req(top_k=7, mask_mode="full"),
            self._make_req(top_k=50, mask_mode="support"),
            self._make_req(top_k=3, mask_mode="support"),
        ]
        info = _make_info()
        info.set_step_state(reqs)
        self.assertEqual(info.sampling_mask_batch_indices.tolist(), [1, 2, 3])
        self.assertEqual(
            info.sampling_support_logprobs_capture_indices.tolist(), [1, 2]
        )
        self.assertEqual(info.sampling_mask_top_ks, [1, 7, 50, 3])

        info.set_step_state(reqs[:2])
        self.assertEqual(info.sampling_mask_batch_indices.tolist(), [1])
        self.assertIsNone(info.sampling_support_logprobs_capture_indices)

        info.set_step_state(reqs[:1])
        self.assertIsNone(info.sampling_mask_batch_indices)
        self.assertIsNone(info.sampling_mask_top_ks)

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
        info.set_step_state([req])
        self.assertTrue(info.has_custom_logit_processor)
        logits = torch.zeros(1, VOCAB_SIZE)
        apply_custom_logit_processor(logits, info)
        self.assertEqual(torch.isfinite(logits[0]).nonzero().flatten().tolist(), [1])
        self.assertFalse(self._exec_ns.features.enable_custom_logit_processor)

    def test_grammars_are_set_only_when_a_request_has_one(self):
        reqs = [self._make_req(), self._make_req()]
        info = _make_info()
        info.set_step_state(reqs)
        self.assertIsNone(info.grammars)

        grammar = object()
        reqs[1].grammar = grammar
        info.set_step_state(reqs)
        self.assertEqual(info.grammars, [None, grammar])

    def test_external_processor_still_requires_the_enable_flag(self):
        req = self._make_req()
        req.custom_logit_processor = DisallowedTokensLogitsProcessor.to_str()
        info = _make_info()
        info.set_step_state([req])
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
        info.set_step_state([req1, req2])

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
