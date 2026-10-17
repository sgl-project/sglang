"""Request and log counters must use the completed round's proposal width."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.entrypoints.openai.utils import spec_tokens_details_from_meta_info
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_req(**changes):
    req = SimpleNamespace(
        is_retracted=False,
        finished=lambda: False,
        grammar=None,
        kv=SimpleNamespace(kv_committed_len=0),
        spec_verify_ct=0,
        spec_num_correct_drafts=0,
        spec_num_proposed_drafts=0,
        update_spec_correct_drafts_histogram=lambda _: None,
    )
    req.__dict__.update(changes)
    return req


def make_result(width, accepted=1):
    return GenerationBatchResult(
        speculative_num_draft_tokens=width,
        next_token_ids=torch.arange(width),
        accept_lens=torch.tensor([accepted]),
    )


class TestSpecProposalAccounting(unittest.TestCase):
    def resolve(self, req, result, callback=None):
        # The controller callback runs BEFORE requests consume this result and
        # can activate a different runtime state. Never read that state here.
        processor = SimpleNamespace(
            model_worker=SimpleNamespace(
                on_verify_complete_cpu=callback or (lambda *a, **kw: None)
            ),
            advance_grammar_fsm=lambda *a: None,
        )
        return SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
            processor, result, SimpleNamespace(reqs=[req])
        )

    def meta(self, proposed, correct, verifies, completion):
        obj = SimpleNamespace(
            spec_verify_ct=[verifies],
            spec_num_correct_drafts=[correct],
            spec_num_proposed_drafts=[proposed],
            completion_tokens=[completion],
            spec_correct_drafts_histogram=[[verifies]],
        )
        out = {}
        TokenizerManager._calculate_spec_decoding_metrics(None, out, obj, 0)
        return out

    def test_static_width_preserves_existing_counts(self):
        req = make_req()
        for _ in range(3):
            self.resolve(req, make_result(3, 2))
        self.assertEqual(req.spec_num_proposed_drafts, 6)
        self.assertEqual(req.spec_num_correct_drafts, 3)
        self.assertEqual(req.spec_verify_ct, 3)
        self.assertEqual(self.meta(6, 3, 3, 7)["spec_accept_rate"], 0.5)

    def test_switch_2_0_0_2_accumulates_actual_proposals(self):
        req = make_req()
        current = SimpleNamespace(width=3)
        for width, accepted in [(3, 2), (1, 1), (1, 1), (3, 3)]:
            result = make_result(width, accepted)
            self.resolve(req, result, lambda *a, **kw: setattr(current, "width", 99))
        self.assertEqual(current.width, 99)
        self.assertEqual(req.spec_num_proposed_drafts, 4)
        self.assertEqual(req.spec_num_correct_drafts, 3)
        self.assertEqual(req.spec_verify_ct, 4)
        self.assertEqual(req.kv.kv_committed_len, 7)
        self.assertEqual(self.meta(4, 3, 4, 8)["spec_accept_rate"], 0.75)

    def test_zero_proposals_keep_counts_and_undefined_rate(self):
        req = make_req()
        for _ in range(127):
            self.resolve(req, make_result(1))
        self.assertEqual(req.spec_num_proposed_drafts, 0)
        out = self.meta(0, 0, 127, 128)
        self.assertIsNone(out["spec_accept_rate"])
        self.assertEqual(out["spec_num_proposed_drafts"], 0)
        self.assertEqual(out["spec_verify_ct"], 127)
        self.assertAlmostEqual(out["spec_accept_length"], 128 / 127)
        self.assertIsNone(spec_tokens_details_from_meta_info(out).spec_accept_rate)

    def test_inactive_requests_do_not_accumulate(self):
        for changes in ({"is_retracted": True}, {"finished": lambda: True}):
            with self.subTest(changes=changes):
                req = make_req(**changes)
                self.resolve(req, make_result(3))
                self.assertEqual(req.spec_verify_ct, 0)
                self.assertEqual(req.spec_num_proposed_drafts, 0)

    def test_no_verify_does_not_publish_metrics(self):
        self.assertEqual(self.meta(0, 0, 0, 1), {})

    def test_verify_width_is_not_output_stride_or_accepted_length(self):
        result = GenerationBatchResult(
            speculative_num_draft_tokens=4,
            speculative_output_stride=5,
            num_non_draft_tokens_per_req=2,
            num_correct_drafts=0,
        )
        self.assertEqual(result.get_num_proposed_drafts(2), 6)
        for width in (None, 0, -1):
            result.speculative_num_draft_tokens = width
            with self.assertRaises(RuntimeError):
                result.get_num_proposed_drafts(2)

    def test_log_window_sums_widths_and_batch_sizes(self):
        reporter = SimpleNamespace(
            spec_num_accept_tokens=0,
            spec_num_correct_drafts=0,
            spec_num_proposed_drafts=0,
            spec_num_forward_ct=0,
            spec_num_block_accept_tokens=0,
            spec_num_cap_tokens=0,
        )
        for bs, width, correct in [(8, 3, 5), (32, 1, 0), (4, 3, 3)]:
            result = make_result(width)
            SchedulerMetricsReporter.update_spec_metrics(
                reporter,
                bs,
                correct,
                num_accept_tokens=bs + correct,
                num_proposed_drafts=result.get_num_proposed_drafts(bs),
            )
        self.assertEqual(reporter.spec_num_proposed_drafts, 24)
        self.assertEqual(reporter.spec_num_correct_drafts, 8)
        self.assertEqual(reporter.spec_num_forward_ct, 44)
        self.assertEqual(reporter.spec_num_accept_tokens, 52)


if __name__ == "__main__":
    unittest.main()
