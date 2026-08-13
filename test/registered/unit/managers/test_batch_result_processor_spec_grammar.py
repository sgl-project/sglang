"""Unit tests for Spec V2 grammar truncation in _resolve_spec_v2_tokens.

The grammar-constrained spec path stops accepting at the grammar-terminating
token, so the over-drafted suffix is never committed to KV nor emitted.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeGrammar:
    """Grammar stub that terminates after `terminate_after` accepted tokens."""

    def __init__(self, terminate_after: int):
        self.accepted = []
        self.finished = False
        self._terminate_after = terminate_after

    def accept_token(self, token_id: int):
        self.accepted.append(token_id)

    def is_terminated(self) -> bool:
        return len(self.accepted) >= self._terminate_after


class _FakeSpecAlgorithm:
    def is_none(self) -> bool:
        return False

    def is_dflash(self) -> bool:
        return False


class _FakeForwardMode:
    def is_decode(self) -> bool:
        return True

    def is_extend(self) -> bool:
        return False


class _FakeBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.has_grammar = any(req.grammar is not None for req in reqs)
        self.forward_mode = _FakeForwardMode()
        self.spec_algorithm = _FakeSpecAlgorithm()


def _make_processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False),
        model_config=SimpleNamespace(think_start_ids=None, think_end_ids=None),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _make_req(terminate_after: int, origin_input_ids=None) -> Req:
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    req = Req(
        rid="r0",
        origin_input_text="",
        origin_input_ids=(
            origin_input_ids if origin_input_ids is not None else [1, 2, 3]
        ),
        sampling_params=sp,
    )
    req.grammar = _FakeGrammar(terminate_after=terminate_after)
    req.kv_committed_len = 0
    return req


def _make_result(num_draft_tokens, accept_lens, flat_tokens):
    return SimpleNamespace(
        next_token_ids=torch.tensor(flat_tokens, dtype=torch.long),
        accept_lens=torch.tensor(accept_lens, dtype=torch.long),
        speculative_num_draft_tokens=num_draft_tokens,
        num_correct_drafts=None,
        num_correct_drafts_per_req_cpu=None,
        block_accept_lens=None,
        cap_lens=None,
        copy_done=None,
        grammar_advanced=False,
    )


class TestSpecV2GrammarTruncation(CustomTestCase):
    def test_resolve_truncates_after_grammar_completion(self):
        req = _make_req(terminate_after=2)
        proc = _make_processor()
        # stride=4, accept_len=3 -> proposed [101, 102, 103]; grammar finishes at 102.
        result = _make_result(4, [3], [101, 102, 103, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[101, 102]])
        # No pre-claim: commit the full retained run (no -1 refund).
        self.assertEqual(req.kv_committed_len, 2)

    def test_resolve_keeps_all_when_grammar_not_terminated(self):
        req = _make_req(terminate_after=99)
        proc = _make_processor()
        result = _make_result(4, [3], [201, 202, 203, 0])

        predict_tokens = proc._resolve_spec_v2_tokens(result, _FakeBatch([req]))

        self.assertEqual(predict_tokens, [[201, 202, 203]])
        self.assertEqual(req.kv_committed_len, 3)


class TestReasoningTokenAccounting(CustomTestCase):
    THINK_START = [5, 6]
    THINK_END = [7, 8]

    def _make_reasoning_setup(self, origin_input_ids=None, with_start_ids=True):
        req = _make_req(terminate_after=99, origin_input_ids=origin_input_ids)
        req.require_reasoning = True
        processor = _make_processor()
        if with_start_ids:
            processor.model_config.think_start_ids = self.THINK_START
        processor.model_config.think_end_ids = self.THINK_END
        return req, processor

    def test_multi_token_end_can_span_decode_steps(self):
        req, processor = self._make_reasoning_setup(with_start_ids=False)

        processor._maybe_update_reasoning_tokens(req, [10, 7])
        processor._maybe_update_reasoning_tokens(req, [8, 11])

        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_over)

    def test_output_without_thinking_block_reports_no_reasoning(self):
        req, processor = self._make_reasoning_setup()

        processor._maybe_update_reasoning_tokens(req, [10, 11])
        processor._maybe_update_reasoning_tokens(req, [12, 13])

        self.assertEqual(req.reasoning_tokens, 0)
        self.assertFalse(req._is_reasoning_over)

    def test_start_emitted_in_output_counts_from_first_token(self):
        req, processor = self._make_reasoning_setup()

        processor._maybe_update_reasoning_tokens(req, [10, 5])
        processor._maybe_update_reasoning_tokens(req, [6, 11])
        processor._maybe_update_reasoning_tokens(req, [7, 8, 12])

        self.assertEqual(req.reasoning_tokens, 6)
        self.assertTrue(req._is_reasoning_over)

    def test_prefilled_start_in_prompt_opens_the_block(self):
        req, processor = self._make_reasoning_setup(origin_input_ids=[1, 5, 6])

        processor._maybe_update_reasoning_tokens(req, [10, 11])

        self.assertEqual(req.reasoning_tokens, 2)
        self.assertFalse(req._is_reasoning_over)

    def test_prompt_start_closed_by_a_later_end_does_not_open_the_block(self):
        req, processor = self._make_reasoning_setup(origin_input_ids=[5, 6, 9, 7, 8])

        processor._maybe_update_reasoning_tokens(req, [10, 11])

        self.assertEqual(req.reasoning_tokens, 0)

    def test_token_appended_without_counting_is_still_seen(self):
        req, processor = self._make_reasoning_setup()
        req.output_ids.append(5)
        req.output_ids.extend([6, 11])

        processor._maybe_update_reasoning_tokens(req, [6, 11])

        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_started)

    def test_end_delimiter_without_a_start_is_not_reasoning(self):
        req, processor = self._make_reasoning_setup()

        processor._maybe_update_reasoning_tokens(req, [10, 11])
        processor._maybe_update_reasoning_tokens(req, [7, 8, 12])

        self.assertEqual(req.reasoning_tokens, 0)
        self.assertFalse(req._is_reasoning_over)

    def test_scalar_token_ids(self):
        req, processor = self._make_reasoning_setup()

        for token in (5, 6, 9, 7, 8, 10):
            processor._maybe_update_reasoning_tokens(req, token)

        self.assertEqual(req.reasoning_tokens, 5)
        self.assertTrue(req._is_reasoning_over)


if __name__ == "__main__":
    unittest.main()
