"""Unit tests for Spec V2 grammar trimming in process_batch_result_decode."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeGrammar:
    """Grammar stub that reports termination after `terminate_after` tokens."""

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


class _FakeBatch:
    def __init__(self, reqs, return_logprob: bool):
        self.reqs = reqs
        self.return_logprob = return_logprob
        self.spec_algorithm = _FakeSpecAlgorithm()

    def batch_size(self) -> int:
        return len(self.reqs)


class _TrimmingProcessor(SchedulerBatchResultProcessor):
    """Runs the real result processor with GPU/IO-bound helpers stubbed."""

    def _normalize_decode_outputs(
        self, *, batch, result, logits_output, next_token_ids
    ):
        return result.test_next_token_ids, result.test_next_token_logprobs

    def _mamba_prefix_cache_update(self, req, batch, result, i):
        pass

    def _handle_finish_state_updated_req(self, req, batch, result, i, logits_output):
        pass


def _make_processor() -> _TrimmingProcessor:
    metrics_reporter = SimpleNamespace(
        num_generated_tokens=0,
        forward_ct_decode=0,
        update_spec_metrics=lambda *a, **k: None,
        report_decode_stats=lambda *a, **k: None,
    )
    allocator = SimpleNamespace(
        free_group_begin=lambda: None,
        free_group_end=lambda: None,
    )
    output_streamer = SimpleNamespace(stream_output=lambda *a, **k: None)
    return _TrimmingProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False),
        model_config=SimpleNamespace(think_end_id=None),
        token_to_kv_pool_allocator=allocator,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=None,
        logprob_result_processor=None,
        output_streamer=output_streamer,
        abort_request=lambda *a, **k: None,
    )


def _make_result(accept_tokens, logprobs):
    return SimpleNamespace(
        copy_done=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(hidden_states=None, customized_info=None),
        next_token_ids=None,
        can_run_cuda_graph=False,
        num_correct_drafts=len(accept_tokens),
        test_next_token_ids=[list(accept_tokens)],
        test_next_token_logprobs=[list(logprobs)],
    )


class TestSpecV2GrammarTrimming(CustomTestCase):
    def _make_req(self, terminate_after: int) -> Req:
        sp = SamplingParams(max_new_tokens=256, temperature=0)
        sp.normalize(None)
        req = Req(
            rid="r0",
            origin_input_text="",
            origin_input_ids=[1, 2, 3],
            sampling_params=sp,
        )
        req.vocab_size = 32000
        req.return_logprob = True
        req.logprob.output_token_logprobs_val = []
        req.logprob.output_token_logprobs_idx = []
        req.grammar = _FakeGrammar(terminate_after=terminate_after)
        return req

    def test_trims_tokens_after_grammar_completion(self):
        req = self._make_req(terminate_after=2)
        proc = _make_processor()
        result = _make_result([101, 102, 103], [-0.1, -0.2, -0.3])
        req.kv_committed_len = len(req.origin_input_ids) + len(
            result.test_next_token_ids[0]
        )
        batch = _FakeBatch([req], return_logprob=True)

        proc.process_batch_result_decode(batch, result)

        self.assertTrue(req.finished())
        self.assertEqual(list(req.output_ids), [101, 102])
        self.assertEqual(result.test_next_token_ids[0], [101, 102])
        self.assertEqual(req.grammar.accepted, [101, 102])
        self.assertTrue(req.grammar.finished)
        self.assertEqual(req.logprob.output_token_logprobs_val, [-0.1, -0.2])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [101, 102])
        self.assertEqual(req.kv_committed_len, len(req.origin_input_ids) + 2)

    def test_keeps_all_tokens_when_grammar_not_terminated(self):
        req = self._make_req(terminate_after=99)
        proc = _make_processor()
        result = _make_result([201, 202, 203], [-0.5, -0.6, -0.7])
        req.kv_committed_len = len(req.origin_input_ids) + len(
            result.test_next_token_ids[0]
        )
        batch = _FakeBatch([req], return_logprob=True)

        proc.process_batch_result_decode(batch, result)

        self.assertFalse(req.finished())
        self.assertEqual(list(req.output_ids), [201, 202, 203])
        self.assertEqual(result.test_next_token_ids[0], [201, 202, 203])
        self.assertEqual(req.grammar.accepted, [201, 202, 203])
        self.assertFalse(req.grammar.finished)
        self.assertEqual(req.logprob.output_token_logprobs_val, [-0.5, -0.6, -0.7])
        self.assertEqual(req.logprob.output_token_logprobs_idx, [201, 202, 203])
        self.assertEqual(req.kv_committed_len, len(req.origin_input_ids) + 3)


if __name__ == "__main__":
    unittest.main()
