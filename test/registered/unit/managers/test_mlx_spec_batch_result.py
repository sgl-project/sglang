from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


def _request(committed):
    return SimpleNamespace(
        rid="request",
        is_retracted=False,
        finished=lambda: False,
        grammar=None,
        kv_committed_len=committed,
        spec_verify_ct=0,
        spec_num_correct_drafts=0,
        spec_num_block_accept_tokens=0,
        spec_num_cap_tokens=0,
        update_spec_correct_drafts_histogram=mock.Mock(),
        update_spec_cap_lens_histogram=mock.Mock(),
    )


def _processor(worker=None):
    worker = worker or mock.Mock()
    worker.on_verify_complete_cpu.return_value = None
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=mock.Mock(),
        model_config=mock.Mock(),
        token_to_kv_pool_allocator=mock.Mock(),
        tree_cache=mock.Mock(),
        hisparse_coordinator=None,
        req_to_token_pool=mock.Mock(),
        decode_offload_manager=None,
        metrics_collector=mock.Mock(),
        metrics_reporter=mock.Mock(),
        draft_worker=worker,
        model_worker=worker,
        logprob_result_processor=mock.Mock(),
        output_streamer=mock.Mock(),
        abort_request=mock.Mock(),
    )


class TestMlxSpecBatchResult(unittest.TestCase):
    def test_prefill_finish_invokes_spec_and_native_release_hooks(self):
        worker = BaseSpecWorker()
        worker.note_request_finished = mock.Mock()
        worker.prepare_for_kv_cache_release = mock.Mock()
        worker.on_verify_complete_cpu = mock.Mock()
        processor = _processor(worker)
        finished = False

        def update_finish_state():
            nonlocal finished
            finished = True
            request.finished_reason = FINISH_MATCHED_TOKEN(matched=7)

        request = SimpleNamespace(
            rid="prefill-finish",
            finished=lambda: finished,
            finished_reason=None,
            inflight_middle_chunks=0,
            is_retracted=False,
            time_stats=mock.Mock(),
            output_ids=[],
            require_reasoning=False,
            update_finish_state=update_finish_state,
            return_routed_experts=False,
            return_sampling_mask=False,
            return_hidden_states=False,
            grammar=None,
        )
        batch = SimpleNamespace(
            reqs=[request],
            return_logprob=False,
            decoding_reqs=None,
            prefill_stats=mock.Mock(),
            dp_cooperation_info=None,
        )
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.tensor([7], dtype=torch.long),
        )
        with mock.patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache"
        ) as release:
            processor.process_batch_result_prefill(batch, result)

        worker.note_request_finished.assert_called_once_with(
            rid=request.rid, natural_stop=True
        )
        worker.prepare_for_kv_cache_release.assert_called_once_with(request)
        release.assert_called_once_with(request, processor.tree_cache)

    def test_processor_commits_accept_length_exactly_once(self):
        processor = _processor()
        for padded, accept_len, expected_tokens in (
            ([11, 12], 2, [11, 12]),
            ([13, -1], 1, [13]),
        ):
            with self.subTest(accept_len=accept_len):
                request = _request(committed=7)
                batch = SimpleNamespace(
                    reqs=[request],
                    has_grammar=False,
                    forward_mode=SimpleNamespace(
                        is_decode=lambda: True, is_extend=lambda: False
                    ),
                )
                result = GenerationBatchResult(
                    next_token_ids=torch.tensor(padded, dtype=torch.long),
                    accept_lens=torch.tensor([accept_len], dtype=torch.int32),
                    speculative_num_draft_tokens=2,
                )

                # The MLX worker/native runner owns private token/cache state but
                # intentionally leaves scheduler ownership untouched.
                self.assertEqual(request.kv_committed_len, 7)
                tokens = processor._resolve_spec_v2_tokens(result, batch)
                self.assertEqual(tokens, [expected_tokens])
                self.assertEqual(request.kv_committed_len, 7 + accept_len)
                self.assertEqual(request.spec_verify_ct, 1)
                self.assertEqual(request.spec_num_correct_drafts, accept_len - 1)

    def test_stride_prevents_rejected_padding_from_escaping(self):
        processor = _processor()
        requests = [_request(3), _request(4)]
        requests[1].rid = "second"
        batch = SimpleNamespace(
            reqs=requests,
            has_grammar=False,
            forward_mode=SimpleNamespace(
                is_decode=lambda: True, is_extend=lambda: False
            ),
        )
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([8, -1, 9, 10]),
            accept_lens=torch.tensor([1, 2], dtype=torch.int32),
            speculative_num_draft_tokens=2,
        )
        self.assertEqual(
            processor._resolve_spec_v2_tokens(result, batch), [[8], [9, 10]]
        )


if __name__ == "__main__":
    unittest.main()
