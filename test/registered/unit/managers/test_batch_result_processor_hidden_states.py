import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput, SamplingMaskStatus
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_processor(
    case, server_mode: str = "full", *, logprob_result_processor=None
) -> SchedulerBatchResultProcessor:
    # The server-side hidden-state ceiling is a bag leaf.
    override = get_context().override_server_args(
        enable_return_hidden_states=True,
        return_hidden_states_mode=server_mode,
    )
    override.install()
    case.addCleanup(override.restore)
    metrics_reporter = Mock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=logprob_result_processor,
        output_streamer=Mock(),
        beam_coordinator=Mock(),
        abort_request=lambda *args, **kwargs: None,
    )


class TestSamplingMaskMaterialization(CustomTestCase):
    def test_packed_ids_are_copied_before_per_request_slicing(self):
        """Non-overlap capture must not perform one device copy per request."""
        packed_ids = Mock()
        packed_ids.shape = (2, 3)
        packed_ids.cpu.return_value = torch.tensor([[7, 8, 0], [9, 0, 0]])
        output = LogitsProcessorOutput(
            next_token_logits=None,
            sampling_mask_output=SimpleNamespace(
                token_ids=packed_ids,
                lengths=torch.tensor([2, 1]),
                selected_logprobs=torch.tensor([-0.5, -0.25]),
                statuses=torch.tensor([SamplingMaskStatus.OK, SamplingMaskStatus.OK]),
            ),
        )
        SchedulerBatchResultProcessor.materialize_sampling_mask_output(
            reqs=[SimpleNamespace(return_sampling_mask=x) for x in (True, False, True)],
            output=output,
        )
        packed_ids.cpu.assert_called_once_with()
        self.assertEqual(output.next_token_sampling_mask_idx, [[7, 8], None, [9]])
        self.assertEqual(output.next_token_sampling_logprobs, [-0.5, None, -0.25])
        self.assertIsNone(output.sampling_mask_output)


class _PrefillReq:
    def __init__(
        self,
        *,
        rid: str,
        inflight_middle_chunks: int,
        return_hidden_states,
        return_logprob: bool = False,
        to_finish=None,
    ):
        self.rid = rid
        self.inflight_middle_chunks = inflight_middle_chunks
        self.return_hidden_states = return_hidden_states
        self.hidden_states = []
        self.is_retracted = False
        self.output_ids = []
        self.to_finish = to_finish
        self.finished_reason = None
        self.time_stats = Mock()
        self.return_logprob = return_logprob
        self.return_sampling_mask = False
        self.grammar = None
        self.require_reasoning = False
        self.customized_info = None
        self.beam_group = None
        self.return_routed_experts = False

    def finished(self):
        return self.finished_reason is not None

    def update_finish_state(self, new_accepted_len=1):
        if self.to_finish is not None:
            self.finished_reason = self.to_finish
            self.to_finish = None


class _DecodeReq:
    def __init__(self):
        self.return_hidden_states = "last"
        self.hidden_states = []
        self.output_ids = []
        self.finished_len = None
        self.is_retracted = False
        self.return_logprob = False
        self.return_sampling_mask = False
        self.grammar = None
        self.beam_group = None
        self.time_stats = Mock()

    def finished(self):
        return self.finished_len is not None

    def update_finish_state(self, new_accept_len):
        if len(self.output_ids) >= 6:
            self.finished_len = 5


class TestPrefillHiddenStateOffsets(CustomTestCase):
    def test_active_middle_chunk_advances_before_new_last_request(self):
        cases = (
            (
                "full",
                CaptureHiddenMode.FULL,
                torch.tensor([[10.0], [11.0], [20.0], [21.0], [22.0]]),
            ),
            (
                "last",
                CaptureHiddenMode.LAST,
                torch.tensor([[11.0], [22.0]]),
            ),
        )

        for server_mode, capture_mode, hidden_states in cases:
            with self.subTest(server_mode=server_mode):
                middle = _PrefillReq(
                    rid="middle",
                    inflight_middle_chunks=1,
                    return_hidden_states=False,
                )
                last = _PrefillReq(
                    rid="last",
                    inflight_middle_chunks=0,
                    return_hidden_states="last",
                )
                batch = SimpleNamespace(
                    reqs=[middle, last],
                    decoding_reqs=[],
                    return_logprob=False,
                    return_hidden_states=True,
                    return_hidden_states_mode=capture_mode,
                    spec_info=None,
                    prefill_stats=None,
                    dp_cooperation_info=None,
                )
                result = SimpleNamespace(
                    copy_done=None,
                    auxiliary_host_output=None,
                    routed_experts_output=None,
                    indexer_topk_output=None,
                    logits_output=SimpleNamespace(
                        hidden_states=hidden_states,
                        customized_info=None,
                        sampling_mask_output=None,
                    ),
                    next_token_ids=torch.tensor([0, 1]),
                    extend_input_len_per_req=[2, 3],
                    extend_logprob_start_len_per_req=None,
                    grammar_advanced=False,
                    can_run_cuda_graph=False,
                    skipped_output_comm=False,
                )
                processor = _make_processor(self, server_mode)

                with (
                    patch(
                        "sglang.srt.managers.scheduler_components."
                        "batch_result_processor.maybe_cache_unfinished_req"
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components."
                        "batch_result_processor.get_memory",
                        return_value=SimpleNamespace(enable_hisparse=False),
                    ),
                ):
                    processor.process_batch_result_prefill(batch, result)

                self.assertEqual(middle.hidden_states, [])
                self.assertEqual(last.hidden_states, [[22.0]])

    def test_aborted_final_prefill_drops_metadata_but_advances_logprob_offset(self):
        self._assert_aborted_final_prefill()

    def test_pending_abort_wins_over_sampling_mask_overflow(self):
        self._assert_aborted_final_prefill(SamplingMaskStatus.OVERFLOW)

    def _assert_aborted_final_prefill(self, sampling_status=None):
        aborted = _PrefillReq(
            rid="aborted",
            inflight_middle_chunks=0,
            return_hidden_states="last",
            return_logprob=True,
            to_finish=object(),
        )
        abort_reason = aborted.to_finish
        aborted.return_sampling_mask = sampling_status is not None
        live = _PrefillReq(
            rid="live",
            inflight_middle_chunks=0,
            return_hidden_states="last",
            return_logprob=True,
        )
        batch = SimpleNamespace(
            reqs=[aborted, live],
            decoding_reqs=[],
            return_logprob=True,
            return_hidden_states=True,
            return_hidden_states_mode=CaptureHiddenMode.LAST,
            spec_info=None,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        logits_output = SimpleNamespace(
            sampling_mask_output=None,
            next_token_sampling_mask_status=[sampling_status, None],
            hidden_states=torch.tensor([[10.0], [20.0]]),
            customized_info={"tag": ["abort-meta", "live-meta"]},
            next_token_logprobs=None,
            input_token_logprobs=torch.tensor([-1.0, -2.0, -10.0, -11.0, -12.0]),
            next_token_top_logprobs_val=[],
            next_token_top_logprobs_idx=[],
            next_token_token_ids_logprobs_val=[],
        )
        result = SimpleNamespace(
            copy_done=None,
            auxiliary_host_output=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=logits_output,
            next_token_ids=torch.tensor([101, 202]),
            extend_input_len_per_req=[2, 3],
            extend_logprob_start_len_per_req=[0, 0],
            grammar_advanced=False,
            can_run_cuda_graph=False,
            skipped_output_comm=False,
        )
        logprob_processor = Mock()
        logprob_processor.calculate_num_input_logprobs.side_effect = [2, 3]
        processor = _make_processor(
            self, "last", logprob_result_processor=logprob_processor
        )

        def assert_clean_before_stream(reqs, *_):
            self.assertEqual(reqs[0].output_ids, [])
            self.assertEqual(reqs[0].hidden_states, [])
            self.assertIsNone(reqs[0].customized_info)

        processor.output_streamer.stream_output.side_effect = assert_clean_before_stream

        with (
            patch(
                "sglang.srt.managers.scheduler_components."
                "batch_result_processor.release_kv_cache"
            ),
            patch(
                "sglang.srt.managers.scheduler_components."
                "batch_result_processor.maybe_cache_unfinished_req"
            ),
            patch(
                "sglang.srt.managers.scheduler_components."
                "batch_result_processor.get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
        ):
            processor.process_batch_result_prefill(batch, result)

        self.assertTrue(aborted.finished())
        self.assertIs(aborted.finished_reason, abort_reason)
        self.assertEqual(aborted.output_ids, [])
        self.assertEqual(aborted.hidden_states, [])
        self.assertIsNone(aborted.customized_info)
        self.assertEqual(live.output_ids, [202])
        self.assertEqual(live.hidden_states, [[20.0]])
        self.assertEqual(live.customized_info, {"tag": ["live-meta"]})

        self.assertEqual(logprob_processor.calculate_num_input_logprobs.call_count, 2)
        logprob_processor.add_logprob_return_values.assert_called_once_with(
            1,
            live,
            2,
            [101, 202],
            3,
            logits_output,
        )


class TestPrefillSkippedOutput(CustomTestCase):
    def test_sampling_mask_middle_chunk_does_not_require_logits_output(self):
        """A non-token-producing PP chunk may omit its logits output."""
        req = _PrefillReq(
            rid="middle",
            inflight_middle_chunks=1,
            return_hidden_states=False,
        )
        req.return_sampling_mask = True
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            return_hidden_states=False,
            return_hidden_states_mode=CaptureHiddenMode.NULL,
            spec_info=None,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            auxiliary_host_output=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            logits_output=None,
            next_token_ids=torch.zeros(1, dtype=torch.int64),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            grammar_advanced=False,
            can_run_cuda_graph=False,
            skipped_output_comm=True,
        )
        processor = _make_processor(self)

        with patch.object(
            envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM,
            "get",
            return_value=True,
        ):
            processor.process_batch_result_prefill(batch, result)

        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertEqual(req.output_ids, [])
        processor.output_streamer.stream_output.assert_called_once_with(
            [req], False, req
        )


class TestDecodeWithoutLogits(CustomTestCase):
    def test_pipeline_result_commits_token_without_sampling_metadata(self):
        processor = _make_processor(self)
        req = _DecodeReq()
        req.return_hidden_states = False
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            batch_size=lambda: 1,
        )
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([8]),
        )

        with (
            patch.object(
                SchedulerBatchResultProcessor, "_maybe_update_reasoning_tokens"
            ),
            patch.object(
                SchedulerBatchResultProcessor, "_handle_finish_state_updated_req"
            ),
        ):
            processor.process_batch_result_decode(batch, result)

        self.assertEqual(req.output_ids, [8])
        self.assertEqual(processor.metrics_reporter.num_generated_tokens, 1)
        processor.output_streamer.stream_output.assert_called_once_with([req], False)


class TestSamplingMaskStatusErrors(CustomTestCase):
    def test_decode_abort_releases_cache_without_committing_token(self):
        processor = _make_processor(self)
        req = _DecodeReq()
        req.output_ids = [7]
        req.return_sampling_mask = True
        req.multimodal_inputs = None
        req.update_finish_state = Mock()
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            batch_size=lambda: 1,
        )
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=None,
                next_token_sampling_mask_status=[SamplingMaskStatus.OVERFLOW],
            ),
            next_token_ids=torch.tensor([8]),
        )
        with patch(
            "sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache"
        ) as release:
            processor.process_batch_result_decode(batch, result)

        self.assertEqual(req.output_ids, [7])
        self.assertEqual(req.to_finish.status_code, 400)
        req.update_finish_state.assert_called_once_with(0)
        processor.model_worker.prepare_for_kv_cache_release.assert_called_once_with(req)
        release.assert_called_once_with(req, processor.tree_cache, is_insert=False)
        processor.output_streamer.stream_output.assert_called_once_with([req], False)

    def test_overflow_and_invalid_have_distinct_http_errors(self):
        processor = _make_processor(self)

        overflow = processor.get_sampling_mask_finish_reason(
            status=SamplingMaskStatus.OVERFLOW
        )
        self.assertEqual(overflow.status_code, 400)
        self.assertEqual(overflow.err_type, "BadRequestError")
        self.assertIn("cutoff ties", overflow.message)

        invalid = processor.get_sampling_mask_finish_reason(
            status=SamplingMaskStatus.INVALID
        )
        self.assertEqual(invalid.status_code, 500)
        self.assertEqual(invalid.err_type, "InternalServerError")


class TestDecodeHiddenStateRetention(CustomTestCase):
    def test_last_mode_multi_step_storage_stays_bounded(self):
        processor = _make_processor(self)
        req = _DecodeReq()
        batch = SimpleNamespace(
            reqs=[req],
            return_logprob=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            batch_size=lambda: 1,
        )
        first_step = torch.arange(8, dtype=torch.float32).view(4, 2)
        second_step = torch.arange(16, dtype=torch.float32).view(8, 2)[4:]

        def result(hidden_states):
            return GenerationBatchResult(
                logits_output=SimpleNamespace(
                    hidden_states=hidden_states, sampling_mask_output=None
                ),
                speculative_num_draft_tokens=4,
            )

        with (
            patch.object(
                SchedulerBatchResultProcessor,
                "_normalize_decode_outputs",
                side_effect=[
                    ([[1, 2, 3]], None),
                    ([[4, 5, 6]], None),
                ],
            ),
            patch.object(
                SchedulerBatchResultProcessor,
                "_maybe_update_reasoning_tokens",
            ),
            patch.object(
                SchedulerBatchResultProcessor,
                "_handle_finish_state_updated_req",
            ),
            patch(
                "sglang.srt.managers.scheduler_components."
                "batch_result_processor.get_observability",
                return_value=SimpleNamespace(enable_metrics=False),
            ),
        ):
            processor.process_batch_result_decode(batch, result(first_step))

            self.assertEqual(req.hidden_states, [first_step[2].tolist()])
            self.assertEqual(len(req.hidden_states), 1)

            # Only the first two accepted tokens are valid because the request
            # stops inside this speculative verify step.
            processor.process_batch_result_decode(batch, result(second_step))

        self.assertEqual(req.hidden_states, [second_step[1].tolist()])
        self.assertEqual(len(req.hidden_states), 1)


if __name__ == "__main__":
    unittest.main()
