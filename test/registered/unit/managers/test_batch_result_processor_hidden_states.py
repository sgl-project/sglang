import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_processor(case, server_mode: str = "full") -> SchedulerBatchResultProcessor:
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
        server_args=SimpleNamespace(
            enable_metrics=False,
            enable_hisparse=False,
        ),
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
        logprob_result_processor=None,
        output_streamer=Mock(),
        abort_request=lambda *args, **kwargs: None,
    )


class _PrefillReq:
    def __init__(self, *, rid: str, inflight_middle_chunks: int, return_hidden_states):
        self.rid = rid
        self.inflight_middle_chunks = inflight_middle_chunks
        self.return_hidden_states = return_hidden_states
        self.hidden_states = []
        self.is_retracted = False
        self.output_ids = []
        self.time_stats = Mock()
        self.return_logprob = False
        self.return_sampling_mask = False
        self.grammar = None
        self.require_reasoning = False
        self.customized_info = None

    def finished(self):
        return False

    def update_finish_state(self):
        return None


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
            return SimpleNamespace(
                copy_done=None,
                auxiliary_host_output=None,
                routed_experts_output=None,
                indexer_topk_output=None,
                logits_output=SimpleNamespace(hidden_states=hidden_states),
                next_token_ids=None,
                can_run_cuda_graph=False,
                num_correct_drafts=0,
                num_block_accept_tokens=0,
                num_cap_tokens=0,
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
