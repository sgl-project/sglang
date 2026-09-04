import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import ReqLogprob
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _PrefillReq:
    def __init__(self, *, rid, origin_input_ids, is_retracted=False):
        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.logprob_start_len = 0
        self.is_retracted = is_retracted
        self.inflight_middle_chunks = 0
        self.return_logprob = True
        self.logprob = ReqLogprob(
            top_logprobs_num=0,
            token_ids_logprob=None,
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
        )
        self.multi_item_delimiter_indices = None
        self.return_flat_raw_top_logprobs = False
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.temp_input_token_ids_logprobs_val = None
        self.temp_input_token_ids_logprobs_idx = None
        self.hidden_states = []
        self.output_ids = []
        self.time_stats = Mock()
        self.return_hidden_states = False
        self.return_sampling_mask = False
        self.grammar = None
        self.require_reasoning = False
        self.customized_info = None
        self.beam_group = None

    def finished(self):
        return False

    def update_finish_state(self):
        return None


def _make_processor(case):
    override = get_context().override_server_args()
    override.install()
    case.addCleanup(override.restore)
    metrics_reporter = Mock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    model_config = SimpleNamespace(think_end_ids=None, vocab_size=1000)
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=model_config,
        token_to_kv_pool_allocator=Mock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=SchedulerLogprobResultProcessor(
            model_config=model_config,
        ),
        output_streamer=Mock(),
        beam_coordinator=Mock(),
        abort_request=lambda *args, **kwargs: None,
    )


class TestPrefillLogprobOffsets(CustomTestCase):
    def test_retracted_request_advances_offset_for_later_requests(self):
        retracted = _PrefillReq(
            rid="retracted", origin_input_ids=[101, 102], is_retracted=True
        )
        active = _PrefillReq(rid="active", origin_input_ids=[201, 202, 203])

        # The flat array holds one block per request in batch order, whether or
        # not that request is skipped below: retracted owns the first two
        # entries, active owns the last three.
        input_token_logprobs = torch.tensor([-1.0, -2.0, -30.0, -31.0, -32.0])

        batch = SimpleNamespace(
            reqs=[retracted, active],
            decoding_reqs=[],
            return_logprob=True,
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
            logits_output=SimpleNamespace(
                hidden_states=None,
                customized_info=None,
                input_token_logprobs=input_token_logprobs,
                next_token_logprobs=torch.tensor([-0.5, -0.6]),
                input_top_logprobs_val=None,
                input_top_logprobs_idx=None,
                input_token_ids_logprobs_val=None,
                input_token_ids_logprobs_idx=None,
                next_token_top_logprobs_val=None,
                next_token_top_logprobs_idx=None,
                next_token_token_ids_logprobs_val=None,
                next_token_token_ids_logprobs_idx=None,
            ),
            next_token_ids=torch.tensor([7, 8]),
            extend_input_len_per_req=[2, 3],
            extend_logprob_start_len_per_req=[0, 0],
            grammar_advanced=False,
            can_run_cuda_graph=False,
            skipped_output_comm=False,
        )

        processor = _make_processor(self)
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

        # _process_input_token_logprobs drops the sampling token and prepends
        # None, so active's three-entry block surfaces as its first two values.
        self.assertEqual(active.logprob.input_token_logprobs_val, [None, -30.0, -31.0])


if __name__ == "__main__":
    unittest.main()
