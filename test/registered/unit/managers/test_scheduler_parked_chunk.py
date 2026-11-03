"""A parked chunk does not make the request's final chunk a middle chunk."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.request_list import RequestList
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.kv = SimpleNamespace(holds_kv=True)
        self.beam_group = None
        self.token_indices_to_pool = None
        self.is_retracted = False
        self.discard_output_reason = None
        self.return_sampling_mask = False
        self.return_logprob = False
        self.grammar = None
        self.require_reasoning = False
        self.customized_info = None
        self.output_ids = []
        self.time_stats = Mock()

    def init_next_round_input(self, tree_cache=None):
        pass

    def finished(self):
        return False

    def update_finish_state(self):
        pass


class _Adder(PrefillAdder):
    """Parks the chunked request (hybrid-SWA joint fit) while `park` is set."""

    park = True

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = 0

    def add_chunked_req(self, req):
        if self.park:
            return req
        self.can_run_list.append(req)
        return None

    def budget_state(self):
        return AddReqResult.CONTINUE

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.can_run_list.append(req)
        return AddReqResult.CONTINUE


class _BatchBuilt(Exception):
    pass


def _scheduler(chunked_req, waiting):
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.waiting_queue = waiting
    s.chunked_req = chunked_req
    s.min_free_slots_delayer = None
    s.policy = Mock()
    s.policy.shortest_prefill_chunk_limit.return_value = None
    s.processed_tokens_counter = 0
    s.chunked_prefill_size = 8192
    s.dynamic_chunk_sizer = None
    s.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
    )
    s.get_prefill_adder_class = lambda: _Adder
    s.page_size = 1
    s.tree_cache = SimpleNamespace(buffer_pipeline=None, storage_prefetch_retries=None)
    s.token_to_kv_pool_allocator = SimpleNamespace(state_slots=None)
    s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
    s.max_prefill_tokens = 1 << 20
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 0
    s.max_running_requests = 4
    s.dllm_config = None
    s.enable_lora = False
    s.disaggregation_mode = DisaggregationMode.NULL
    s.disagg_prefill_bootstrap_queue = None
    s.enable_hicache_storage = False
    s.enable_lmcache = False
    s.truncation_align_size = None
    s.model_config = None
    s.enable_overlap = False
    s.spec_algorithm = None
    s.req_to_token_pool = SimpleNamespace(available_size=lambda: 4)
    s.beam_coordinator = SimpleNamespace(pending_member_rows=lambda batch: 0)
    s.running_reqs = RequestList()
    return s


def _build_prefill(s):
    """Run one admission pass; return the requests and the record it builds."""

    def capture(can_run_list, *args, **kwargs):
        raise _BatchBuilt(list(can_run_list), kwargs["middle_chunk_reqs"])

    with (
        get_context().override_server_args(pp_max_micro_batch_size=4),
        patch.object(ScheduleBatch, "init_new", side_effect=capture),
        patch("sglang.srt.managers.scheduler.set_time_batch"),
    ):
        try:
            s._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None,
                running_batch=ScheduleBatch(reqs=[]),
            )
        except _BatchBuilt as built:
            return built.args
    raise AssertionError("no batch built")


def _processor():
    metrics_reporter = Mock(num_generated_tokens=0, forward_ct_decode=0)
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(
            think_end_ids=None, self_opened_think_start_ids=None
        ),
        token_to_kv_pool_allocator=Mock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        kv_weight_version_tracker=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=None,
        output_streamer=Mock(),
        beam_coordinator=Mock(),
        abort_request=lambda *args, **kwargs: None,
    )


class TestParkedChunk(CustomTestCase):
    def test_final_chunk_after_parked_pass_commits_first_token(self):
        chunked, admitted = _Req("chunked"), _Req("admitted")
        s = _scheduler(chunked_req=chunked, waiting=[admitted])

        # The chunk does not fit this pass, but a smaller request is admitted.
        _Adder.park = True
        self.addCleanup(setattr, _Adder, "park", True)
        reqs, middle_chunk_reqs = _build_prefill(s)
        self.assertEqual(reqs, [admitted])
        self.assertEqual(middle_chunk_reqs, ())
        self.assertIs(s.chunked_req, chunked)

        # The next pass runs the request's final chunk.
        _Adder.park = False
        reqs, middle_chunk_reqs = _build_prefill(s)
        self.assertEqual(reqs, [chunked])
        self.assertIsNone(s.chunked_req)

        batch = SimpleNamespace(
            reqs=reqs,
            middle_chunk_reqs=middle_chunk_reqs,
            decoding_reqs=None,
            return_logprob=False,
            return_hidden_states=False,
            return_hidden_states_mode=CaptureHiddenMode.NULL,
            spec_info=None,
            prefill_stats=None,
        )
        result = SimpleNamespace(
            copy_done=None,
            auxiliary_host_output=None,
            routed_experts_output=None,
            indexer_topk_output=None,
            kv_weight_version_record=None,
            logits_output=SimpleNamespace(
                hidden_states=None, customized_info=None, sampling_mask_output=None
            ),
            next_token_ids=torch.tensor([7]),
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
            grammar_advanced=False,
            can_run_cuda_graph=False,
        )
        with (
            get_context().override_server_args(),
            patch(
                "sglang.srt.managers.scheduler_components."
                "batch_result_processor.checkpoint_kv_cache"
            ),
        ):
            _processor().process_batch_result_prefill(batch, result)

        self.assertEqual(chunked.output_ids, [7])


if __name__ == "__main__":
    unittest.main()
