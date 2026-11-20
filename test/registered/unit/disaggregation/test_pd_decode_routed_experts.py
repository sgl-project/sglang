"""Unit tests for PD decode routed-experts collection boundaries."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.runtime_context import get_parallel, publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.state_capturer.base import BaseTopkCapturer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _StubCapturer:
    """Minimal host-cache capturer backed by BaseTopkCapturer.get_topk."""

    def __init__(self, host_buffer: torch.Tensor):
        self.has_host_cache = True
        self.host_cache = SimpleNamespace(buffer=host_buffer)

    def get_topk(self, req_pool_idx, seqlen, req_to_token_pool, start_len=0):
        return BaseTopkCapturer.get_topk(
            self,
            req_pool_idx=req_pool_idx,
            seqlen=seqlen,
            req_to_token_pool=req_to_token_pool,
            start_len=start_len,
        )


def _make_processor(mode: DisaggregationMode, req_to_token: torch.Tensor):
    proc = object.__new__(SchedulerBatchResultProcessor)
    object.__setattr__(proc, "disaggregation_mode", mode)
    object.__setattr__(
        proc, "req_to_token_pool", SimpleNamespace(req_to_token=req_to_token)
    )
    return proc


def _make_req(
    prompt_ids: list[int],
    output_ids: list[int],
    *,
    return_routed_experts: bool = True,
    routed_experts_start_len: int = 0,
    req_pool_idx: int = 0,
) -> Req:
    req = Req(
        rid="req-0",
        origin_input_text="",
        origin_input_ids=prompt_ids,
        sampling_params=SamplingParams(),
        return_routed_experts=return_routed_experts,
        routed_experts_start_len=routed_experts_start_len,
    )
    req.output_ids = list(output_ids)
    req.kv.req_pool_idx = req_pool_idx
    return req


class TestPDDecodeRoutedExperts(CustomTestCase):
    def test_decode_collects_only_decode_rows_and_concat_matches_non_pd(self):
        prompt_len = 5
        output_len = 4
        seqlen = prompt_len + output_len
        num_layers, topk = 2, 2

        # Ground-truth routing for all tokens [0, seqlen - 1).
        truth = torch.arange(
            1, (seqlen - 1) * num_layers * topk + 1, dtype=torch.int32
        ).reshape(seqlen - 1, num_layers, topk)
        req_to_token = torch.arange(seqlen, dtype=torch.int32).unsqueeze(0)

        # Prefill host cache only wrote prompt tokens [0, prompt_len).
        prefill_host = torch.zeros((seqlen, num_layers, topk), dtype=torch.int32)
        prefill_host[:prompt_len] = truth[:prompt_len]

        # Decode host cache never ran prompt tokens (stale/zero in [0, prompt_len)),
        # and only wrote decode tokens [prompt_len, seqlen - 1).
        decode_host = torch.zeros((seqlen, num_layers, topk), dtype=torch.int32)
        decode_host[:prompt_len] = 999  # stale sentinel from prior evicted reqs
        decode_host[prompt_len : seqlen - 1] = truth[prompt_len : seqlen - 1]

        for start_len in (0, 2, prompt_len):
            with self.subTest(start_len=start_len):
                p_req = _make_req(
                    list(range(prompt_len)),
                    [100],
                    routed_experts_start_len=start_len,
                )
                d_req = _make_req(
                    list(range(prompt_len)),
                    [100, 101, 102, 103],
                    routed_experts_start_len=start_len,
                )

                p_proc = _make_processor(DisaggregationMode.PREFILL, req_to_token)
                d_proc = _make_processor(DisaggregationMode.DECODE, req_to_token)

                with patch(
                    "sglang.srt.managers.scheduler_components.batch_result_processor.get_global_experts_capturer",
                    return_value=_StubCapturer(prefill_host),
                ):
                    p_proc._maybe_collect_routed_experts(p_req)

                with (
                    patch(
                        "sglang.srt.managers.scheduler_components.batch_result_processor.get_global_experts_capturer",
                        return_value=_StubCapturer(decode_host),
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.batch_result_processor.logger.warning"
                    ) as warn_mock,
                ):
                    d_proc._maybe_collect_routed_experts(d_req)
                    warn_mock.assert_not_called()

                self.assertEqual(p_req.routed_experts.shape[0], prompt_len - start_len)
                self.assertEqual(d_req.routed_experts.shape[0], output_len - 1)
                merged = torch.cat([p_req.routed_experts, d_req.routed_experts], dim=0)
                torch.testing.assert_close(merged, truth[start_len : seqlen - 1])

    def test_decode_single_output_token_returns_empty_rows_without_warning(self):
        prompt_len = 6
        req_to_token = torch.arange(prompt_len + 1, dtype=torch.int32).unsqueeze(0)
        decode_host = torch.full((prompt_len + 1, 2, 2), 77, dtype=torch.int32)
        d_req = _make_req(list(range(prompt_len)), [200], routed_experts_start_len=0)
        d_proc = _make_processor(DisaggregationMode.DECODE, req_to_token)

        with (
            patch(
                "sglang.srt.managers.scheduler_components.batch_result_processor.get_global_experts_capturer",
                return_value=_StubCapturer(decode_host),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.batch_result_processor.logger.warning"
            ) as warn_mock,
        ):
            d_proc._maybe_collect_routed_experts(d_req)
            warn_mock.assert_not_called()

        self.assertEqual(d_req.routed_experts.shape, (0, 2, 2))

    def test_scheduler_handle_generate_request_clamps_decode_start_len(self):
        def _run_arrival(mode: DisaggregationMode, start_len: int):
            queued: list[Req] = []
            sched = object.__new__(Scheduler)
            sched.disaggregation_mode = mode
            sched.max_req_input_len = 1024
            sched.max_req_len = 2048
            sched.model_config = SimpleNamespace(
                hf_eos_token_id=None,
                vocab_size=32000,
                token_id_vocab_size=32000,
                is_encoder_decoder=False,
            )
            sched.tokenizer = None
            sched.is_generation = True
            sched.is_hybrid = False
            sched.enable_metrics = False
            sched.metrics_reporter = SimpleNamespace(enable_metrics=False)
            sched.metrics_collector = None
            sched.dllm_config = None
            sched.enable_session_radix_cache = False
            sched.enable_hierarchical_cache = False
            sched.enable_unified_cache_external_linker = False
            sched.enable_lmcache = False
            sched.tree_cache = SimpleNamespace(cache_controller=None)
            sched.grammar_manager = SimpleNamespace(
                process_req_with_grammar=lambda req: False
            )
            sched.invariant_checker = SimpleNamespace(expect_output=lambda req: None)
            sched.req_to_token_pool = SimpleNamespace()
            sched.lora_manager = None
            sched.mm_receiver = None
            sched.session_controller = None
            sched.spec_algorithm = SpeculativeAlgorithm.NONE
            sched.init_req_max_new_tokens = lambda req: None
            sched._add_request_to_queue = lambda req: queued.append(req)
            sched._maybe_namespace_elastic_radix_cache = lambda req: None
            sched._record_kv_event_lora_name = lambda req, name: None
            sched._bind_pp_arrival_prefetch = lambda req, recv_req: None
            sched._create_generation_req = lambda recv_req, *args, **kwargs: Req(
                *args, **kwargs
            )

            recv_req = TokenizedGenerateReqInput(
                rid="rid-1",
                input_text="",
                input_ids=[10, 11, 12, 13],
                mm_inputs=None,
                sampling_params=SamplingParams(max_new_tokens=8),
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
                input_embeds=None,
                token_type_ids=None,
                bootstrap_host="127.0.0.1",
                bootstrap_port=31001,
                bootstrap_room=12345,
                return_routed_experts=True,
                routed_experts_start_len=start_len,
            )
            publish(
                ServerArgs(
                    model_path="dummy",
                    enable_return_routed_experts=True,
                    skip_tokenizer_init=True,
                ),
                role="scheduler",
            )
            try:
                with get_parallel().override(tp_rank=0):
                    sched.handle_generate_request(recv_req)
            finally:
                reset_context()
            self.assertEqual(len(queued), 1)
            return queued[0]

        # Valid DECODE requests clamp start_len to prompt length (4).
        for start in (0, 2, 4):
            req = _run_arrival(DisaggregationMode.DECODE, start)
            self.assertIsNone(req.to_finish)
            self.assertEqual(req.routed_experts_start_len, 4)

        # Invalid start_len is still rejected on DECODE before clamping.
        for bad_start in (-1, 5):
            req = _run_arrival(DisaggregationMode.DECODE, bad_start)
            self.assertIsInstance(req.to_finish, FINISH_ABORT)

        # PREFILL and NULL preserve caller start_len unchanged.
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.NULL):
            for start in (0, 2, 4):
                req = _run_arrival(mode, start)
                self.assertIsNone(req.to_finish)
                self.assertEqual(req.routed_experts_start_len, start)


if __name__ == "__main__":
    unittest.main()
