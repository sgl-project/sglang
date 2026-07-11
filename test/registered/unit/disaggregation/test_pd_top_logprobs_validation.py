"""Unit tests for PD top-logprobs metadata buffer limits."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MAX_PD_TOP_LOGPROBS_NUM,
    MetadataBuffers,
    validate_pd_top_logprobs_num,
)
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req_for_set_buf(top_logprobs_count: int) -> SimpleNamespace:
    return SimpleNamespace(
        metadata_buffer_index=0,
        output_ids=[42],
        cached_tokens=0,
        cached_tokens_device=0,
        cached_tokens_host=0,
        cached_tokens_storage=0,
        multimodal_inputs=None,
        return_logprob=True,
        bootstrap_room=1,
        logprob=SimpleNamespace(
            output_token_logprobs_val=[1.0],
            output_token_logprobs_idx=[1],
            output_top_logprobs_val=[[0.1] * top_logprobs_count],
            output_top_logprobs_idx=[[i for i in range(top_logprobs_count)]],
        ),
        hidden_states_tensor=None,
    )


def _make_tokenizer_manager(mode: DisaggregationMode) -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.disaggregation_mode = mode
    tm.context_len = 4096
    tm.num_reserved_tokens = 0
    tm.allow_auto_truncate = False
    tm.validate_total_tokens = False
    tm.is_generation = True
    tm.model_config = MagicMock(vocab_size=32000)
    tm.server_args = MagicMock(enable_return_hidden_states=False)
    tm.server_args.enable_custom_logit_processor = False
    return tm


def _make_tokenized_req(top_logprobs_num: int) -> TokenizedGenerateReqInput:
    recv = TokenizedGenerateReqInput(
        rid="r1",
        input_text="hi",
        input_ids=array("q", [1, 2, 3]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        return_logprob=True,
        logprob_start_len=0,
        top_logprobs_num=top_logprobs_num,
        token_ids_logprob=None,
        stream=False,
        bootstrap_host="127.0.0.1",
        bootstrap_port=8998,
        bootstrap_room=1,
    )
    recv.time_stats = MagicMock()
    return recv


def _make_prefill_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    scheduler.server_args = MagicMock()
    scheduler.server_args.disaggregation_bootstrap_port = 8998
    scheduler.server_args.enable_session_radix_cache = False
    scheduler.model_config = MagicMock()
    scheduler.model_config.hf_eos_token_id = set()
    scheduler.model_config.vocab_size = 32000
    scheduler.metrics_reporter = MagicMock()
    scheduler.metrics_reporter.enable_metrics = False
    scheduler.dllm_config = None
    scheduler.tokenizer = MagicMock()
    scheduler.output_streamer = MagicMock()
    scheduler.transfer_backend = MagicMock()
    return scheduler


class TestPdTopLogprobsValidation(CustomTestCase):
    def test_metadata_buffers_set_buf_overflows_at_limit_plus_one(self):
        # Documents the fixed PD metadata capacity that validation must protect.
        bufs = MetadataBuffers(size=1, hidden_size=8, hidden_states_dtype=torch.float32)
        req = _make_req_for_set_buf(MAX_PD_TOP_LOGPROBS_NUM + 1)

        with self.assertRaises(RuntimeError):
            bufs.set_buf(req)

    def test_validate_pd_top_logprobs_num_allows_limit(self):
        validate_pd_top_logprobs_num(MAX_PD_TOP_LOGPROBS_NUM)

    def test_validate_pd_top_logprobs_num_allows_zero(self):
        validate_pd_top_logprobs_num(0)

    def test_validate_pd_top_logprobs_num_rejects_over_limit(self):
        with self.assertRaisesRegex(
            ValueError,
            f"top_logprobs_num {MAX_PD_TOP_LOGPROBS_NUM + 1} exceeds the maximum "
            f"{MAX_PD_TOP_LOGPROBS_NUM}",
        ):
            validate_pd_top_logprobs_num(MAX_PD_TOP_LOGPROBS_NUM + 1)

    def test_tokenizer_rejects_over_limit_on_pd_prefill(self):
        tm = _make_tokenizer_manager(DisaggregationMode.PREFILL)
        obj = GenerateReqInput(
            text="hello",
            return_logprob=True,
            top_logprobs_num=MAX_PD_TOP_LOGPROBS_NUM + 1,
        )
        obj.normalize_batch_and_arguments()

        with self.assertRaisesRegex(ValueError, str(MAX_PD_TOP_LOGPROBS_NUM + 1)):
            tm._validate_one_request(obj, [1, 2, 3])

    def test_tokenizer_allows_limit_on_pd_prefill(self):
        tm = _make_tokenizer_manager(DisaggregationMode.PREFILL)
        obj = GenerateReqInput(
            text="hello",
            return_logprob=True,
            top_logprobs_num=MAX_PD_TOP_LOGPROBS_NUM,
        )
        obj.normalize_batch_and_arguments()
        tm._validate_one_request(obj, [1, 2, 3])

    def test_tokenizer_allows_over_limit_on_unified_mode(self):
        tm = _make_tokenizer_manager(DisaggregationMode.NULL)
        obj = GenerateReqInput(
            text="hello",
            return_logprob=True,
            top_logprobs_num=MAX_PD_TOP_LOGPROBS_NUM + 1,
        )
        obj.normalize_batch_and_arguments()
        tm._validate_one_request(obj, [1, 2, 3])

    def test_scheduler_prefill_aborts_over_limit(self):
        scheduler = _make_prefill_scheduler()
        recv = _make_tokenized_req(MAX_PD_TOP_LOGPROBS_NUM + 1)

        scheduler.handle_generate_request(recv)

        scheduler.output_streamer.stream_output.assert_called_once()
        req = scheduler.output_streamer.stream_output.call_args[0][0][0]
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertEqual(req.finished_reason.status_code, 400)
        self.assertIn(str(MAX_PD_TOP_LOGPROBS_NUM + 1), req.finished_reason.message)


if __name__ == "__main__":
    unittest.main()
