import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import BatchStrOutput
from sglang.srt.managers.multi_tokenizer_mixin import (
    TokenizerWorker,
    _handle_output_by_index,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.utils import TypeBasedDispatcher

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_batch_str_output() -> BatchStrOutput:
    return BatchStrOutput(
        rids=["rid-0", "rid-1"],
        spec_verify_ct=[0, 0],
        spec_num_correct_drafts=[0, 0],
        spec_correct_drafts_histogram=[[], []],
        finished_reasons=[None, {"type": "length"}],
        output_strs=["first", "second"],
        output_ids=[[1], [2]],
        prompt_tokens=[10, 20],
        completion_tokens=[1, 2],
        reasoning_tokens=[0, 0],
        cached_tokens=[3, 4],
        cached_tokens_details=[
            {"device": 3, "host": 0},
            {"device": 1, "host": 3},
        ],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=[0.0, 0.0],
        output_hidden_states=[None, None],
        routed_experts=[None, None],
        indexer_topk=[None, None],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
        retraction_counts=[0, 0],
    )


class TestMultiTokenizerMixin(unittest.TestCase):
    def test_batch_str_output_preserves_cached_tokens_details(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )

    def test_tokenizer_worker_metrics_use_real_disaggregation_mode(self):
        observed = {}

        server_args = SimpleNamespace(
            bucket_e2e_request_latency=None,
            bucket_inter_token_latency=None,
            bucket_time_to_first_token=None,
            crash_dump_folder=None,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            enable_metrics=True,
            encoder_urls=[],
            extra_metric_labels=None,
            language_only=False,
            preferred_sampling_params=None,
            served_model_name="test-model",
            tokenizer_metrics_allowed_custom_labels=None,
        )
        port_args = SimpleNamespace(tokenizer_ipc_name="ipc://test-tokenizer")

        def init_model_config(manager):
            manager.enable_priority_scheduling = False

        def init_request_dispatcher(manager):
            manager._result_dispatcher = TypeBasedDispatcher([])

        def noop(*args, **kwargs):
            pass

        def capture_metric_labels(manager):
            observed["mode_at_metrics_init"] = manager.server_args.disaggregation_mode
            observed["engine_type"] = DisaggregationMode.to_engine_type(
                manager.server_args.disaggregation_mode
            )
            observed["disaggregation_mode"] = manager.disaggregation_mode

        noop_methods = [
            "init_tokenizer_and_processor",
            "init_ipc_channels",
            "init_running_status",
            "init_request_logging_and_dumping",
            "init_weight_update",
            "init_lora",
        ]
        patches = [
            mock.patch.object(TokenizerManager, "init_model_config", init_model_config),
            mock.patch.object(
                TokenizerManager, "init_request_dispatcher", init_request_dispatcher
            ),
            mock.patch.object(
                TokenizerManager,
                "init_metric_collector_watchdog",
                capture_metric_labels,
            ),
            mock.patch.object(TokenizerManager, "_dispatch_to_scheduler"),
            mock.patch(
                "sglang.srt.managers.tokenizer_manager.start_disagg_service",
                side_effect=AssertionError(
                    "TokenizerWorker must not start a duplicate bootstrap service"
                ),
            ),
        ]
        patches.extend(
            mock.patch.object(TokenizerManager, method, noop) for method in noop_methods
        )

        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            TokenizerWorker(server_args, port_args)

        self.assertEqual(observed["mode_at_metrics_init"], "decode")
        self.assertEqual(observed["engine_type"], "decode")
        self.assertEqual(observed["disaggregation_mode"], DisaggregationMode.DECODE)


if __name__ == "__main__":
    unittest.main()
