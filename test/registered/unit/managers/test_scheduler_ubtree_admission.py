"""UBTree rejects unsupported requests before admission and stays usable."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _request(rid, *, params=None, **fields):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text="hello",
        input_ids=array("q", [1, 2, 3]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(
            **({"temperature": 0, "max_new_tokens": 32} | (params or {}))
        ),
        return_logprob=fields.pop("return_logprob", False),
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        bootstrap_port=1,
        **fields,
    )


class TestSchedulerUBTreeAdmission(CustomTestCase):
    def test_rejection_does_not_enter_queue_or_poison_next_request(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_session_radix_cache = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.model_config = SimpleNamespace(hf_eos_token_id={0}, vocab_size=128)
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.kv_events_publisher = SimpleNamespace(enable_kv_cache_events=False)
        scheduler.tokenizer = None
        scheduler.dllm_config = None
        scheduler.spec_algorithm = SpeculativeAlgorithm.from_string("UBTREE")
        scheduler._maybe_namespace_elastic_radix_cache = lambda req: None
        scheduler.init_req_max_new_tokens = lambda req: None
        scheduler.max_req_input_len = 128
        scheduler.tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(pp_prefetch_command_group=None)
        )
        scheduler.grammar_manager = SimpleNamespace(
            process_req_with_grammar=Mock(return_value=False)
        )
        scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=Mock())
        scheduler._add_request_to_queue = Mock()

        with (
            get_context().override_server_args(),
            get_parallel().override(tp_rank=0, pp_rank=0),
        ):
            for fields in (
                {"params": {"temperature": 0.8, "top_k": 10}},
                {"params": {"regex": "[a-z]+"}},
                {"params": {"presence_penalty": 0.5}},
                {"return_logprob": True},
                {"custom_logit_processor": "invalid-processor"},
                {"return_hidden_states": True},
                {"return_sampling_mask": True},
            ):
                with self.subTest(fields=fields):
                    scheduler._add_request_to_queue.reset_mock()
                    scheduler.grammar_manager.process_req_with_grammar.reset_mock()
                    scheduler.ipc_channels.send_to_tokenizer.reset_mock()
                    scheduler.handle_generate_request(_request("invalid", **fields))
                    scheduler._add_request_to_queue.assert_not_called()
                    scheduler.grammar_manager.process_req_with_grammar.assert_not_called()
                    sender = scheduler.ipc_channels.send_to_tokenizer.send_output
                    sender.assert_called_once()
                    abort = sender.call_args.args[0]
                    self.assertEqual(abort.finished_reason["status_code"], 400)
                    self.assertIn("UBTREE", abort.finished_reason["message"])
                    scheduler.handle_generate_request(_request("valid"))
                    scheduler._add_request_to_queue.assert_called_once()
                    self.assertEqual(
                        scheduler._add_request_to_queue.call_args.args[0].rid, "valid"
                    )


if __name__ == "__main__":
    unittest.main()
