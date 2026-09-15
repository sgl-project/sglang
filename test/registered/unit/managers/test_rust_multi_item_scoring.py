"""Native scoring inputs and sparse delimiter logprobs use Python's contract."""

import asyncio
import base64
import json
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.srt.rust_server.server import RustServer
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "rust/sglang-server/testdata/multi_item_scoring_python.json"
)


def scoring_fixture():
    requests = []
    for body in (
        {"input_ids": [1, 2, 3, 4], "multi_item_delimiter_indices": [0, 3]},
        {
            "input_ids": [[1, 2, 3, 4], [5, 6, 7]],
            "multi_item_delimiter_indices": [[0, 3], [1, 2]],
        },
        {
            "input_ids": [[1, 2, 3, 4], [5, 6, 7]],
            "multi_item_delimiter_indices": [[0, 3], [1, 2]],
            "sampling_params": {"n": 2},
        },
    ):
        request = GenerateReqInput(**body)
        request.normalize_batch_and_arguments()
        prompts = (
            [request]
            if request.is_single
            else [request[i] for i in range(request.batch_size)]
        )
        requests.append(
            {
                "body": body,
                "expected": [
                    prompt.multi_item_delimiter_indices
                    for prompt in prompts
                    for _ in range(request.parallel_sample_num)
                ],
            }
        )

    native, python_output = Mock(), Mock()
    override = get_context().override_server_args(enable_mis=True)
    override.install()
    try:
        for rust in (False, True):
            req = Req(
                "delimiter-scores",
                "",
                array("q", [5, 6, 7, 8, 9, 10]),
                SamplingParams(max_new_tokens=0),
                return_logprob=True,
                top_logprobs_num=2,
                token_ids_logprob=[11, 22],
                multi_item_delimiter_indices=[1, 4],
            )
            processor = SchedulerLogprobResultProcessor(
                model_config=SimpleNamespace(vocab_size=256)
            )
            processor.add_input_logprob_return_values(
                0,
                req,
                SimpleNamespace(
                    input_token_logprobs=(-0.5, -0.25),
                    input_top_logprobs_val=[[[-0.25, -0.5], [-0.125, -0.25]]],
                    input_top_logprobs_idx=[[[11, 22], [22, 11]]],
                    input_token_ids_logprobs_val=[[[-0.25, -0.5], [-0.25, -0.125]]],
                    input_token_ids_logprobs_idx=[[[11, 22], [11, 22]]],
                ),
                0,
                2,
                last_prefill_chunk=True,
            )
            req.finished_reason = FINISH_LENGTH(0)
            streamer = SchedulerOutputStreamer(
                send_to_detokenizer=python_output,
                tree_cache=SimpleNamespace(),
                ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
                server_args=get_context().server_args,
                is_generation=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                disaggregation_mode=DisaggregationMode.NULL,
                enable_hicache_storage=lambda: False,
                rust_server=RustServer(native, http_port=30000) if rust else None,
            )
            streamer.stream_output([req], True)
        header, buffers = native.push_decode_result_batch.call_args.args
        payload = python_output.send_output.call_args.args[0]
        obj = GenerateReqInput(top_logprobs_num=2, token_ids_logprob=[11, 22])
        state = ReqState(
            out_list=[], finished=True, event=asyncio.Event(), obj=obj, time_stats=None
        )
        expected = {}
        manager = object.__new__(TokenizerManager)
        manager.convert_logprob_style(expected, state, 2, [11, 22], False, payload, 0)
    finally:
        override.restore()
    return {
        "requests": requests,
        "field_index": TokenizedGenerateReqInput.__struct_fields__.index(
            "multi_item_delimiter_indices"
        )
        + 1,
        "header": msgspec.msgpack.decode(header),
        "data_b64": base64.b64encode(b"".join(buffers)).decode(),
        "expected": expected,
    }


class TestRustMultiItemScoring(unittest.TestCase):
    def test_shared_fixture_matches_python_scoring_and_response(self):
        self.assertEqual(
            json.loads(FIXTURE.read_text()), json.loads(json.dumps(scoring_fixture()))
        )


if __name__ == "__main__":
    unittest.main()
