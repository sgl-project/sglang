"""Beam candidates retain scheduler ranking and Python response semantics."""

import base64
import json
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from sglang.srt.beam_search.beam_group import BeamGroup, BeamResult
from sglang.srt.beam_search.output import (
    build_beam_search_out,
    try_build_beam_search_out_dict,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.rust_server.server import RustServer
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.weight_versions import add_weight_versions_to_meta_info
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

FIXTURE_DIR = Path(__file__).resolve().parents[4] / "rust/sglang-server/testdata"


def _requests(skip_special_tokens, no_stop_trim):
    requests = []
    for name in ("ordinary", "best-stop", "best-length"):
        req = Req(
            name,
            "",
            array("q", [5]),
            SamplingParams(
                skip_special_tokens=skip_special_tokens, no_stop_trim=no_stop_trim
            ),
        )
        req.output_ids.extend([3] if name == "ordinary" else [0, 0, 0])
        req.finished_reason = FINISH_LENGTH(len(req.output_ids))
        if name != "ordinary":
            group = BeamGroup(beam_width=2, max_new_tokens=3)
            group.leader = req
            sequences = [([3, 2], 2), ([7, 3, 5], None)]
            if name == "best-length":
                sequences.reverse()
            group.final_results = [
                BeamResult(tokens, -0.25 * (i + 1), -0.125 * (i + 1), matched)
                for i, (tokens, matched) in enumerate(sequences)
            ]
            req.beam_group = group
        requests.append(req)
    return requests


def _fixture_data():
    tokenizer_config = json.loads((FIXTURE_DIR / "decoder_python.json").read_text())[1]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(tokenizer_config["tokenizer"])),
        clean_up_tokenization_spaces=False,
    )
    cases = []
    override = get_context().override_server_args(weight_version="beam-fixture")
    override.install()
    try:
        for skip_tokenizer in (False, True):
            for skip_special in (False, True):
                for no_trim in (False, True):
                    case = {
                        "skip_tokenizer": skip_tokenizer,
                        "skip_special_tokens": skip_special,
                        "no_stop_trim": no_trim,
                    }
                    for rust in (False, True):
                        native, python_egress = Mock(), Mock()
                        streamer = SchedulerOutputStreamer(
                            send_to_detokenizer=python_egress,
                            tree_cache=SimpleNamespace(),
                            ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
                            server_args=get_context().server_args,
                            is_generation=True,
                            spec_algorithm=SpeculativeAlgorithm.NONE,
                            disaggregation_mode=DisaggregationMode.NULL,
                            enable_hicache_storage=lambda: False,
                            rust_server=(
                                RustServer(native, http_port=30000) if rust else None
                            ),
                        )
                        streamer.stream_output(_requests(skip_special, no_trim), False)
                        if rust:
                            header, buffers = (
                                native.push_decode_result_batch.call_args.args
                            )
                            case["header"] = msgspec.msgpack.decode(header)
                            case["data_b64"] = base64.b64encode(
                                b"".join(buffers)
                            ).decode()
                            continue
                        payload = python_egress.send_output.call_args.args[0]
                        if not skip_tokenizer:
                            manager = object.__new__(DetokenizerManager)
                            manager.tokenizer = tokenizer
                            manager.vocab_size = len(tokenizer)
                            manager.decode_status = {}
                            manager.disable_tokenizer_batch_decode = False
                            manager.is_tool_call_parser_gpt_oss = False
                            payload = manager.handle_batch_token_id_out(payload)
                        expected = []
                        for i, rid in enumerate(payload.rids):
                            metadata = {
                                "id": rid,
                                "prompt_tokens": payload.prompt_tokens[i],
                                "finish_reason": payload.finished_reasons[i],
                                "weight_version": "beam-fixture",
                                "reasoning_tokens": payload.reasoning_tokens[i],
                                "completion_tokens": payload.completion_tokens[i],
                                "cached_tokens": payload.cached_tokens[i],
                                "cached_tokens_details": payload.cached_tokens_details[
                                    i
                                ],
                                "num_retractions": payload.retraction_counts[i],
                                "dp_rank": payload.dp_ranks[i],
                            }
                            add_weight_versions_to_meta_info(
                                metadata,
                                payload.weight_versions[i],
                                num_output_tokens=payload.completion_tokens[i],
                            )
                            output = try_build_beam_search_out_dict(
                                payload, i, metadata
                            )
                            expected.append(
                                build_beam_search_out(output) if output else None
                            )
                        case["expected"] = expected
                    cases.append(case)
    finally:
        override.restore()
    requests = []
    for body in (
        {"input_ids": [5], "sampling_params": {"beam_width": 4, "n": 2}},
        {"input_ids": [[5], [3, 7]], "sampling_params": {"beam_width": 4, "n": 2}},
        {"input_ids": [5], "sampling_params": {"beam_width": 1, "n": 2}},
    ):
        request = GenerateReqInput(**body)
        request.normalize_batch_and_arguments()
        prompts = (
            [request]
            if request.is_single
            else [request[i] for i in range(request.batch_size)]
        )
        expected = [
            {
                "input_ids": prompt.input_ids,
                "beam_width": prompt.sampling_params["beam_width"],
                "n": prompt.sampling_params["n"],
            }
            for prompt in prompts
            for _ in range(request.parallel_sample_num)
        ]
        requests.append({"body": body, "expected": expected})
    header, data = cases[0]["header"], cases[0]["data_b64"]
    for case in cases:
        assert case.pop("header") == header
        assert case.pop("data_b64") == data
    return {
        "tokenizer_fixture": 1,
        "vocab_size": len(tokenizer),
        "header": header,
        "data_b64": data,
        "requests": requests,
        "cases": cases,
    }


class TestRustBeamOutputs(unittest.TestCase):
    def test_scheduler_and_python_decoder_match_native_fixtures(self):
        expected = json.loads((FIXTURE_DIR / "beam_outputs_python.json").read_text())
        self.assertEqual(_fixture_data(), expected)


if __name__ == "__main__":
    unittest.main()
