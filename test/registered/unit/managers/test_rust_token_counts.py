"""Scheduler accounting survives Rust egress for mixed streaming batches."""

import json
import math
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from sglang.srt import runtime_context as rc
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.rust_server.server import RustServer
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FileSystem:
    pass


class TestRustTokenCounts(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = rc.get_context().override_server_args(
            stream_interval=1, speculative_num_draft_tokens=4
        )
        override.install()
        self.addCleanup(override.restore)

    def test_sampling_support_columns_preserve_nulls_and_stream_offsets(self):
        fixtures = json.loads(
            (
                Path(__file__).resolve().parents[4]
                / "rust/sglang-server/testdata/sampling_masks_python.json"
            ).read_text()
        )["steps"]
        for rust in (False, True):
            native = Mock()
            python_egress = Mock()
            streamer = SchedulerOutputStreamer(
                send_to_detokenizer=python_egress,
                tree_cache=SimpleNamespace(),
                ps=SimpleNamespace(dp_rank=0),
                server_args=rc.get_context().server_args,
                is_generation=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                disaggregation_mode=DisaggregationMode.NULL,
                enable_hicache_storage=lambda: False,
                rust_server=RustServer(native, http_port=30000) if rust else None,
            )
            requests = [
                Req(
                    str(i),
                    "",
                    array("q", [1, 2]),
                    SamplingParams(top_k=4),
                    stream=True,
                    return_sampling_mask=i != 1,
                )
                for i in range(3)
            ]
            for fixture in fixtures:
                for i, req in enumerate(requests):
                    req.output_ids.extend(fixture["tokens"][i])
                    if req.return_sampling_mask:
                        req.output_token_sampling_mask.extend(fixture["masks"][i])
                        req.output_token_sampling_logprobs.extend(
                            fixture["logprobs"][i]
                        )
                streamer.stream_output(requests, return_logprob=False)
                if rust:
                    header, data = native.push_decode_result_batch.call_args.args
                    self.assertEqual(
                        msgspec.msgpack.decode(header)[20], fixture["shapes"]
                    )
                    ids, vals = array("i"), array("f")
                    ids.frombytes(data[-2])
                    vals.frombytes(data[-1])
                    self.assertEqual(list(ids), fixture["ids"])
                    self.assertEqual(
                        [None if math.isnan(v) else v for v in vals], fixture["values"]
                    )
                else:
                    payload = python_egress.send_output.call_args.args[0]
                    self.assertEqual(
                        payload.output_token_sampling_mask, fixture["masks"]
                    )
                    self.assertEqual(
                        payload.output_token_sampling_logprobs, fixture["logprobs"]
                    )

    def test_hidden_state_shapes_match_python_scheduler_output(self):
        fixtures = json.loads(
            (
                Path(__file__).resolve().parents[4]
                / "rust/sglang-server/testdata/hidden_states_python.json"
            ).read_text()
        )["outputs"]
        for rust in (False, True):
            native = Mock()
            python_egress = Mock()
            streamer = SchedulerOutputStreamer(
                send_to_detokenizer=python_egress,
                tree_cache=SimpleNamespace(),
                ps=SimpleNamespace(dp_rank=0),
                server_args=rc.get_context().server_args,
                is_generation=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                disaggregation_mode=DisaggregationMode.NULL,
                enable_hicache_storage=lambda: False,
                rust_server=RustServer(native, http_port=30000) if rust else None,
            )
            requests = []
            for i, fixture in enumerate(fixtures):
                req = Req(
                    str(i),
                    "",
                    array("q", [1, 2]),
                    SamplingParams(),
                    stream=True,
                    return_hidden_states=fixture["mode"],
                )
                req.hidden_states = fixture["stored"]
                req.output_ids.append(3)
                requests.append(req)
            streamer.stream_output(requests, return_logprob=False)
            if not rust:
                payload = python_egress.send_output.call_args.args[0]
                self.assertEqual(
                    payload.output_hidden_states, [f["output"] for f in fixtures]
                )
            else:
                header, data = native.push_decode_result_batch.call_args.args
                columns = msgspec.msgpack.decode(header)
                self.assertEqual(columns[14], [len(f["lengths"]) for f in fixtures])
                self.assertEqual(
                    columns[15], [n for f in fixtures for n in f["lengths"]]
                )
                self.assertEqual(columns[19], [f["shape"] for f in fixtures])
                values = array("f")
                values.frombytes(data[-1])
                self.assertEqual(
                    list(values), [v for f in fixtures for v in f["values"]]
                )

    def test_scheduler_cache_breakdown_and_counts_survive_batch_reordering(self):
        native = Mock()
        python_egress = Mock()
        streamer = SchedulerOutputStreamer(
            send_to_detokenizer=python_egress,
            tree_cache=SimpleNamespace(
                cache_controller=SimpleNamespace(storage_backend=FileSystem())
            ),
            ps=SimpleNamespace(dp_rank=2),
            server_args=rc.get_context().server_args,
            is_generation=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            disaggregation_mode=DisaggregationMode.NULL,
            enable_hicache_storage=lambda: True,
            rust_server=RustServer(native, http_port=30000),
        )
        cached = Req(
            "cached", "", array("q", [11] * 512), SamplingParams(), stream=True
        )
        cold = Req("cold", "", array("q", [12] * 64), SamplingParams(), stream=True)
        cached.origin_input_ids_unpadded = array("q", [21] * 7 + [-101, 2000000000])
        cold.origin_input_ids_unpadded = array("q", [22] * 6)
        cached.cached_tokens = 192
        cached.cached_tokens_device = 128
        cached.cached_tokens_host = 32
        cached.cached_tokens_storage = 32
        cached.mm_image_tokens = 8

        for step, requests in enumerate(([cached, cold], [cold, cached]), start=1):
            cached.reasoning_tokens = step
            cached.retraction_count = step
            cached.spec_verify_ct = step
            cached.spec_num_cap_tokens = 4 * step
            cached.spec_num_block_accept_tokens = step
            cached.spec_correct_drafts_histogram = [step, 0, 0]
            cached.spec_cap_lens_histogram = [0, 0, 0, step]
            cached.customized_info = {"safety_probe_logits": [None, 0.25][:step]}
            cached.output_ids.append(25)
            cold.output_ids.append(26)
            streamer.stream_output(requests, return_logprob=False)
            header, data = native.push_decode_result_batch.call_args.args
            columns = msgspec.msgpack.decode(header)
            self.assertEqual(columns[0], [req.rid for req in requests])
            self.assertEqual(columns[3], [1, 1])
            self.assertEqual(columns[4:16], [[]] * 12)
            counts = columns[16]
            hit, miss = requests.index(cached), requests.index(cold)
            self.assertEqual(counts["cached_tokens"][hit], 192)
            self.assertEqual(counts["cached_tokens"][miss], 0)
            self.assertEqual(counts["reasoning_tokens"][hit], step)
            self.assertEqual(counts["retraction_counts"][hit], step)
            self.assertEqual(counts["retraction_counts"][miss], 0)
            self.assertEqual(
                counts["cached_tokens_details"][hit],
                {
                    "device": 128,
                    "host": 32,
                    "storage": 32,
                    "storage_backend": "FileSystem",
                },
            )
            self.assertIsNone(counts["cached_tokens_details"][miss])
            self.assertEqual(counts["dp_ranks"], [2, 2])
            self.assertEqual(counts["image_tokens"][hit], 8)
            self.assertEqual(counts["image_tokens"][miss], 0)
            self.assertEqual(counts["generation_tokens"], [step, step])
            self.assertEqual(counts["spec_num_draft_tokens"], 4)
            self.assertEqual(counts["spec_verify_ct"][hit], step)
            self.assertEqual(counts["spec_verify_ct"][miss], 0)
            self.assertEqual(counts["spec_num_correct_drafts"][hit], 0)
            self.assertEqual(counts["spec_num_cap_tokens"][hit], 4 * step)
            self.assertEqual(counts["spec_num_block_accept_tokens"][hit], step)
            self.assertEqual(counts["spec_correct_drafts_histogram"][hit], [step, 0, 0])
            self.assertEqual(counts["spec_cap_lens_histogram"][hit], [0, 0, 0, step])
            self.assertEqual(
                columns[17]["safety_probe_logits"][hit], [None] if step == 1 else [0.25]
            )
            self.assertEqual(columns[17]["safety_probe_logits"][miss], [None])
            self.assertEqual(
                columns[18],
                (
                    [list(req.origin_input_ids_unpadded[-5:]) for req in requests]
                    if step == 1
                    else [[], []]
                ),
            )
            self.assertEqual(len(data[0]), 8)
        python_egress.send_output.assert_not_called()

    def test_native_decoder_fixtures_match_python_for_interleaved_requests(self):
        fixtures = json.loads(
            (
                Path(__file__).resolve().parents[4]
                / "rust/sglang-server/testdata/decoder_python.json"
            ).read_text()
        )
        for fixture in fixtures:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer.from_str(json.dumps(fixture["tokenizer"])),
                clean_up_tokenization_spaces=False,
            )
            self.assertEqual(len(tokenizer), fixture["vocab_size"])
            for disable_batch_decode in [False, True]:
                manager = object.__new__(DetokenizerManager)
                manager.tokenizer = tokenizer
                manager.vocab_size = len(tokenizer)
                manager.decode_status = {}
                manager.disable_tokenizer_batch_decode = disable_batch_decode
                manager.is_tool_call_parser_gpt_oss = False
                for step in range(
                    max(len(case["chunks"]) for case in fixture["cases"])
                ):
                    cases = [
                        case for case in fixture["cases"] if step < len(case["chunks"])
                    ]
                    payload = SimpleNamespace(
                        rids=[case["name"] for case in cases],
                        decoded_texts=[""] * len(cases),
                        decode_ids=[
                            (case["prompt_context"] if step == 0 else [])
                            + case["chunks"][step]
                            for case in cases
                        ],
                        read_offsets=[
                            len(case["prompt_context"]) if step == 0 else 0
                            for case in cases
                        ],
                        finished_reasons=[
                            (
                                {
                                    "type": "length",
                                    "length": sum(map(len, case["chunks"])),
                                }
                                if step + 1 == len(case["chunks"])
                                else None
                            )
                            for case in cases
                        ],
                        no_stop_trim=[False] * len(cases),
                        skip_special_tokens=[
                            case["skip_special_tokens"] for case in cases
                        ],
                        spaces_between_special_tokens=[
                            case["spaces_between_special_tokens"] for case in cases
                        ],
                    )
                    with self.subTest(
                        tokenizer=fixture["name"],
                        disable_batch_decode=disable_batch_decode,
                        step=step,
                    ):
                        self.assertEqual(
                            manager._decode_batch_token_id_output(payload),
                            [case["expected"][step] for case in cases],
                        )
                self.assertFalse(manager.decode_status)


if __name__ == "__main__":
    unittest.main()
