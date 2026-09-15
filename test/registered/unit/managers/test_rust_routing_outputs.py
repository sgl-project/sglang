"""Expert and indexer outputs retain Python's request and raw-byte contracts."""

import base64
import json
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import torch

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
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fixtures():
    return json.loads(
        (
            Path(__file__).resolve().parents[4]
            / "rust/sglang-server/testdata/routing_outputs_python.json"
        ).read_text()
    )


class TestRustRoutingOutputs(unittest.TestCase):
    def test_request_flags_offsets_and_parallel_samples_match_python(self):
        fields = (
            "return_routed_experts",
            "routed_experts_start_len",
            "return_indexer_topk",
        )
        for fixture in _fixtures()["requests"]:
            with self.subTest(body=fixture["body"]):
                request = GenerateReqInput(**fixture["body"])
                request.normalize_batch_and_arguments()
                prompts = (
                    [request]
                    if request.is_single
                    else [request[i] for i in range(request.batch_size)]
                )
                self.assertEqual(
                    [
                        [getattr(prompt, key) for key in fields]
                        for prompt in prompts
                        for _ in range(request.parallel_sample_num)
                    ],
                    fixture["expected"],
                )

    def test_mixed_tensor_columns_match_python_base64(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        fixtures = _fixtures()["outputs"]
        for rust in (False, True):
            native, python_egress = Mock(), Mock()
            requests = []
            for i, fixture in enumerate(fixtures):
                req = Req(
                    str(i),
                    "",
                    array("q", [1, 2]),
                    SamplingParams(),
                    return_routed_experts=fixture["routed_experts"] is not None,
                    return_indexer_topk=fixture["indexer_topk"] is not None,
                )
                for key in ("routed_experts", "indexer_topk"):
                    value = fixture[key]
                    tensor = None
                    if value is not None:
                        tensor = torch.tensor(value["values"], dtype=torch.int32)
                        if value["transpose"]:
                            tensor = tensor.T
                            self.assertFalse(tensor.is_contiguous())
                    if key == "routed_experts":
                        req.routed_experts = tensor
                    else:
                        req.indexer_topk = tensor
                req.output_ids.append(3)
                req.finished_reason = FINISH_LENGTH(1)
                requests.append(req)
            streamer = SchedulerOutputStreamer(
                send_to_detokenizer=python_egress,
                tree_cache=SimpleNamespace(),
                ps=SimpleNamespace(dp_rank=0, attn_tp_rank=0),
                server_args=get_context().server_args,
                is_generation=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                disaggregation_mode=DisaggregationMode.NULL,
                enable_hicache_storage=lambda: False,
                rust_server=RustServer(native, http_port=30000) if rust else None,
            )
            streamer.stream_output(requests, return_logprob=False)
            expected_data = []
            for column, key in enumerate(("routed_experts", "indexer_topk"), 22):
                expected = [fixture["expected"].get(key) for fixture in fixtures]
                if rust:
                    header, buffers = native.push_decode_result_batch.call_args.args
                    raw = [
                        None if value is None else base64.b64decode(value)
                        for value in expected
                    ]
                    self.assertEqual(
                        msgspec.msgpack.decode(header)[column],
                        [None if value is None else len(value) for value in raw],
                    )
                    expected_data.extend(value for value in raw if value is not None)
                else:
                    payload = python_egress.send_output.call_args.args[0]
                    self.assertEqual(
                        DetokenizerManager._b64_encode_per_request(
                            getattr(payload, key)
                        ),
                        expected,
                    )
            if rust:
                self.assertEqual(buffers[1:], expected_data)


if __name__ == "__main__":
    unittest.main()
