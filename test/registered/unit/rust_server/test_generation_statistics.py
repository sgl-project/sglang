"""The Rust egress must retain the statistics collected for Python responses."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
    _GenerationStreamAccumulator,
)
from sglang.srt.rust_server.server import RustServer
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_payload(*, rust, storage=False, rank=None, count=2):
    cache = SimpleNamespace(
        enable_hicache_storage=lambda: storage,
        _get_storage_backend_type=lambda: 'test"store',
    )
    accumulator = _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=1,
        get_cached_tokens_details=lambda req: (
            SchedulerOutputStreamer.get_cached_tokens_details(cache, req)
        ),
        current_weight_version=None,
        rust_server_mode=rust,
    )
    for i in range(count):
        req = Req(
            rid=str(i),
            origin_input_text="hi",
            origin_input_ids=array("q", range(16)),
            sampling_params=SamplingParams(max_new_tokens=8),
            stream=True,
        )
        req.output_ids.append(20)
        req.cached_tokens = i * 7
        req.reasoning_tokens = i
        req.retraction_count = i * 2
        if storage and i:
            req.cached_tokens_device = 3
            req.cached_tokens_host = 2
            req.cached_tokens_storage = 2
        if rust:
            req.init_incremental_detokenize = Mock(
                side_effect=AssertionError("Rust must not run Python detokenization")
            )
        accumulator.accept(req=req)
    return accumulator.to_payload(dp_rank=rank, is_idle_batch=count == 0)


def encode(payload):
    sink = Mock()
    RustServer(sink, http_port=0).push_generation(payload)
    header, buffers = sink.push_decode_result_batch.call_args.args
    return msgspec.msgpack.decode(header), buffers


class TestGenerationStatistics(CustomTestCase):
    def test_python_statistics_survive_rust_egress(self):
        for storage, rank in ((False, None), (True, 0)):
            with self.subTest(storage=storage, rank=rank):
                python = make_payload(rust=False, storage=storage, rank=rank)
                rust = make_payload(rust=True, storage=storage, rank=rank)
                expected = [
                    python.cached_tokens,
                    python.cached_tokens_details,
                    python.reasoning_tokens,
                    python.retraction_counts,
                    python.dp_ranks,
                ]
                self.assertEqual(expected[0], [0, 7])
                self.assertIsNone(expected[1][0])
                self.assertEqual(expected[1][1]["device"], 3 if storage else 7)
                self.assertEqual(expected[2:], [[0, 1], [0, 2], [rank, rank]])
                for extras in (False, True):
                    if extras:
                        rust.output_token_logprobs_val = [[-0.5], []]
                        rust.output_token_logprobs_idx = [[20], []]
                    header, buffers = encode(rust)
                    self.assertEqual(len(header), 17)
                    self.assertEqual(header[16], expected)
                    self.assertEqual(
                        header[:4], [["0", "1"], [None, None], [16, 16], [1, 1]]
                    )
                    self.assertEqual(header[4:6], [[1, 0], []] if extras else [[], []])
                    self.assertEqual(buffers[0], array("q", [20, 20]).tobytes())

    def test_idle_and_misaligned_statistics(self):
        header, buffers = encode(make_payload(rust=True, count=0))
        self.assertEqual(header[16], [[], [], [], [], []])
        self.assertEqual(buffers, [b""])
        for name in (
            "cached_tokens",
            "cached_tokens_details",
            "reasoning_tokens",
            "retraction_counts",
            "dp_ranks",
        ):
            with self.subTest(column=name):
                payload = make_payload(rust=True)
                getattr(payload, name).pop()
                with self.assertRaisesRegex(ValueError, "one entry per request"):
                    encode(payload)


if __name__ == "__main__":
    unittest.main()
