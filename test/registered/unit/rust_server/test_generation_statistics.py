"""Scheduler statistics survive the native Rust egress boundary without a GPU."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
    _GenerationStreamAccumulator,
)
from sglang.srt.rust_server.server import RustServer
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.weight_versions import WeightVersionEvent, WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def payload(*, rust, details=False, rank=None, count=2, version="default", finished=()):
    cache = SimpleNamespace(
        enable_hicache_storage=lambda: details,
        _get_storage_backend_type=lambda: "test",
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
        current_weight_version=version,
        rust_server_mode=rust,
    )
    for i in range(count):
        req = Req(
            rid=str(i),
            origin_input_text="hi",
            origin_input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(max_new_tokens=8),
            stream=True,
        )
        req.output_ids.extend([3])
        if i in finished:
            req.finished_reason = FINISH_LENGTH(1)
            req.weight_version_events.append(
                WeightVersionEvent(old_version="v0", num_output_tokens=0)
            )
        req.cached_tokens = i * 7
        req.reasoning_tokens = i * 4
        req.retraction_count = i * 2
        if details and i:
            req.cached_tokens_device = 3
            req.cached_tokens_host = 2
            req.cached_tokens_storage = 2
        if rust:
            req.init_incremental_detokenize = Mock(
                side_effect=AssertionError("Rust must not call Python detokenization")
            )
        accumulator.accept(req=req)
    return accumulator.to_payload(dp_rank=rank, is_idle_batch=count == 0)


def encode(output, version="default"):
    sink = Mock()
    with patch(
        "sglang.srt.rust_server.server.get_serving",
        return_value=SimpleNamespace(weight_version=version),
    ):
        RustServer(sink, http_port=0).push_generation(output)
    return sink.push_decode_result_batch.call_args.args


class TestGenerationStatistics(CustomTestCase):
    def test_statistics_match_python_and_preserve_cache_details(self):
        for detailed in (False, True):
            for rank in (None, 0, 1):
                with self.subTest(details=detailed, rank=rank):
                    python = payload(rust=False, details=detailed, rank=rank)
                    rust = payload(rust=True, details=detailed, rank=rank)
                    for name in (
                        "cached_tokens",
                        "cached_tokens_details",
                        "reasoning_tokens",
                        "retraction_counts",
                        "dp_ranks",
                    ):
                        self.assertEqual(getattr(rust, name), getattr(python, name))
                    expected_details = {"device": 7, "host": 0}
                    if detailed:
                        expected_details = {
                            "device": 3,
                            "host": 2,
                            "storage": 2,
                            "storage_backend": "test",
                        }
                    expected = [
                        [0, 7],
                        [None, expected_details],
                        [0, 4],
                        [0, 2],
                        [rank, rank],
                    ]
                    for extras in (False, True):
                        with self.subTest(extras=extras):
                            if extras:
                                rust.output_token_logprobs_val = [[-0.5], []]
                                rust.output_token_logprobs_idx = [[3], []]
                            header, data = encode(rust)
                            columns = msgspec.msgpack.decode(header)
                            self.assertEqual(len(columns), 23)
                            self.assertEqual(columns[16:21], expected)
                            self.assertEqual(columns[21:], ["default", None])
                            self.assertEqual(
                                columns[:4], [["0", "1"], [None, None], [2, 2], [1, 1]]
                            )
                            self.assertEqual(data[0], array("i", [3, 3]).tobytes())
                            self.assertEqual(
                                columns[4:6], [[1, 0], []] if extras else [[], []]
                            )

    def test_idle_batch_has_empty_statistics_columns(self):
        header, data = encode(payload(rust=True, count=0))
        self.assertEqual(
            msgspec.msgpack.decode(header), [[] for _ in range(21)] + ["default", None]
        )
        self.assertEqual(data, [b""])

    def test_incomplete_statistics_are_not_sent(self):
        for name in (
            "cached_tokens",
            "cached_tokens_details",
            "reasoning_tokens",
            "retraction_counts",
            "dp_ranks",
        ):
            for value in (None, [], [0]):
                with self.subTest(column=name, value=value):
                    output = payload(rust=True)
                    setattr(output, name, value)
                    sink = Mock()
                    with self.assertRaisesRegex(ValueError, "one entry per request"):
                        RustServer(sink, http_port=0).push_generation(output)
                    sink.push_decode_result_batch.assert_not_called()


class TestGenerationWeightVersions(CustomTestCase):
    def test_live_version_and_terminal_spans_keep_wire_positions(self):
        for version in ("default", "parity-v1"):
            for finished in ((), (1,), (0, 1)):
                with self.subTest(version=version, finished=finished):
                    output = payload(rust=True, version=version, finished=finished)
                    python = payload(rust=False, version=version, finished=finished)
                    self.assertEqual(output.weight_versions, python.weight_versions)
                    for extras in (False, True):
                        if extras:
                            output.output_token_logprobs_val = [[-0.5], []]
                            output.output_token_logprobs_idx = [[3], []]
                        header, _ = encode(output, version)
                        columns = msgspec.msgpack.decode(header)
                        self.assertEqual(len(columns), 23)
                        self.assertEqual(columns[21], version)
                        self.assertEqual(
                            columns[22],
                            [
                                [["v0", 0, 0], [version, 0, 1]]
                                if i in finished
                                else None
                                for i in range(2)
                            ]
                            if finished
                            else None,
                        )
                        if finished:
                            typed = msgspec.msgpack.decode(
                                msgspec.msgpack.encode(columns[22]),
                                type=list[list[WeightVersionSpan] | None],
                            )
                            self.assertEqual(typed, output.weight_versions)

        # Read serving state at each send, including when the payload is reused.
        output = payload(rust=True)
        for version in ("v1", "v2"):
            self.assertEqual(
                msgspec.msgpack.decode(encode(output, version)[0])[21], version
            )

    def test_partial_span_columns_are_not_sent(self):
        output = payload(rust=True)
        for invalid in ([], [None]):
            output.weight_versions = invalid
            sink = Mock()
            with self.assertRaisesRegex(ValueError, "weight versions"):
                RustServer(sink, http_port=0).push_generation(output)
            sink.push_decode_result_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
