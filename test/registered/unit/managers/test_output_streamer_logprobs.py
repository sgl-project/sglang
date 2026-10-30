import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import RequestOutputStats
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.output_streamer import (
    OutputSink,
    _GenerationStreamAccumulator,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REQUEST_STATS = RequestOutputStats(
    prompt_tokens=0,
    cached_tokens=0,
    cached_tokens_details=None,
    retraction_count=0,
    image_tokens=0,
    audio_tokens=0,
    video_tokens=0,
    time_stats=None,
)


class _FakeReq:
    def __init__(self, *, is_retracted: bool, max_new_tokens: int):
        self.rid = "req"
        self.http_worker_ipc = None
        self.finished_reason = None
        self.finished_output = False
        self.finished_len = None
        self.stream = False
        self.sampling_params = SimpleNamespace(
            max_new_tokens=max_new_tokens,
            stream_interval=None,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        self.output_ids = []
        self.output_ids_through_stop = []
        self.send_token_offset = 0
        self.send_output_token_logprobs_offset = 0
        self.detokenize_header_sent = False
        self.origin_input_ids = []
        self.origin_input_ids_unpadded = array("q")
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.retraction_count = 0
        self.request_stats_retraction_count = None
        self.time_stats = SimpleNamespace(
            prefill_finished_time=0.0, enable_metrics=False
        )
        self.weight_version_events = []
        self.mm_image_tokens = 0
        self.mm_audio_tokens = 0
        self.mm_video_tokens = 0
        self.multimodal_inputs = None
        self.customized_info = None
        self.is_retracted = is_retracted
        self.beam_group = None

        self.return_logprob = True
        self.input_logprob_sent = True
        self.logprob = SimpleNamespace(
            output_token_logprobs_val=[-0.5],
            output_token_logprobs_idx=[42],
            output_top_logprobs_val=[[(-0.5, 42)]],
            output_top_logprobs_idx=[[42]],
            output_token_ids_logprobs_val=[[-0.5]],
            output_token_ids_logprobs_idx=[[42]],
        )

    def finished(self):
        return False

    take_request_stats_due = Req.take_request_stats_due
    take_detokenize_header_due = Req.take_detokenize_header_due

    def get_mm_token_counts(self):
        return self.mm_image_tokens, self.mm_audio_tokens, self.mm_video_tokens


def _make_accumulator(
    *, output_sink: OutputSink = OutputSink.DETOKENIZER
) -> _GenerationStreamAccumulator:
    return _GenerationStreamAccumulator(
        return_logprob=True,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=1,
        build_request_stats=lambda req: _REQUEST_STATS,
        output_sink=output_sink,
        is_request_level_customized_info_key=lambda key: False,
        current_weight_version=None,
    )


class TestOutputStreamerLogprobs(unittest.TestCase):
    def test_retracted_empty_output_does_not_advance_logprob_offset(self):
        req = _FakeReq(is_retracted=True, max_new_tokens=16)
        accumulator = _make_accumulator()

        accumulator.accept(req=req)

        self.assertEqual(req.send_token_offset, 0)
        self.assertEqual(req.send_output_token_logprobs_offset, 0)
        self.assertEqual(accumulator.output_token_logprobs_val, [[]])

    def test_rust_server_mode_reports_cached_tokens_per_request(self):
        """The Rust frontend rejects a batch whose cached_tokens column is shorter than rids."""
        req = _FakeReq(is_retracted=False, max_new_tokens=16)
        req.cached_tokens = 7
        accumulator = _make_accumulator(output_sink=OutputSink.RUST)

        accumulator.accept(req=req)

        self.assertEqual(accumulator.rids, ["req"])
        self.assertEqual(accumulator.cached_tokens, [7])


class TestOutputStreamerReasoningUsage(unittest.TestCase):
    def test_reasoning_usage_matches_retained_prefix(self):
        # 99 closes reasoning; tokens after it are ordinary content.
        cases = (
            ([10, 11, 12, 13], 2, 2),  # Stop inside reasoning.
            ([10, 99, 20, 21], 1, 1),  # Stop before the reasoning end tag.
            ([10, 99, 20, 21], 2, 2),  # Stop at the end tag.
            ([10, 99, 20, 21], 3, 2),  # Stop inside content.
            ([10, 99, 20, 21], 4, 2),  # No discarded tokens.
        )
        for tokens, retained, expected in cases:
            for stream in (False, True):
                for rust in (False, True):
                    with self.subTest(
                        tokens=tokens, retained=retained, stream=stream, rust=rust
                    ):
                        req = _FakeReq(is_retracted=False, max_new_tokens=16)
                        req.output_ids = tokens
                        req._is_reasoning_over = False
                        req._think_end_matcher = None
                        req._think_end_match_len = 0
                        Req.update_reasoning_tokens(req, tokens, [99])
                        original_count = req.reasoning_tokens
                        req.finished_len = retained
                        req.output_ids_through_stop = Req.output_ids_through_stop.fget(
                            req
                        )
                        req.finished = lambda: True
                        req.finished_reason = SimpleNamespace(
                            to_json=lambda: {"type": "stop"}
                        )
                        req.stream = stream
                        # Streaming usage is cumulative, even after a prior emission.
                        req.send_token_offset = 1 if stream else 0
                        accumulator = _make_accumulator(
                            output_sink=(
                                OutputSink.RUST if rust else OutputSink.DETOKENIZER
                            )
                        )
                        accumulator.accept(req=req)
                        self.assertEqual(accumulator.reasoning_tokens, [expected])
                        self.assertEqual(req.reasoning_tokens, original_count)
                        self.assertEqual(
                            accumulator.output_ids_flat,
                            array("q", tokens[1 if stream else 0 : retained]),
                        )
                        if not rust:
                            self.assertEqual(accumulator.completion_tokens, [retained])


if __name__ == "__main__":
    unittest.main()
