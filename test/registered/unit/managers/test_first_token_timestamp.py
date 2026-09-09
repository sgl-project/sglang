import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_output_streamer_customized_info import _FakeReq

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import unwrap_from_pickle
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.observability import req_time_stats
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def accumulator():
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=50,
        get_cached_tokens_details=lambda req: None,
        current_weight_version=None,
    )


class TestFirstTokenTimestamp(unittest.TestCase):
    def request(self, tokens, metrics=True):
        req = _FakeReq("test", tokens)
        req.time_stats = SchedulerReqTimeStats(enable_metrics=metrics)
        return req

    def test_buffered_first_speculative_tokens_record_without_sending(self):
        req = self.request(list(range(8)))
        acc = accumulator()
        with patch(
            "sglang.srt.managers.scheduler_components.output_streamer.time.perf_counter",
            return_value=12,
        ) as clock:
            acc.accept(req=req)
            self.assertEqual(clock.call_count, 1)
        self.assertEqual(req.time_stats.first_token_ready_time, 12)
        self.assertIsNone(acc.to_payload(dp_rank=0, is_idle_batch=False))
        req.output_ids = req.output_ids_through_stop = list(range(50))
        acc = accumulator()
        with patch(
            "sglang.srt.managers.scheduler_components.output_streamer.time.perf_counter"
        ) as clock:
            acc.accept(req=req)
            clock.assert_not_called()
        payload = acc.to_payload(dp_rank=0, is_idle_batch=False)
        stats = unwrap_from_pickle(payload.time_stats)[0]
        self.assertEqual(stats.first_token_ready_time, 12)
        self.assertEqual(payload.output_ids, [list(range(50))])

    def test_streaming_uses_same_readiness_event(self):
        req = self.request([1])
        req.stream = True
        acc = accumulator()
        with patch(
            "sglang.srt.managers.scheduler_components.output_streamer.time.perf_counter",
            return_value=12,
        ):
            acc.accept(req=req)
        self.assertEqual(req.time_stats.first_token_ready_time, 12)
        self.assertIsNotNone(acc.to_payload(dp_rank=0, is_idle_batch=False))

    def test_finished_first_output_is_recorded(self):
        req = self.request([1])
        req._finished = True
        with patch(
            "sglang.srt.managers.scheduler_components.output_streamer.time.perf_counter",
            return_value=12,
        ):
            acc = accumulator()
            acc.accept(req=req)
        self.assertEqual(req.time_stats.first_token_ready_time, 12)
        self.assertIsNotNone(acc.to_payload(dp_rank=0, is_idle_batch=False))

    def test_beam_candidates_do_not_record_deliverable_output(self):
        req = self.request([1])
        req.beam_group = object()
        req.is_beam_leader = True
        with patch(
            "sglang.srt.managers.scheduler_components.output_streamer.time.perf_counter"
        ) as clock:
            accumulator().accept(req=req)
        clock.assert_not_called()
        self.assertEqual(req.time_stats.first_token_ready_time, 0)

    def test_empty_output_and_disabled_metrics_do_not_read_clock(self):
        for req in (self.request([]), self.request([1], metrics=False)):
            with self.subTest(
                tokens=req.output_ids, metrics=req.time_stats.enable_metrics
            ):
                with patch(
                    "sglang.srt.managers.scheduler_components.output_streamer.time.perf_counter"
                ) as clock:
                    accumulator().accept(req=req)
                    clock.assert_not_called()
                self.assertEqual(req.time_stats.first_token_ready_time, 0)

    def test_readiness_survives_existing_metadata_serialization(self):
        stats = SchedulerReqTimeStats(enable_metrics=True, first_token_ready_time=12)
        restored = pickle.loads(pickle.dumps(stats))
        self.assertEqual(restored.first_token_ready_time, 12)
        # The detokenizer forwards the timing payload to the tokenizer.
        forwarded = pickle.loads(pickle.dumps(restored))
        self.assertEqual(forwarded.first_token_ready_time, 12)
        empty = SchedulerReqTimeStats(enable_metrics=True)
        self.assertNotIn("first_token_ready_time", empty.__getstate__())

    def test_existing_clock_conversion_applies_to_readiness(self):
        stats = SchedulerReqTimeStats(enable_metrics=True, first_token_ready_time=12)
        state = stats.__getstate__()
        state["diff_realtime_monotonic"] = (
            req_time_stats.global_diff_realtime_monotonic + 100
        )
        restored = SchedulerReqTimeStats()
        restored.__setstate__(state)
        self.assertAlmostEqual(restored.first_token_ready_time, 112, places=5)

    def test_ttft_uses_readiness_without_changing_arrival_or_itl(self):
        api = APIServerReqTimeStats(
            created_time=10, first_token_time=20, last_time=20, finished_time=30
        )
        ready = SimpleNamespace(first_token_ready_time=12)
        self.assertEqual(api.get_first_token_latency(ready), 2)
        self.assertEqual(api.get_e2e_latency(), 20)
        self.assertEqual(api.first_token_time, 20)
        self.assertEqual(api.last_time, 20)
        self.assertEqual(api.get_decode_latency(), 10)

    def test_missing_or_invalid_readiness_keeps_legacy_behavior(self):
        api = APIServerReqTimeStats(created_time=10, first_token_time=20)
        for stats in (
            None,
            SimpleNamespace(),
            SimpleNamespace(first_token_ready_time=0),
            SimpleNamespace(first_token_ready_time=9),
            SimpleNamespace(first_token_ready_time=21),
            SimpleNamespace(first_token_ready_time=float("nan")),
            SimpleNamespace(first_token_ready_time=float("inf")),
        ):
            with self.subTest(stats=stats):
                self.assertEqual(api.get_first_token_latency(stats), 10)


if __name__ == "__main__":
    unittest.main()
