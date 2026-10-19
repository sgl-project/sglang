"""Unit tests for ReqTimeStats.

ReqTimeStatsBase.__setstate__ rebases perf_counter fields onto the receiving
process's clock anchor. Rebasing a field that was never stamped (0.0) turns
the sentinel into a tiny epsilon (sender_diff - receiver_diff), which defeats
== 0.0 / > 0.0 "was this stamped?" checks downstream. Concretely, a PD decode
server never stamps prefill_finished_time locally; if the sentinel arrives at
the tokenizer as an epsilon, first-token bookkeeping mistakes it for a real
stamp and the TTFT / inter-token-latency histograms record ~node-uptime-sized
garbage samples.

The same "was this stamped?" rule governs the gen_ai.latency.* span attrs: the
ones derived from finished_time are only emitted once it is stamped, and
set_finished_time() closes the trace root span, so they have to be derived
inside that call rather than by the caller.
"""

import pickle
import unittest
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.srt.observability.trace import SpanAttributes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class TestSetstatePreservesUnsetTimeSentinels(CustomTestCase):
    def test_two_hop_round_trip(self):
        src = rts.SchedulerReqTimeStats()
        src.enable_metrics = True
        src.wait_queue_entry_time = 123.456
        src.prefill_finished_time = 0.0

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(src)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            hop1 = pickle.loads(blob)
            blob2 = pickle.dumps(hop1)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            hop2 = pickle.loads(blob2)

        self.assertEqual(hop2.prefill_finished_time, 0.0)
        self.assertAlmostEqual(hop2.wait_queue_entry_time, 123.456 - 9.0)


class TestOutputMetaInfo(CustomTestCase):
    def test_first_token_latency_requires_valid_timestamps(self):
        for created, first, expected in (
            (1.0, 1.5, 0.5),
            (0.0, 1.5, None),
            (1.0, 0.0, None),
            (1.0, 1.0, None),
            (1.0, 0.5, None),
        ):
            with self.subTest(created=created, first=first):
                stats = rts.APIServerReqTimeStats()
                stats.created_time = created
                stats.first_token_time = first
                meta_info = stats.convert_to_output_meta_info()
                if expected is None:
                    self.assertNotIn("first_token_latency", meta_info)
                else:
                    self.assertAlmostEqual(meta_info["first_token_latency"], expected)

    def test_tpot_excludes_first_token_from_denominator(self):
        stats = rts.APIServerReqTimeStats()
        stats.created_time = 1.0
        stats.first_token_time = 1.5
        stats.finished_time = 3.5
        # 101 completion tokens, 100 of them in the 2.0s decode interval.
        self.assertAlmostEqual(stats.get_time_per_output_token(101), 0.02)
        self.assertAlmostEqual(stats.get_decode_throughput(101), 50.0)
        meta_info = stats.convert_to_output_meta_info(completion_tokens=101)
        self.assertAlmostEqual(meta_info["decode_throughput"], 50.0)

    def test_tpot_undefined_cases(self):
        for first, finished, completion_tokens in (
            # Single-batch finish: first token stamped in the same batch.
            (3.5, 3.5, 40),
            # First token never stamped.
            (0.0, 3.5, 40),
            # One or zero output tokens: no decode interval to measure.
            (1.5, 3.5, 1),
            (1.5, 3.5, 0),
        ):
            with self.subTest(
                first=first, finished=finished, completion_tokens=completion_tokens
            ):
                stats = rts.APIServerReqTimeStats()
                stats.created_time = 1.0
                stats.first_token_time = first
                stats.finished_time = finished
                self.assertIsNone(stats.get_time_per_output_token(completion_tokens))
                self.assertIsNone(stats.get_decode_throughput(completion_tokens))
                meta_info = stats.convert_to_output_meta_info(
                    completion_tokens=completion_tokens
                )
                self.assertNotIn("decode_throughput", meta_info)


class TestConvertToGenAiSpanAttrs(CustomTestCase):
    def _stats_after_first_token(self) -> rts.APIServerReqTimeStats:
        stats = rts.APIServerReqTimeStats()
        stats.created_time = 1.0
        stats.api_server_dispatch_finish_time = 1.1
        stats.first_token_time = 1.5
        return stats

    def test_prefill_without_finished_time_omits_e2e_and_decode(self):
        attrs = self._stats_after_first_token().convert_to_gen_ai_span_attrs()
        self.assertIn(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL, attrs)
        self.assertIn(SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN, attrs)
        self.assertNotIn(SpanAttributes.GEN_AI_LATENCY_E2E, attrs)
        self.assertNotIn(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE, attrs)
        self.assertNotIn(SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE, attrs)

    def test_finished_time_populates_e2e_and_decode(self):
        stats = self._stats_after_first_token()
        stats.finished_time = 2.0
        attrs = stats.convert_to_gen_ai_span_attrs()
        self.assertAlmostEqual(attrs[SpanAttributes.GEN_AI_LATENCY_E2E], 1.0)
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE], 0.5
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL], 0.4
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE], 0.9
        )


class TestSetFinishedTimeSpanAttrs(CustomTestCase):
    def _tracing_stats(self) -> rts.APIServerReqTimeStats:
        stats = rts.APIServerReqTimeStats()
        stats.trace_ctx = mock.MagicMock()
        stats.trace_ctx.tracing_enable = True
        return stats

    def test_passes_caller_attrs_into_trace_req_finish(self):
        stats = self._tracing_stats()

        stats.set_finished_time(ts=1.25, span_attrs={"gen_ai.request.id": "rid-1"})

        self.assertEqual(stats.finished_time, 1.25)
        stats.trace_ctx.trace_req_finish.assert_called_once_with(
            mock.ANY, attrs={"gen_ai.request.id": "rid-1"}
        )

    def test_merges_latency_attrs_derived_from_finished_time(self):
        stats = self._tracing_stats()
        stats.created_time = 1.0
        stats.api_server_dispatch_finish_time = 1.1
        stats.first_token_time = 1.5

        stats.set_finished_time(ts=2.0, span_attrs={"gen_ai.request.id": "rid-1"})

        attrs = stats.trace_ctx.trace_req_finish.call_args.kwargs["attrs"]
        self.assertEqual(attrs["gen_ai.request.id"], "rid-1")
        self.assertAlmostEqual(attrs[SpanAttributes.GEN_AI_LATENCY_E2E], 1.0)
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE], 0.5
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE], 0.9
        )
        self.assertAlmostEqual(
            attrs[SpanAttributes.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL], 0.4
        )

    def test_caller_attrs_not_mutated(self):
        stats = self._tracing_stats()
        stats.created_time = 1.0
        caller_attrs = {"gen_ai.request.id": "rid-1"}

        stats.set_finished_time(ts=2.0, span_attrs=caller_attrs)

        self.assertEqual(caller_attrs, {"gen_ai.request.id": "rid-1"})


class TestQueueTimeAcrossRetraction(CustomTestCase):
    """A retracted request re-enters the waiting queue after its first forward.

    queue_time must be the total time spent waiting across every entry. It used
    to be first-forward minus the latest queue entry, which went negative.
    """

    T0 = 1000.0

    def _stats(self, disagg_mode=rts.DisaggregationMode.NULL):
        stats = rts.SchedulerReqTimeStats(disagg_mode=disagg_mode)
        stats.enable_metrics = True
        stats.metrics_collector = mock.MagicMock()
        return stats

    def _retract_once(self, stats, forward_calls=(0.10,), readmit_calls=(1.50,)):
        # waits: [0.00, 0.10) before the first forward, [1.00, 1.50) after retraction
        stats.set_wait_queue_entry_time(self.T0 + 0.00)
        for dt in forward_calls:
            stats.set_forward_entry_time(self.T0 + dt)
        stats.set_wait_queue_entry_time(self.T0 + 1.00)
        for dt in readmit_calls:
            stats.set_forward_entry_time(self.T0 + dt)
        stats.set_completion_time(self.T0 + 2.00)

    def test_retracted_request_reports_total_queue_time(self):
        stats = self._stats()
        self._retract_once(stats)

        self.assertAlmostEqual(stats.convert_to_output_meta_info()["queue_time"], 0.6)
        with mock.patch.object(rts, "SGLANG_TEST_REQUEST_TIME_STATS", True):
            self.assertIn("queue_duration=600.00ms", stats.convert_to_duration())

    def test_queue_time_survives_clock_rebase_across_hops(self):
        stats = self._stats()
        self._retract_once(stats)

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(stats)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            blob = pickle.dumps(pickle.loads(blob))
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            tokenizer_side = pickle.loads(blob)

        self.assertAlmostEqual(tokenizer_side.get_queueing_time(), 0.6)

    def test_prefill_chunks_do_not_extend_queue_time(self):
        stats = self._stats()
        self._retract_once(
            stats, forward_calls=(0.10, 0.30, 0.50), readmit_calls=(1.50, 1.70)
        )

        self.assertAlmostEqual(stats.get_queueing_time(), 0.6)

    def test_queue_time_histogram_keeps_one_first_wait_sample(self):
        stats = self._stats()
        self._retract_once(stats)

        stats.metrics_collector.observe_queue_time.assert_called_once()
        (sample,) = stats.metrics_collector.observe_queue_time.call_args.args
        self.assertAlmostEqual(sample, 0.1)

    def test_decode_rebootstrap_counts_both_waits(self):
        stats = self._stats(disagg_mode=rts.DisaggregationMode.DECODE)
        stats.set_wait_queue_entry_time(self.T0 + 0.00)
        stats.set_forward_entry_time(self.T0 + 0.10)
        # held for re-bootstrap: not in the waiting queue until [1.00, 1.25)
        stats.set_retract_time(self.T0 + 0.50)
        stats.set_wait_queue_entry_time(self.T0 + 1.00)
        stats.set_forward_entry_time(self.T0 + 1.25)
        stats.set_quick_finish_time(self.T0 + 1.30)

        self.assertAlmostEqual(stats.get_queueing_time(), 0.35)

    def test_prefill_retry_reset_discards_earlier_attempt_waits(self):
        stats = self._stats(disagg_mode=rts.DisaggregationMode.PREFILL)
        stats.set_wait_queue_entry_time(self.T0 + 0.00)
        stats.set_forward_entry_time(self.T0 + 0.30)
        stats.reset_prefill_retry_time()
        stats.set_wait_queue_entry_time(self.T0 + 1.00)
        stats.set_forward_entry_time(self.T0 + 1.20)

        self.assertAlmostEqual(stats.get_queueing_time(), 0.2)


class TestStorageAdmissionWait(CustomTestCase):
    def _stats(self):
        stats = rts.SchedulerReqTimeStats(
            enable_metrics=True, metrics_collector=mock.MagicMock()
        )
        stats.set_wait_queue_entry_time(100.0)
        return stats

    def test_blocked_polls_accumulate_wait_episodes(self):
        stats = self._stats()
        for observe, output_key in (
            (stats.observe_storage_prefetch_ready, "storage_prefetch_wait"),
            (stats.observe_storage_staging_ready, "storage_staging_wait"),
        ):
            with (
                self.subTest(gate=output_key),
                mock.patch.object(rts.time, "perf_counter") as clock,
            ):
                for ts, ready in (
                    (100.0, True),
                    (100.1, False),
                    (100.15, False),
                    (100.2, False),
                    (100.3, True),
                    (100.4, True),
                    (100.5, False),
                    (100.6, True),
                ):
                    clock.return_value = ts
                    observe(ready)
                    if not ready:
                        self.assertNotIn(
                            output_key, stats.convert_to_output_meta_info()
                        )
                self.assertAlmostEqual(
                    stats.convert_to_output_meta_info()[output_key], 0.3
                )

    def test_zero_missing_and_open_intervals_survive_serialization(self):
        stats = self._stats()
        stats.observe_storage_prefetch_ready(True)
        stats.observe_storage_staging_ready(True)
        zero = pickle.loads(pickle.dumps(stats)).convert_to_output_meta_info()
        self.assertEqual(zero["storage_prefetch_wait"], 0.0)
        self.assertEqual(zero["storage_staging_wait"], 0.0)

        with mock.patch.object(rts.time, "perf_counter", return_value=100.1):
            stats.observe_storage_prefetch_ready(False)
            stats.observe_storage_staging_ready(False)
        incomplete = pickle.loads(pickle.dumps(stats)).convert_to_output_meta_info()
        self.assertNotIn("storage_prefetch_wait", incomplete)
        self.assertNotIn("storage_staging_wait", incomplete)

        legacy = rts.SchedulerReqTimeStats()
        legacy.__setstate__({"has_timing_data": True, "queue_duration_s": 1.0})
        missing = pickle.loads(pickle.dumps(legacy)).convert_to_output_meta_info()
        self.assertNotIn("storage_prefetch_wait", missing)
        self.assertNotIn("storage_staging_wait", missing)

    def test_overlap_retraction_clock_rebase_and_retry(self):
        stats = self._stats()
        with mock.patch.object(
            rts.time, "perf_counter", side_effect=[100.1, 100.2, 100.4, 100.5]
        ):
            stats.observe_storage_staging_ready(False)
            stats.observe_storage_prefetch_ready(False)
            stats.observe_storage_prefetch_ready(True)
            stats.observe_storage_staging_ready(True)
        stats.set_forward_entry_time(100.6)
        stats.set_wait_queue_entry_time(101.0)
        with mock.patch.object(rts.time, "perf_counter", side_effect=[101.1, 101.4]):
            stats.observe_storage_prefetch_ready(False)
            stats.observe_storage_prefetch_ready(True)
        stats.set_forward_entry_time(101.5)

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(stats)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            blob = pickle.dumps(pickle.loads(blob))
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            meta = pickle.loads(blob).convert_to_output_meta_info()
        self.assertAlmostEqual(meta["storage_prefetch_wait"], 0.5)
        self.assertAlmostEqual(meta["storage_staging_wait"], 0.4)
        self.assertAlmostEqual(meta["queue_time"], 1.1)

        with mock.patch.object(rts.time, "perf_counter", return_value=102.0):
            stats.observe_storage_staging_ready(False)
        stats.reset_prefill_retry_time()
        stats.set_wait_queue_entry_time(103.0)
        meta = stats.convert_to_output_meta_info()
        self.assertEqual(meta["storage_prefetch_wait"], 0.0)
        self.assertEqual(meta["storage_staging_wait"], 0.0)

    def test_metrics_disabled_and_decode_do_not_record_local_prefetch(self):
        for stats in (
            rts.SchedulerReqTimeStats(),
            rts.SchedulerReqTimeStats(
                enable_metrics=True,
                metrics_collector=mock.MagicMock(),
                disagg_mode=rts.DisaggregationMode.DECODE,
            ),
        ):
            stats.set_wait_queue_entry_time(100.0)
            stats.observe_storage_prefetch_ready(False)
            stats.observe_storage_prefetch_ready(True)
            stats.observe_storage_staging_ready(False)
            stats.observe_storage_staging_ready(True)
            meta = stats.convert_to_output_meta_info()
            self.assertNotIn("storage_prefetch_wait", meta)
            self.assertNotIn("storage_staging_wait", meta)


if __name__ == "__main__":
    unittest.main()
