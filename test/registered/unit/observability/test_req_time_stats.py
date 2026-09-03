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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
