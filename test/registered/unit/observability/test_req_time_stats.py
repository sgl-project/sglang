"""Unit tests for ReqTimeStats IPC serialization.

ReqTimeStatsBase.__setstate__ rebases perf_counter fields onto the receiving
process's clock anchor. Rebasing a field that was never stamped (0.0) turns
the sentinel into a tiny epsilon (sender_diff - receiver_diff), which defeats
== 0.0 / > 0.0 "was this stamped?" checks downstream. Concretely, a PD decode
server never stamps prefill_finished_time locally; if the sentinel arrives at
the tokenizer as an epsilon, first-token bookkeeping mistakes it for a real
stamp and the TTFT / inter-token-latency histograms record ~node-uptime-sized
garbage samples.
"""

import pickle
import unittest
from unittest import mock

import sglang.srt.observability.req_time_stats as rts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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

    def test_decode_preallocation_trace_is_split(self):
        stats = rts.SchedulerReqTimeStats()
        stats.decode_prealloc_queue_entry_time = 10.0
        stats.bootstrap_done_time = 12.0

        with mock.patch.object(stats, "trace_slice") as trace_slice:
            stats.set_decode_transfer_queue_entry_time(ts=15.0)

        slices = {
            call.args[0].stage_name: (call.args[1], call.args[2])
            for call in trace_slice.call_args_list
        }
        self.assertEqual(slices["decode_bootstrap_handshake"], (10.0, 12.0))
        self.assertEqual(slices["decode_kv_allocation_wait"], (12.0, 15.0))

    def test_decode_preallocation_without_bootstrap_timestamp(self):
        stats = rts.SchedulerReqTimeStats()
        stats.decode_prealloc_queue_entry_time = 10.0

        with mock.patch.object(stats, "trace_slice") as trace_slice:
            stats.set_decode_transfer_queue_entry_time(ts=15.0)

        slices = [call.args[0].stage_name for call in trace_slice.call_args_list]
        self.assertEqual(slices, ["decode_kv_allocation_wait"])

    def test_decode_preallocation_metrics_observed_once(self):
        stats = rts.SchedulerReqTimeStats()
        stats.enable_metrics = True
        stats.decode_prealloc_queue_entry_time = 10.0
        stats.bootstrap_done_time = 12.0
        stats.metrics_collector = mock.MagicMock()

        stats.set_decode_transfer_queue_entry_time(ts=15.0)

        stats.metrics_collector.observe_kv_transfer_bootstrap.assert_called_once_with(
            bootstrap_ms=2000.0,
            alloc_ms=3000.0,
        )

    def test_decode_preallocation_with_tracing_disabled(self):
        stats = rts.SchedulerReqTimeStats()
        stats.decode_prealloc_queue_entry_time = 10.0
        stats.bootstrap_done_time = 12.0
        stats.trace_ctx = mock.MagicMock(tracing_enable=False)

        stats.set_decode_transfer_queue_entry_time(ts=15.0)

        stats.trace_ctx.trace_slice.assert_not_called()


if __name__ == "__main__":
    unittest.main()
