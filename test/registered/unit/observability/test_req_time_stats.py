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

    def test_opt_in_scheduler_timestamps_survive_round_trip(self):
        src = rts.SchedulerReqTimeStats(has_timing_data=True)
        src.dpc_dispatch_time = 99.0
        src.scheduler_recv_time = 100.0
        src.wait_queue_entry_time = 101.0
        src.forward_entry_time = 108.0

        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            blob = pickle.dumps(src)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            dst = pickle.loads(blob)

        self.assertAlmostEqual(dst.dpc_dispatch_time, 94.0)
        self.assertAlmostEqual(dst.scheduler_recv_time, 95.0)
        self.assertAlmostEqual(dst.wait_queue_entry_time, 96.0)
        self.assertAlmostEqual(dst.forward_entry_time, 103.0)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0):
            meta_info = dst.convert_to_output_meta_info()
            api = rts.APIServerReqTimeStats()
            merged = api.convert_to_output_meta_info(dst)
        self.assertAlmostEqual(meta_info["dpc_dispatch_time"], 1_000_099.0)
        self.assertAlmostEqual(meta_info["scheduler_recv_time"], 1_000_100.0)
        self.assertAlmostEqual(meta_info["wait_queue_entry_time"], 1_000_101.0)
        self.assertAlmostEqual(meta_info["forward_entry_time"], 1_000_108.0)
        self.assertAlmostEqual(merged["dpc_dispatch_time"], 1_000_099.0)
        self.assertAlmostEqual(merged["scheduler_recv_time"], 1_000_100.0)
        self.assertAlmostEqual(merged["wait_queue_entry_time"], 1_000_101.0)
        self.assertAlmostEqual(merged["forward_entry_time"], 1_000_108.0)

    def test_dpc_timestamp_survives_two_ipc_hops(self):
        src = rts.DPControllerReqTimeStats(
            dpc_dispatch_time=100.0, has_timing_data=True
        )
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_000.0):
            scheduler_blob = pickle.dumps(src)
        with (
            mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_005.0),
            mock.patch.object(rts, "calibrate_time_diff"),
        ):
            dpc_hop = pickle.loads(scheduler_blob)
            scheduler = rts.SchedulerReqTimeStats.new_from_obj(dpc_hop)
            scheduler.has_timing_data = True
            tokenizer_blob = pickle.dumps(scheduler)
        with mock.patch.object(rts, "global_diff_realtime_monotonic", 1_000_009.0):
            tokenizer_hop = pickle.loads(tokenizer_blob)
            meta_info = tokenizer_hop.convert_to_output_meta_info()

        self.assertAlmostEqual(tokenizer_hop.dpc_dispatch_time, 91.0)
        self.assertAlmostEqual(meta_info["dpc_dispatch_time"], 1_000_100.0)

    def test_dpc_timestamp_is_omitted_by_default(self):
        src = rts.DPControllerReqTimeStats(dpc_dispatch_time=100.0)
        state = src.__getstate__()
        self.assertNotIn("dpc_dispatch_time", state)
        self.assertNotIn("has_timing_data", state)


if __name__ == "__main__":
    unittest.main()
