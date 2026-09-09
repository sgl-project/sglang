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

    def test_direct_linker_stage_durations_are_logged_and_serialized(self):
        stats = rts.SchedulerReqTimeStats()
        stats.created_time = 0.2
        stats.tokenize_finish_time = 0.8
        stats.wait_queue_entry_time = 1.0
        stats.forward_entry_time = 2.0
        stats.prefill_finished_time = 5.0
        stats.completion_time = 6.0
        stats.set_direct_load_prepare_start_time(1.2)
        stats.set_direct_load_prepare_finish_time(1.5)
        stats.set_direct_load_start_time(2.1)
        stats.set_direct_load_finish_time(4.6)
        stats.add_direct_lookup_duration(0.04)
        stats.add_direct_lookup_duration(0.06)

        duration = stats.convert_to_duration()
        self.assertIn("token_preprocess_duration=600.00ms", duration)
        self.assertIn("request_dispatch_duration=200.00ms", duration)
        self.assertIn("queue_duration=1000.00ms", duration)
        self.assertIn("mooncake_lookup_duration=100.00ms", duration)
        self.assertIn("mooncake_lookup_count=2", duration)
        self.assertIn("mooncake_lookup_avg_duration=50.00ms", duration)
        self.assertIn("direct_load_prepare_duration=300.00ms", duration)
        self.assertIn("mooncake_l3_to_l1_duration=2500.00ms", duration)
        self.assertIn("prefill_inference_duration=3000.00ms", duration)
        self.assertIn("forward_duration=4000.00ms", duration)

        restored = pickle.loads(pickle.dumps(stats))
        self.assertAlmostEqual(restored.direct_load_prepare_start_time, 1.2)
        self.assertAlmostEqual(restored.direct_load_finish_time, 4.6)
        self.assertAlmostEqual(restored.direct_lookup_duration, 0.1)
        self.assertEqual(restored.direct_lookup_count, 2)

    def test_api_token_preprocess_timestamps_survive_scheduler_hops(self):
        api_stats = rts.APIServerReqTimeStats()
        api_stats.created_time = 10.0
        api_stats.tokenize_finish_time = 10.25

        api_hop = pickle.loads(pickle.dumps(api_stats))
        dp_stats = rts.DPControllerReqTimeStats.new_from_obj(api_hop)
        dp_hop = pickle.loads(pickle.dumps(dp_stats))
        scheduler_stats = rts.SchedulerReqTimeStats.new_from_obj(dp_hop)

        self.assertAlmostEqual(scheduler_stats.created_time, 10.0)
        self.assertAlmostEqual(scheduler_stats.tokenize_finish_time, 10.25)

    def test_queue_wait_breakdown_attributes_blockers(self):
        stats = rts.SchedulerReqTimeStats()
        stats.set_wait_queue_entry_time(1.0)
        stats.set_queue_wait_reason("running_batch_full", 2.0)
        stats.set_queue_wait_reason("running_batch_full", 3.0)
        stats.set_queue_wait_reason("token_or_kv_capacity", 4.5)
        stats.set_forward_entry_time(6.0)
        stats.completion_time = 7.0

        self.assertEqual(
            stats.queue_reason_durations,
            {
                "awaiting_scheduler_check": 1.0,
                "running_batch_full": 2.5,
                "token_or_kv_capacity": 1.5,
            },
        )
        self.assertEqual(stats.queue_reason_checks["running_batch_full"], 2)
        duration = stats.convert_to_duration()
        self.assertIn("queue_duration=5000.00ms", duration)
        self.assertIn("running_batch_full=2500.00ms/checks=2", duration)
        self.assertIn("token_or_kv_capacity=1500.00ms/checks=1", duration)


if __name__ == "__main__":
    unittest.main()
