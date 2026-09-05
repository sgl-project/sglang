"""Unit tests for ReqTimeStats IPC serialization and duration formatting.

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
import re
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


class TestPrefillBootstrapSubPhases(CustomTestCase):
    """The prefill log line splits bootstrap_queue into rendezvous + alloc wait."""

    @staticmethod
    def _prefill_stats(**kwargs):
        stats = rts.SchedulerReqTimeStats(disagg_mode=rts.DisaggregationMode.PREFILL)
        for name, value in kwargs.items():
            setattr(stats, name, value)
        return stats

    @staticmethod
    def _field(line, name):
        match = re.search(rf"\b{name}=([0-9.]+)ms", line)
        return None if match is None else float(match.group(1))

    def test_sub_phases_reconstruct_bootstrap_queue_duration(self):
        stats = self._prefill_stats(
            prefill_bootstrap_queue_entry_time=100.0,
            bootstrap_done_time=100.3,
            wait_queue_entry_time=100.5,
            forward_entry_time=100.6,
            completion_time=100.9,
        )

        with mock.patch.object(rts, "SGLANG_TEST_REQUEST_TIME_STATS", True):
            line = stats.convert_to_duration()

        bootstrap = self._field(line, "bootstrap_duration")
        alloc_wait = self._field(line, "alloc_wait_duration")
        self.assertIsNotNone(bootstrap)
        self.assertIsNotNone(alloc_wait)
        self.assertAlmostEqual(bootstrap + alloc_wait, 500.0, places=2)

    def test_out_of_order_stamps_report_the_stamp_not_a_duration(self):
        # Optimistic prefill: the request leaves the bootstrap queue before the
        # rendezvous, so wait_queue_entry_time precedes bootstrap_done_time.
        stats = self._prefill_stats(
            prefill_bootstrap_queue_entry_time=100.0,
            wait_queue_entry_time=100.2,
            bootstrap_done_time=100.4,
            forward_entry_time=100.6,
            completion_time=100.9,
        )

        with mock.patch.object(rts, "SGLANG_TEST_REQUEST_TIME_STATS", True):
            line = stats.convert_to_duration()

        self.assertIn("bootstrap_duration=", line)
        self.assertNotIn("alloc_wait_duration=", line)
        self.assertIn(
            f"bootstrap_done_time={stats.format_wallclock(stats.bootstrap_done_time)}",
            line,
        )

    def test_unstamped_bootstrap_done_falls_back_to_queue_duration(self):
        # A request that fails bootstrap never observes KVPoll.WaitingForInput.
        stats = self._prefill_stats(
            prefill_bootstrap_queue_entry_time=100.0,
            wait_queue_entry_time=100.5,
            forward_entry_time=100.6,
            completion_time=100.9,
        )

        with mock.patch.object(rts, "SGLANG_TEST_REQUEST_TIME_STATS", True):
            line = stats.convert_to_duration()

        self.assertAlmostEqual(
            self._field(line, "bootstrap_queue_duration"), 500.0, places=2
        )
        self.assertNotIn("alloc_wait_duration=", line)

    def test_bootstrap_done_time_is_first_write_wins(self):
        stats = self._prefill_stats(prefill_bootstrap_queue_entry_time=100.0)

        stats.set_bootstrap_done_time(200.0)
        stats.set_bootstrap_done_time(300.0)

        self.assertEqual(stats.bootstrap_done_time, 200.0)


if __name__ == "__main__":
    unittest.main()
