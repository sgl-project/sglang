"""When the scheduler publishes its load snapshot.

on_idle's stalled-path publish is wall-clock bounded: a non-storage no-batch
stall spins on_idle without sleeping, so the gate must cap the O(queue)
get_loads for both the DP-balancing writer and the load socket. Ingesting new
requests publishes once they are queued. CPU-only: builds a bare Scheduler with
mocked collaborators, like test_scheduler_flush_cache.
"""

import unittest
from unittest.mock import MagicMock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestOnIdleStallPublish(CustomTestCase):
    def _stalled_scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.enable_hicache_storage = False
        s.disaggregation_mode = DisaggregationMode.NULL
        s.enable_lmcache = False
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=False)  # stalled, not idle
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
        s.metrics_reporter = MagicMock()
        s.dp_attn_adapter = MagicMock()
        s._last_stall_publish_ts = float("-inf")
        return s

    def test_spinning_stall_publishes_once_within_the_floor(self):
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            for _ in range(100):
                s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 1)
        self.assertEqual(s.load_publisher.publish_load_stat.call_count, 1)

    def test_publishes_again_after_the_floor_elapses(self):
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic") as mono:
            mono.return_value = 100.0
            s.on_idle()
            mono.return_value = 100.10  # > LOAD_STALL_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)


class TestIngestPublish(CustomTestCase):
    def test_a_snapshot_follows_newly_queued_requests(self):
        """A rank publishes its load after each forward, before it reads the
        requests sent meanwhile; without a publish once they are queued, DP
        balancing resets its estimate to that stale load and sends it more."""
        s = Scheduler.__new__(Scheduler)
        s._deferred_input_requests = []
        s._poll_timeout_aborts = MagicMock(return_value=[])
        s.request_receiver = MagicMock()
        s.request_receiver.recv_requests.return_value = ["req"]
        s.metrics_reporter = MagicMock()
        steps = MagicMock()
        s.process_input_requests = steps.queue
        s.publish_load_snapshot = steps.publish
        with published_topology():
            s.ingest_requests()
        self.assertEqual(
            steps.mock_calls, [call.queue(["req"]), call.publish(force=True)]
        )


if __name__ == "__main__":
    unittest.main()
