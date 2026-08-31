"""on_idle's stalled-path load publish is wall-clock bounded.

A no-batch-but-not-idle stall spins on_idle without sleeping, so the gate must
cap the O(queue) get_loads for both the DP-balancing writer and the load
socket. CPU-only: builds a bare Scheduler with mocked collaborators, like
test_scheduler_flush_cache.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestOnIdleStallPublish(CustomTestCase):
    def _stalled_scheduler(self) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=False)  # stalled, not idle
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
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


if __name__ == "__main__":
    unittest.main()
