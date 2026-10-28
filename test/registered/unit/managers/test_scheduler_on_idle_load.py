"""on_idle's stalled-path load publish is wall-clock bounded.

A non-storage no-batch stall spins on_idle without sleeping, so the gate must
cap the O(queue) get_loads for both the DP-balancing writer and the load
socket. CPU-only: builds a bare Scheduler with mocked collaborators, like
test_scheduler_flush_cache.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestOnIdleStallPublish(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def _stalled_scheduler(self) -> Scheduler:
        s = partial_scheduler()
        s.scheduler_stage_metrics = None
        s.mode.disaggregation_mode = DisaggregationMode.NULL
        s.enable_lmcache = False
        s.parts.output_streamer = MagicMock()
        s.hub.request_registry = MagicMock(
            **{"is_fully_idle.return_value": False}
        )  # stalled, not idle
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.parts.load_publisher = MagicMock()
        s.hub.load_inquirer = MagicMock()
        s.parts.metrics_sink = MagicMock()
        s.parts.dp_attn_adapter = MagicMock()
        s.parts.deferred_responses = MagicMock(**{"has_pending.return_value": False})
        s._last_stall_publish_ts = float("-inf")
        return s

    def test_stall_publish_is_wall_clock_bounded_by_the_floor(self):
        s = self._stalled_scheduler()
        with patch("sglang.srt.managers.scheduler.time.monotonic") as mono:
            # A spinning stall inside the floor publishes once.
            mono.return_value = 100.0
            for _ in range(100):
                s.on_idle()
            self.assertEqual(s.publish_load_snapshot.call_count, 1)
            self.assertEqual(s.parts.load_publisher.publish_load_stat.call_count, 1)
            # After the floor elapses the next stall publishes again.
            mono.return_value = 100.10  # > LOAD_STALL_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)


if __name__ == "__main__":
    unittest.main()
