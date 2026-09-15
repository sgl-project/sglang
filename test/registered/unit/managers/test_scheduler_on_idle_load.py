"""on_idle's spinning-path load publishes are wall-clock bounded.

Both a no-batch-but-not-idle stall and a fully idle scheduler spin on_idle
without sleeping, so the gate must cap the O(queue) get_loads for the
DP-balancing writer and the load socket. The busy->idle transition still
publishes immediately, so a router never sees a stale gauge after the engine
drains -- including when the busy phase never reached on_idle, which is why
`process_batch_result` clears the transition flag too. CPU-only: builds a bare
Scheduler with mocked collaborators, like test_scheduler_flush_cache.
"""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class OnIdlePublishTestCase(CustomTestCase):
    def _bare_scheduler(self, *, fully_idle: bool) -> Scheduler:
        s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = MagicMock()
        s.maybe_send_health_check_signal = MagicMock()
        s.is_fully_idle = MagicMock(return_value=fully_idle)
        s.publish_load_snapshot = MagicMock(return_value=None)
        s.load_publisher = MagicMock()
        s.load_inquirer = MagicMock()
        s.metrics_reporter = MagicMock()
        s._last_spin_publish_ts = float("-inf")
        s._was_fully_idle = False
        s.enable_fpm = False
        s.batch_result_processor = MagicMock()
        s._record_step_counters = MagicMock()
        s._maybe_clear_mm_inputs = MagicMock()
        if fully_idle:
            # Collaborators the fully-idle housekeeping walks before publishing.
            s.enable_unified_memory = False
            s.enable_hisparse = False
            s.disaggregation_mode = DisaggregationMode.NULL
            s.invariant_checker = MagicMock()
            s.invariant_checker._check_all_pools.return_value = (False, [])
            s.pool_stats_observer = MagicMock()
            s.token_to_kv_pool_allocator = MagicMock()
            s.token_to_kv_pool_allocator.verify_byte_accounting.return_value = []
            s.kv_events_publisher = MagicMock()
            s.new_token_ratio_tracker = MagicMock()
            s.maybe_sleep_on_idle = MagicMock()
        return s


class TestOnIdleStallPublish(OnIdlePublishTestCase):
    def test_spinning_stall_publishes_once_within_the_floor(self):
        s = self._bare_scheduler(fully_idle=False)
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            for _ in range(100):
                s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 1)
        self.assertEqual(s.load_publisher.publish_load_stat.call_count, 1)

    def test_publishes_again_after_the_floor_elapses(self):
        s = self._bare_scheduler(fully_idle=False)
        with patch("sglang.srt.managers.scheduler.time.monotonic") as mono:
            mono.return_value = 100.0
            s.on_idle()
            mono.return_value = 100.10  # > LOAD_SPIN_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)


class TestOnIdleIdlePublish(OnIdlePublishTestCase):
    def test_spinning_idle_publishes_once_within_the_floor(self):
        s = self._bare_scheduler(fully_idle=True)
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            for _ in range(1000):
                s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 1)
        self.assertEqual(s.load_publisher.publish_load_stat.call_count, 1)

    def test_idle_transition_publishes_immediately(self):
        """The busy->idle transition is never delayed, even when a stall
        publish just happened: the floor only caps the passes after it."""
        s = self._bare_scheduler(fully_idle=True)
        s._last_spin_publish_ts = 100.0
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 1)

    def test_idle_publishes_again_after_the_floor_elapses(self):
        s = self._bare_scheduler(fully_idle=True)
        with patch("sglang.srt.managers.scheduler.time.monotonic") as mono:
            mono.return_value = 100.0
            s.on_idle()
            mono.return_value = 100.10  # > LOAD_SPIN_REFRESH_S
            s.on_idle()
        self.assertEqual(s.publish_load_snapshot.call_count, 2)

    def test_every_return_to_idle_publishes(self):
        s = self._bare_scheduler(fully_idle=True)
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            for _ in range(3):
                s.is_fully_idle.return_value = False  # busy pass
                s.on_idle()
                s.is_fully_idle.return_value = True  # drains again
                s.on_idle()
        # One stall publish (first busy pass, floor empty) + one per transition.
        self.assertEqual(s.publish_load_snapshot.call_count, 4)

    def test_real_batch_makes_the_next_idle_pass_publish(self):
        """A busy phase that never calls on_idle must not let the floor swallow
        the transition publish."""
        s = self._bare_scheduler(fully_idle=True)
        with patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0):
            s.on_idle()  # idle pass: transition publish, flag -> True
            self.assertEqual(s.publish_load_snapshot.call_count, 1)
            s.mark_engine_busy()  # a real batch ran while the loop was busy
            s.on_idle()  # drains again, still inside the 50 ms floor
        self.assertEqual(s.publish_load_snapshot.call_count, 2)


def _batch_mock(*, idle: bool) -> MagicMock:
    forward_mode = MagicMock()
    forward_mode.is_idle.return_value = idle
    forward_mode.is_extend.return_value = False
    forward_mode.is_decode.return_value = False
    forward_mode.is_prebuilt.return_value = False
    batch = MagicMock()
    batch.forward_mode = forward_mode
    batch.reqs = []
    return batch


class TestBatchResultClearsTheTransitionFlag(OnIdlePublishTestCase):
    def test_real_batch_clears_the_flag(self):
        s = self._bare_scheduler(fully_idle=True)
        s._was_fully_idle = True
        s.process_batch_result(_batch_mock(idle=False), MagicMock())
        self.assertFalse(s._was_fully_idle)

    def test_idle_filler_batch_keeps_the_flag(self):
        """dp-attention/overlap filler batches run while the engine has no work
        of its own; clearing the flag there would re-open the unthrottled
        publish on every idle pass."""
        s = self._bare_scheduler(fully_idle=True)
        s._was_fully_idle = True
        s.process_batch_result(_batch_mock(idle=True), MagicMock())
        self.assertTrue(s._was_fully_idle)


if __name__ == "__main__":
    unittest.main()
