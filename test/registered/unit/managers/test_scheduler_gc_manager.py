"""Unit tests for srt/managers/scheduler_gc_manager.py.

SchedulerGCManager keeps stop-the-world GC pauses off the scheduler's batch
critical path: it defers automatic gen-2 collections, freezes the startup
heap once on the first idle tick after warmup, and thereafter runs full
collections only while the scheduler is idle (throttled).

Note: ``gc.freeze()`` moves currently tracked objects into the permanent
generation and cannot be undone. That is harmless for test correctness (the
permanent generation is simply skipped by traversal), but tests below assert
on freeze-count deltas rather than absolute values.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import gc
import unittest

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_gc_manager import SchedulerGCManager


class TestSchedulerGCManager(unittest.TestCase):
    def setUp(self):
        self._saved_threshold = gc.get_threshold()
        # SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT defaults to off; enable it
        # explicitly so these tests exercise the managed behavior.
        self._enable_ctx = envs.SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT.override(True)
        self._enable_ctx.__enter__()

    def tearDown(self):
        self._enable_ctx.__exit__(None, None, None)
        gc.set_threshold(*self._saved_threshold)

    def test_disabled_is_a_strict_noop(self):
        before = gc.get_threshold()
        with envs.SGLANG_ENABLE_SCHEDULER_GC_MANAGEMENT.override(False):
            manager = SchedulerGCManager()
            self.assertFalse(manager.enabled)
            self.assertEqual(gc.get_threshold(), before)

            frozen_before = gc.get_freeze_count()
            manager.on_idle()
            self.assertEqual(gc.get_freeze_count(), frozen_before)
            self.assertFalse(manager._did_initial_freeze)

    def test_gen2_threshold_raised_but_never_lowered(self):
        with envs.SGLANG_SCHEDULER_GC_GEN2_THRESHOLD.override(10000):
            gc.set_threshold(700, 10, 10)
            SchedulerGCManager()
            self.assertEqual(gc.get_threshold(), (700, 10, 10000))

            # An operator-configured larger threshold must not be lowered.
            gc.set_threshold(700, 10, 50000)
            SchedulerGCManager()
            self.assertEqual(gc.get_threshold(), (700, 10, 50000))

    def test_gen0_gen1_thresholds_untouched(self):
        gc.set_threshold(123, 45, 10)
        SchedulerGCManager()
        gen0, gen1, _ = gc.get_threshold()
        self.assertEqual((gen0, gen1), (123, 45))

    def test_first_idle_tick_collects_and_freezes_once(self):
        manager = SchedulerGCManager()
        self.assertFalse(manager._did_initial_freeze)

        frozen_before = gc.get_freeze_count()
        manager.on_idle()
        self.assertTrue(manager._did_initial_freeze)
        self.assertGreater(
            gc.get_freeze_count(),
            frozen_before,
            "first idle tick must freeze the startup heap",
        )

        # Subsequent idle ticks must not freeze again (only collect, and
        # only when the throttle interval has elapsed).
        frozen_after_first = gc.get_freeze_count()
        manager._last_idle_gc_time = 0.0  # force the throttle open
        manager.on_idle()
        self.assertEqual(
            gc.get_freeze_count(),
            frozen_after_first,
            "periodic idle GC must not re-freeze (frozen-then-garbage "
            "objects would leak permanently)",
        )

    def test_idle_gc_is_throttled_by_interval(self):
        with envs.SGLANG_SCHEDULER_IDLE_GC_INTERVAL.override(3600.0):
            manager = SchedulerGCManager()
            manager.on_idle()  # initial freeze
            stamp = manager._last_idle_gc_time

            manager.on_idle()  # immediately again: throttled, must no-op
            self.assertEqual(manager._last_idle_gc_time, stamp)

            manager._last_idle_gc_time -= 7200.0  # pretend interval elapsed
            manager.on_idle()
            self.assertGreater(manager._last_idle_gc_time, stamp - 7200.0)

    def test_idle_gc_reclaims_reference_cycles(self):
        manager = SchedulerGCManager()
        manager.on_idle()  # initial freeze (cycles created after this)

        class Node:
            pass

        a, b = Node(), Node()
        a.other, b.other = b, a  # unreachable cycle once dropped
        del a, b

        manager._last_idle_gc_time = 0.0  # open the throttle
        gc.set_threshold(*self._saved_threshold)  # restore for isolation
        before = gc.get_count()[0]
        manager.on_idle()
        # The cycle objects were created after the freeze, so the idle
        # collect must be able to reclaim them (no exception, gen0 drained).
        self.assertLessEqual(gc.get_count()[0], before)


if __name__ == "__main__":
    unittest.main()
