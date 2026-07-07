"""Unit tests for the idle GC freeze gate."""

import unittest

from sglang.srt.managers.idle_gc import IdleGCFreezeGate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIdleGCFreezeGate(CustomTestCase):
    def test_freezes_after_sustained_idle_following_work(self):
        gate = IdleGCFreezeGate(min_idle_duration_s=5.0, min_freeze_interval_s=60.0)
        self.assertFalse(gate.note(busy=True, now=100.0))
        self.assertFalse(gate.note(busy=False, now=101.0))  # idle 0s
        self.assertFalse(gate.note(busy=False, now=104.0))  # idle 3s
        self.assertTrue(gate.note(busy=False, now=107.0))  # idle 6s
        gate.mark_frozen(107.0)

    def test_never_freezes_without_prior_work(self):
        gate = IdleGCFreezeGate(min_idle_duration_s=5.0, min_freeze_interval_s=0.0)
        self.assertFalse(gate.note(busy=False, now=0.0))
        self.assertFalse(gate.note(busy=False, now=100.0))

    def test_short_idle_gaps_in_live_traffic_never_freeze(self):
        gate = IdleGCFreezeGate(min_idle_duration_s=5.0, min_freeze_interval_s=0.0)
        now = 0.0
        for _ in range(50):
            self.assertFalse(gate.note(busy=True, now=now))
            now += 0.5
            self.assertFalse(gate.note(busy=False, now=now))  # sub-second gap
            now += 0.5

    def test_min_interval_between_freezes(self):
        gate = IdleGCFreezeGate(min_idle_duration_s=1.0, min_freeze_interval_s=60.0)
        gate.note(busy=True, now=0.0)
        gate.note(busy=False, now=1.0)  # idle observed from t=1
        self.assertTrue(gate.note(busy=False, now=2.5))
        gate.mark_frozen(2.5)
        gate.note(busy=True, now=3.0)  # new work
        gate.note(busy=False, now=4.0)
        self.assertFalse(gate.note(busy=False, now=10.0))  # interval not passed
        self.assertTrue(gate.note(busy=False, now=70.0))

    def test_no_refreeze_without_new_work(self):
        gate = IdleGCFreezeGate(min_idle_duration_s=1.0, min_freeze_interval_s=0.0)
        gate.note(busy=True, now=0.0)
        gate.note(busy=False, now=1.0)
        self.assertTrue(gate.note(busy=False, now=2.5))
        gate.mark_frozen(2.5)
        self.assertFalse(gate.note(busy=False, now=100.0))
        self.assertFalse(gate.note(busy=False, now=200.0))


if __name__ == "__main__":
    unittest.main()
