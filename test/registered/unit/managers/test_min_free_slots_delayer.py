"""Unit tests for managers/min_free_slots_delayer.py — no server, no model loading.

Covers `resolve_min_free_slots` (the threshold-resolution helper, including the
DFlash formula, user-value capping, and the disabled cases) and
`MinFreeSlotsDelayer.should_delay` (the admission-delay predicate).
"""

import unittest

from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_min_free_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResolveMinFreeSlots(CustomTestCase):
    def test_unset_non_dflash_is_disabled(self):
        self.assertIsNone(resolve_min_free_slots(None, 64))

    def test_unset_dflash_falls_back_to_formula(self):
        # formula(64) = min(4, max(2, (64+5)//6)) = 4; max_running >= 8 so enabled.
        self.assertEqual(resolve_min_free_slots(None, 64, is_dflash=True), 4)

    def test_unset_dflash_still_disabled_when_max_running_small(self):
        # DFlash would use the formula, but max_running_requests < 8 disables it.
        self.assertIsNone(resolve_min_free_slots(None, 4, is_dflash=True))

    def test_user_value_leq_one_is_disabled(self):
        for v in (1, 0, -3):
            self.assertIsNone(resolve_min_free_slots(v, 64), f"user_value={v}")

    def test_disabled_when_max_running_below_eight(self):
        self.assertIsNone(resolve_min_free_slots(4, 7))

    def test_enabled_at_max_running_eight_boundary(self):
        # formula(8) = 2; min(2, 2) = 2.
        self.assertEqual(resolve_min_free_slots(2, 8), 2)

    def test_user_value_capped_to_formula(self):
        # formula(8) = 2, so a large user value is capped down to 2.
        self.assertEqual(resolve_min_free_slots(10, 8), 2)
        # formula(100) = 4, so 10 is capped to 4.
        self.assertEqual(resolve_min_free_slots(10, 100), 4)

    def test_user_value_below_formula_is_kept(self):
        # formula(100) = 4; user value 3 is under the cap, so it is kept.
        self.assertEqual(resolve_min_free_slots(3, 100), 3)

    def test_formula_grows_with_max_running(self):
        # formula: 8 -> 2, 16 -> 3, 24 -> 4 (a large user value exposes the formula).
        self.assertEqual(resolve_min_free_slots(99, 8), 2)
        self.assertEqual(resolve_min_free_slots(99, 16), 3)
        self.assertEqual(resolve_min_free_slots(99, 24), 4)

    def test_negative_max_running_is_clamped_then_disabled(self):
        # max_running clamps to 0 (< 8), so the result is disabled.
        self.assertIsNone(resolve_min_free_slots(5, -10))


class TestMinFreeSlotsDelayer(CustomTestCase):
    def setUp(self):
        self.delayer = MinFreeSlotsDelayer(min_free_slots=2)

    def test_delays_when_allocatable_below_threshold(self):
        delayed = self.delayer.should_delay(running_bs=4, num_allocatable_reqs=1)
        self.assertTrue(delayed)

    def test_no_delay_at_threshold(self):
        # num_allocatable_reqs == min_free_slots is not "below", so no delay.
        delayed = self.delayer.should_delay(running_bs=4, num_allocatable_reqs=2)
        self.assertFalse(delayed)

    def test_no_delay_when_plenty_allocatable(self):
        delayed = self.delayer.should_delay(running_bs=4, num_allocatable_reqs=5)
        self.assertFalse(delayed)

    def test_no_delay_when_nothing_running(self):
        # running_bs == 0 short-circuits to no delay regardless of allocatable count.
        delayed = self.delayer.should_delay(running_bs=0, num_allocatable_reqs=0)
        self.assertFalse(delayed)


if __name__ == "__main__":
    unittest.main()
