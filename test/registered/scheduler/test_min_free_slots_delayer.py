import unittest

from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_min_free_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestResolveMinFreeSlots(unittest.TestCase):
    """Unit tests for resolve_min_free_slots threshold resolution."""

    def test_unset_non_dflash_disables(self):
        # Unset + not DFlash -> trigger stays disabled.
        self.assertIsNone(resolve_min_free_slots(None, 512, is_dflash=False))

    def test_unset_dflash_auto_enables(self):
        # Unset + DFlash -> falls back to the legacy formula (full mapping).
        self.assertEqual(resolve_min_free_slots(None, 512, is_dflash=True), 4)
        self.assertEqual(resolve_min_free_slots(None, 8, is_dflash=True), 2)

    def test_unset_dflash_small_cluster_disables(self):
        # DFlash auto-default still respects the < 8 guard.
        self.assertIsNone(resolve_min_free_slots(None, 7, is_dflash=True))
        self.assertIsNone(resolve_min_free_slots(None, 0, is_dflash=True))

    def test_le_one_disables(self):
        # <= 1 can never batch, so it is a no-op.
        self.assertIsNone(resolve_min_free_slots(1, 512))
        self.assertIsNone(resolve_min_free_slots(0, 512))

    def test_small_cluster_disables(self):
        # max_running_requests < 8 disables, matching DFlash.
        self.assertIsNone(resolve_min_free_slots(4, 7))

    def test_caps_to_formula(self):
        # Capped down so it never delays more aggressively than DFlash.
        self.assertEqual(resolve_min_free_slots(10, 512), 4)
        self.assertEqual(resolve_min_free_slots(10, 8), 2)  # (8 + 5) // 6 = 2

    def test_respects_smaller_user_value(self):
        # Below the formula cap is taken as-is.
        self.assertEqual(resolve_min_free_slots(3, 512), 3)
        self.assertEqual(resolve_min_free_slots(2, 8), 2)

    def test_user_value_overrides_dflash_default(self):
        # An explicit user value wins over the DFlash auto-default.
        self.assertEqual(resolve_min_free_slots(3, 512, is_dflash=True), 3)


class TestMinFreeSlotsDelayer(unittest.TestCase):
    """Unit tests for the per-rank local should_delay decision."""

    def test_delays_below_threshold(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        self.assertTrue(delayer.should_delay(running_bs=100, num_allocatable_reqs=2))

    def test_no_delay_at_or_above_threshold(self):
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        self.assertFalse(delayer.should_delay(running_bs=100, num_allocatable_reqs=4))
        self.assertFalse(delayer.should_delay(running_bs=100, num_allocatable_reqs=8))

    def test_no_delay_when_idle(self):
        # Nothing running: no decode batch to protect, prefill at once.
        delayer = MinFreeSlotsDelayer(min_free_slots=4)
        self.assertFalse(delayer.should_delay(running_bs=0, num_allocatable_reqs=0))


if __name__ == "__main__":
    unittest.main()
