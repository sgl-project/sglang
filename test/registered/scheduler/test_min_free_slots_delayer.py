import unittest

from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_min_free_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestResolveMinFreeSlots(unittest.TestCase):
    def test_unset_non_dflash_disables(self):
        self.assertIsNone(resolve_min_free_slots(None, 512, is_dflash_family=False))

    def test_unset_dflash_auto_enables(self):
        self.assertEqual(resolve_min_free_slots(None, 512, is_dflash_family=True), 4)
        self.assertEqual(resolve_min_free_slots(None, 8, is_dflash_family=True), 2)

    def test_unset_dflash_small_cluster_disables(self):
        self.assertIsNone(resolve_min_free_slots(None, 7, is_dflash_family=True))
        self.assertIsNone(resolve_min_free_slots(None, 0, is_dflash_family=True))

    def test_le_one_disables(self):
        # <= 1 can never batch, so it is a no-op.
        self.assertIsNone(resolve_min_free_slots(1, 512))
        self.assertIsNone(resolve_min_free_slots(0, 512))

    def test_explicit_value_survives_small_cluster(self):
        # The < 8 guard belongs to the DFlash auto-default, not explicit values.
        self.assertEqual(resolve_min_free_slots(4, 7), 4)
        self.assertEqual(resolve_min_free_slots(4, 7, is_dflash_family=True), 4)

    def test_non_dflash_uses_explicit_value(self):
        self.assertEqual(resolve_min_free_slots(2, 8), 2)
        self.assertEqual(resolve_min_free_slots(3, 512), 3)
        self.assertEqual(resolve_min_free_slots(8, 512), 8)
        self.assertEqual(resolve_min_free_slots(16, 512), 16)

    def test_explicit_value_is_capped_to_max_running_requests(self):
        self.assertEqual(resolve_min_free_slots(16, 8), 8)

    def test_user_value_overrides_dflash_default(self):
        self.assertEqual(resolve_min_free_slots(3, 512, is_dflash_family=True), 3)
        self.assertEqual(resolve_min_free_slots(16, 512, is_dflash_family=True), 16)

    def test_explicit_one_disables_dflash_default(self):
        self.assertIsNone(resolve_min_free_slots(1, 512, is_dflash_family=True))


class TestMinFreeSlotsDelayer(unittest.TestCase):
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
