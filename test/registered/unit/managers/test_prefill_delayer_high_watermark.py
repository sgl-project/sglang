import unittest

from sglang.srt.managers.prefill_delayer import (
    RecentPrefillBatchSizeTracker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillDelayerHighWatermark(CustomTestCase):
    def test_peak_expires_after_recent_admission_window(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        self.assertEqual(tracker.observe_admission(100), 100)
        for admitted_prefill_bs in [2, 1, 2]:
            self.assertEqual(tracker.observe_admission(admitted_prefill_bs), 100)

        self.assertEqual(tracker.observe_admission(2), 2)

    def test_recurring_small_peak_remains_effective(self):
        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        for admitted_prefill_bs in [2, 1, 1, 2, 1, 1, 2]:
            self.assertEqual(tracker.observe_admission(admitted_prefill_bs), 2)

    def test_rejects_non_admission_updates(self):
        with self.assertRaisesRegex(ValueError, "window_size must be positive"):
            RecentPrefillBatchSizeTracker(window_size=0)

        tracker = RecentPrefillBatchSizeTracker(window_size=4)

        with self.assertRaisesRegex(ValueError, "real admission"):
            tracker.observe_admission(0)

        self.assertEqual(tracker.max_prefill_bs, 0)


if __name__ == "__main__":
    unittest.main()
