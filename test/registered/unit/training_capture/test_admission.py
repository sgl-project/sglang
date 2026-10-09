"""Pressure, cooldown and recovery boundaries for adaptive capture admission."""

import unittest

from sglang.srt.training_capture.admission import CaptureAdmission
from sglang.srt.training_capture.config import AdaptiveCaptureConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCaptureAdmission(CustomTestCase):
    def test_disabled_preserves_configured_sampling_and_zero_stays_zero(self):
        for config in (None, AdaptiveCaptureConfig()):
            with self.subTest(config=config):
                controller = CaptureAdmission(0.0, config)
                controller.failure(1.0, "writer_error")
                for now in (2.0, 10.0, 100.0):
                    self.assertEqual(
                        controller.observe(now, occupancy=0.0, writer_age_seconds=0), 0
                    )
        controller = CaptureAdmission(0.2, None)
        controller.failure(0.0, "writer_error")
        self.assertEqual(
            controller.observe(1.0, occupancy=1.0, writer_age_seconds=100), 0.2
        )
        self.assertEqual(controller.stats(1.0)["failures"], 0)

    def test_pressure_is_interval_limited_and_recovery_has_hysteresis(self):
        controller = CaptureAdmission(0.2, AdaptiveCaptureConfig())
        self.assertEqual(
            controller.observe(1, occupancy=0.75, writer_age_seconds=0), 0.1
        )
        self.assertEqual(
            controller.observe(1.1, occupancy=1, writer_age_seconds=0), 0.1
        )
        self.assertEqual(controller.observe(2, occupancy=1, writer_age_seconds=0), 0.05)
        self.assertEqual(
            controller.observe(3, occupancy=0.5, writer_age_seconds=0), 0.05
        )
        self.assertAlmostEqual(
            controller.observe(3, occupancy=0.25, writer_age_seconds=0), 0.07
        )
        self.assertAlmostEqual(
            controller.observe(3.1, occupancy=0, writer_age_seconds=0), 0.07
        )
        for now in range(4, 30):
            controller.observe(now, occupancy=0, writer_age_seconds=0)
        self.assertEqual(controller.ratio(30), 0.2)
        self.assertEqual(controller.stats(30)["reason"], "configured")
        self.assertEqual(controller.stats(30, disabled=True)["effective_ratio"], 0)

    def test_stalled_writer_pauses_until_after_drain_and_cooldown(self):
        controller = CaptureAdmission(1.0, AdaptiveCaptureConfig())
        self.assertEqual(
            controller.observe(10, occupancy=0.25, writer_age_seconds=10), 0
        )
        self.assertEqual(
            controller.observe(14, occupancy=0.25, writer_age_seconds=14), 0
        )
        self.assertEqual(controller.observe(18, occupancy=0, writer_age_seconds=0), 0)
        self.assertAlmostEqual(
            controller.observe(19, occupancy=0, writer_age_seconds=0), 0.35
        )
        self.assertEqual(controller.stats(19)["pauses"], 1)

    def test_failure_burst_extends_cooldown_without_repeated_decreases(self):
        controller = CaptureAdmission(1.0, AdaptiveCaptureConfig())
        for now in (10, 10.1, 10.2):
            controller.failure(now, "catalog_error")
        self.assertEqual(controller.target_ratio, 0.5)
        self.assertEqual(controller.ratio(15), 0)
        self.assertEqual(controller.ratio(15.2), 0.5)
        self.assertEqual(controller.stats(15.2)["failures"], 3)
        for now in range(20, 200):
            controller.observe(now, occupancy=1, writer_age_seconds=0)
        self.assertEqual(controller.ratio(200), 0.01)
        self.assertLess(controller.stats(200)["decreases"], 10)

    def test_expired_cooldown_requires_a_fresh_observation_to_clear_a_stall(self):
        controller = CaptureAdmission(
            1.0, AdaptiveCaptureConfig(writer_stall_seconds=0.02, cooldown_seconds=0.01)
        )
        controller.observe(1, occupancy=0.5, writer_age_seconds=0.03)
        self.assertEqual(controller.stats(2)["effective_ratio"], 0)
        self.assertEqual(controller.stats(2)["cooldown_remaining_seconds"], 0)
        self.assertGreater(controller.observe(2, occupancy=0, writer_age_seconds=0), 0)


if __name__ == "__main__":
    unittest.main()
