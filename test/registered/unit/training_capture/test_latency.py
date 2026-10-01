"""Latency budgets, stale-signal protection and committed-token accounting."""

import unittest
from types import SimpleNamespace

import msgspec
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.training_capture.admission import CaptureAdmission
from sglang.srt.training_capture.config import (
    AdaptiveCaptureConfig,
    CaptureLatencyConfig,
)
from sglang.srt.training_capture.latency import CaptureLatency, request_latency
from sglang.srt.training_capture.protocol import ContractError
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCaptureLatency(CustomTestCase):
    def config(self, **kwargs):
        values = {
            "ttft_seconds": 1.0,
            "tpot_seconds": 0.1,
            "min_observations": 2,
            "max_observations": 4,
            "window_seconds": 2.0,
        }
        values.update(kwargs)
        config = msgspec.convert(values, type=CaptureLatencyConfig)
        config.validate()
        return config

    def controller(self, **kwargs):
        return CaptureAdmission(
            1.0,
            AdaptiveCaptureConfig(
                interval_seconds=0.1,
                cooldown_seconds=0.5,
                latency=self.config(**kwargs),
            ),
        )

    def assess(self, controller, now):
        return controller.observe(now, occupancy=0, writer_age_seconds=0)

    def test_breach_pauses_and_fresh_healthy_samples_restore_after_cooldown(self):
        controller = self.controller()
        latency = controller.latency
        latency.observe(1, ttft=1.5, tpot=0.02)
        self.assertEqual(self.assess(controller, 1), 1)
        latency.observe(1.2, ttft=1.5, tpot=0.02)
        self.assertEqual(self.assess(controller, 1.2), 0)
        self.assertEqual(controller.reason, "latency_ttft")
        self.assertEqual(controller.target_ratio, 0.5)
        self.assertEqual(self.assess(controller, 4), 0)
        self.assertEqual(latency.state, "stale")
        self.assertEqual(self.assess(controller, 40), 0)
        latency.observe(40.1, ttft=0.5, tpot=0.02)
        self.assertEqual(self.assess(controller, 40.1), 0)
        latency.observe(40.3, ttft=0.5, tpot=0.02)
        self.assertAlmostEqual(self.assess(controller, 40.3), 0.6)
        self.assertFalse(latency.blocked)
        self.assertTrue(latency.recovery_ready)
        self.assertEqual(latency.state, "healthy")

    def test_any_ready_metric_can_pause_and_hysteresis_prevents_oscillation(self):
        controller = self.controller(min_observations=1)
        latency = controller.latency
        latency.observe(1, tpot=0.2)
        self.assertEqual(self.assess(controller, 1), 0)
        self.assertEqual(controller.reason, "latency_tpot")
        latency.observe(4, ttft=0.5, tpot=0.09)
        self.assertEqual(self.assess(controller, 4), 0)
        self.assertEqual(controller.reason, "latency_hysteresis")
        latency.observe(7, ttft=0.5, tpot=0.02)
        self.assertGreater(self.assess(controller, 7), 0)

    def test_fresh_recovery_does_not_bypass_cooldown(self):
        controller = self.controller(min_observations=1, window_seconds=0.2)
        latency = controller.latency
        latency.observe(1, ttft=2, tpot=0.2)
        self.assertEqual(self.assess(controller, 1), 0)
        latency.observe(1.3, ttft=0.5, tpot=0.02)
        self.assertEqual(self.assess(controller, 1.3), 0)
        self.assertTrue(latency.recovery_ready)
        self.assertFalse(latency.blocked)
        self.assertEqual(controller.target_ratio, 0.5)
        latency.observe(1.6, ttft=0.5, tpot=0.02)
        self.assertAlmostEqual(self.assess(controller, 1.6), 0.6)

    def test_ring_is_bounded_and_quantile_uses_only_recent_finite_values(self):
        latency = CaptureLatency(self.config(tpot_seconds=None, percentile=0.5), 0.1)
        for value in range(10):
            latency.observe(1, ttft=value / 10)
        self.assertEqual(len(latency.samples["ttft"]), 4)
        latency.observe(1, ttft=float("nan"))
        latency.observe(1, ttft=float("inf"))
        latency.observe(1, ttft=-1)
        report = latency.stats(1)
        self.assertEqual(report["metrics"]["ttft"]["observations"], 10)
        self.assertEqual(report["invalid_observations"], 3)
        self.assertAlmostEqual(report["metrics"]["ttft"]["percentile_seconds"], 0.7)
        self.assertIsNone(latency.stats(4)["metrics"]["ttft"]["percentile_seconds"])
        self.assertEqual(latency.state, "stale")
        self.assertFalse(latency.recovery_ready)

    def test_invalid_configuration(self):
        for values in (
            {},
            {"ttft_seconds": 0},
            {"tpot_seconds": -1},
            {"ttft_seconds": float("inf")},
            {"ttft_seconds": 1, "min_observations": 5, "max_observations": 4},
            {"ttft_seconds": 1, "percentile": 0},
            {"ttft_seconds": 1, "recovery_fraction": 1},
        ):
            with (
                self.subTest(values=values),
                self.assertRaises((msgspec.ValidationError, ContractError)),
            ):
                config = msgspec.convert(values, type=CaptureLatencyConfig)
                config.validate()

    def test_observations_count_real_output_not_draft_width_or_duplicate_callbacks(
        self,
    ):
        req = SimpleNamespace(
            training_capture_latency=None,
            time_stats=SimpleNamespace(scheduler_recv_time=1.0),
            output_ids=[],
            finished_len=None,
            finished_reason=None,
            is_retracted=False,
        )
        req.finished = lambda: req.finished_reason is not None
        self.assertEqual(request_latency(req, 1.1), (None, None))
        req.output_ids.append(10)
        ttft, tpot = request_latency(req, 1.2)
        self.assertAlmostEqual(ttft, 0.2)
        self.assertIsNone(tpot)
        self.assertEqual(request_latency(req, 1.25), (None, None))
        req.output_ids += [11, 12, 13]
        self.assertAlmostEqual(request_latency(req, 1.5)[1], 0.1)
        req.is_retracted = True
        self.assertEqual(request_latency(req, 1.8), (None, None))
        req.is_retracted = False
        req.output_ids += [14, 15, 16]
        req.finished_len = 5
        req.finished_reason = FINISH_LENGTH(5)
        self.assertAlmostEqual(request_latency(req, 2)[1], 0.5)
        self.assertEqual(request_latency(req, 3), (None, None))

    def test_aborted_request_does_not_supply_a_successful_latency_sample(self):
        req = SimpleNamespace(
            training_capture_latency=None, finished_reason=FINISH_ABORT()
        )
        self.assertEqual(request_latency(req, 1), (None, None))
        self.assertTrue(req.training_capture_latency.done)


if __name__ == "__main__":
    unittest.main()
