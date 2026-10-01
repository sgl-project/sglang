"""Prometheus lifecycle counters, bounded labels and zeroed state transitions."""

import math
import unittest

from prometheus_client import CollectorRegistry, generate_latest
from sglang.srt.training_capture.admission import CaptureAdmission
from sglang.srt.training_capture.config import (
    AdaptiveCaptureConfig,
    CaptureLatencyConfig,
)
from sglang.srt.training_capture.metrics import CaptureMetrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCaptureMetrics(CustomTestCase):
    def setUp(self):
        self.registry = CollectorRegistry()
        self.labels = {"model_name": "test", "tp_rank": 0}
        self.metrics = CaptureMetrics(self.labels, registry=self.registry)
        self.stats = {
            "counters": {"admitted": 2, "ready": 1, "sampled_out": 3},
            "admission": CaptureAdmission(0.1, None).stats(0),
            "states": {"available": 1, "writing": 1},
            "host_pool": {
                "allocated_bytes": 1024,
                "device_allocated_bytes": 256,
                "device_limit_bytes": 512,
                "free": 0,
                "filling": 2,
                "quarantined": 0,
            },
            "disabled_reason": None,
            "queued": 0,
            "occupied_fraction": 0.5,
            "writer_age_seconds": 2.0,
        }

    def value(self, name, **labels):
        return self.registry.get_sample_value(
            "sglang:training_capture_" + name,
            {k: str(v) for k, v in {**self.labels, **labels}.items()},
        )

    def test_repeated_updates_count_each_event_once_and_clear_inactive_states(self):
        self.metrics.update(self.stats)
        self.metrics.update(self.stats)
        self.assertEqual(self.value("events_total", event="admitted"), 2)
        self.assertEqual(self.value("reservations", state="writing"), 1)
        self.assertEqual(self.value("kv_staging_allocated_bytes"), 256)
        self.assertEqual(self.value("kv_staging_limit_bytes"), 512)
        self.stats["counters"]["ready"] = 2
        self.stats["states"] = {"available": 2}
        self.stats["writer_age_seconds"] = 0
        self.stats["occupied_fraction"] = 0
        self.metrics.update(self.stats)
        self.assertEqual(self.value("events_total", event="ready"), 2)
        self.assertEqual(self.value("reservations", state="writing"), 0)
        self.assertEqual(self.value("writer_age_seconds"), 0)
        self.assertEqual(self.value("occupied_fraction"), 0)
        self.assertGreater(self.value("metrics_update_timestamp_seconds"), 0)

    def test_failures_have_bounded_labels_and_no_payload_identifiers(self):
        for index in range(200):
            self.stats["counters"][f"failed_private-request-{index}"] = 1
            self.stats["counters"][f"writer_failed_private-error-{index}"] = 1
            self.stats["counters"][f"excluded_private-feature-{index}"] = 1
        self.metrics.update(self.stats)
        self.metrics.update(self.stats)
        for event in ("capture_failed", "writer_failed", "excluded_unsupported"):
            self.assertEqual(self.value("events_total", event=event), 200)
        exported = generate_latest(self.registry).decode()
        self.assertNotIn("private-", exported)
        events = [
            sample
            for family in self.registry.collect()
            for sample in family.samples
            if sample.name == "sglang:training_capture_events_total"
        ]
        self.assertEqual(len(events), len(CaptureMetrics.EVENTS))

    def test_disabled_and_quarantined_are_distinct_from_adaptive_pause(self):
        self.stats["disabled_reason"] = "private-error-details"
        self.stats["admission"].update(effective_ratio=0, pauses=1)
        self.stats["host_pool"].update(filling=1, quarantined=1)
        self.metrics.update(self.stats)
        self.assertEqual(self.value("disabled"), 1)
        self.assertEqual(self.value("sample_ratio", kind="effective"), 0)
        self.assertEqual(self.value("sample_ratio", kind="configured"), 0.1)
        self.assertEqual(self.value("host_slots", state="quarantined"), 1)
        self.assertEqual(self.value("admission_adjustments_total", action="pauses"), 1)
        self.assertNotIn(
            "private-error-details", generate_latest(self.registry).decode()
        )

    def test_latency_series_distinguish_missing_observations_and_budget_breach(self):
        controller = CaptureAdmission(
            1.0,
            AdaptiveCaptureConfig(
                latency=CaptureLatencyConfig(
                    ttft_seconds=1.0, min_observations=1, window_seconds=2
                )
            ),
        )
        self.stats["admission"] = controller.stats(0)
        self.metrics.update(self.stats)
        self.assertEqual(self.value("latency_control_enabled"), 1)
        self.assertTrue(
            math.isnan(
                self.value("scheduler_latency_seconds", metric="ttft", kind="observed")
            )
        )
        controller.latency.observe(1, ttft=2)
        controller.observe(1, occupancy=0, writer_age_seconds=0)
        self.stats["admission"] = controller.stats(1)
        self.metrics.update(self.stats)
        self.metrics.update(self.stats)
        self.assertEqual(self.value("latency_blocked"), 1)
        self.assertEqual(
            self.value("scheduler_latency_seconds", metric="ttft", kind="observed"), 2
        )
        self.assertEqual(self.value("latency_observations_total", metric="ttft"), 1)
        self.stats["admission"] = controller.stats(4)
        self.metrics.update(self.stats)
        self.assertEqual(self.value("latency_state", state="stale"), 1)
        self.assertEqual(self.value("latency_state", state="breached"), 0)
        self.assertEqual(self.value("latency_blocked"), 1)
        self.assertEqual(self.value("latency_window_observations", metric="ttft"), 0)


if __name__ == "__main__":
    unittest.main()
