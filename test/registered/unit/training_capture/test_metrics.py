"""Prometheus lifecycle counters, bounded labels and zeroed state transitions."""

import math
import unittest
from concurrent.futures import ThreadPoolExecutor

from prometheus_client import CollectorRegistry, generate_latest
from sglang.srt.training_capture.admission import CaptureAdmission
from sglang.srt.training_capture.config import (
    AdaptiveCaptureConfig,
    CaptureLatencyConfig,
)
from sglang.srt.training_capture.metrics import CaptureMetrics
from sglang.srt.training_capture.timings import CaptureTimings
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

    def test_routing_decisions_are_exported_separately_and_only_once(self):
        self.stats["request_router"] = {
            "considered": 13,
            "selected": 3,
            "sampled_out": 8,
            "excluded": 1,
            "backpressure": 1,
            "attached": 3,
            "bound": 3,
        }
        for _ in range(2):
            self.metrics.update(self.stats)
        self.assertEqual(self.value("routing_events_total", event="sampled_out"), 8)
        self.assertEqual(self.value("routing_events_total", event="selected"), 3)
        self.assertEqual(self.value("routing_events_total", event="backpressure"), 1)
        self.assertEqual(self.value("events_total", event="admitted"), 2)
        self.assertEqual(self.value("events_total", event="sampled_out"), 3)
        self.stats["request_router"]["sampled_out"] = 0
        self.metrics.update(self.stats)
        self.assertEqual(self.value("routing_events_total", event="sampled_out"), 8)
        self.stats["request_router"]["sampled_out"] = 10
        self.metrics.update(self.stats)
        self.assertEqual(self.value("routing_events_total", event="sampled_out"), 10)

    def test_routing_labels_are_bounded_and_missing_router_has_no_samples(self):
        self.metrics.update(self.stats)
        self.assertIsNone(self.value("routing_events_total", event="selected"))
        self.stats["request_router"] = {
            f"private-request-{index}": 1 for index in range(100)
        }
        self.metrics.update(self.stats)
        self.metrics.update(self.stats)
        self.assertEqual(self.value("routing_events_total", event="other"), 100)
        samples = [
            s
            for family in self.registry.collect()
            for s in family.samples
            if s.name == "sglang:training_capture_routing_events_total"
        ]
        self.assertEqual(len(samples), len(CaptureMetrics.ROUTING_EVENTS))
        self.assertNotIn("private-", generate_latest(self.registry).decode())

    def test_follower_bindings_do_not_count_as_ingress_selections(self):
        self.stats["request_router"] = {"attached": 3, "bound": 3, "cancelled": 1}
        self.metrics.update(self.stats)
        self.assertEqual(self.value("routing_events_total", event="selected"), 0)
        self.assertEqual(self.value("routing_events_total", event="considered"), 0)
        self.assertEqual(self.value("routing_events_total", event="bound"), 3)
        self.assertEqual(self.value("routing_events_total", event="cancelled"), 1)

    def test_device_arena_metrics_keep_legacy_aliases_and_clear_on_close(self):
        for allocated in (256, 0):
            self.stats["host_pool"]["device_allocated_bytes"] = allocated
            self.metrics.update(self.stats)
            for prefix in ("device", "kv_staging"):
                self.assertEqual(self.value(prefix + "_allocated_bytes"), allocated)
                self.assertEqual(self.value(prefix + "_limit_bytes"), 512)

    def test_export_byte_counters_are_idempotent_and_have_bounded_destinations(self):
        self.metrics.update(self.stats)
        for destination in ("host", "device"):
            self.assertEqual(
                self.value("kv_export_enqueued_bytes_total", destination=destination), 0
            )
        self.stats["host_pool"].update(
            kv_export_host_enqueued_bytes=100,
            kv_export_device_enqueued_bytes=200,
            kv_export_private_request_enqueued_bytes=999,
        )
        self.metrics.update(self.stats)
        self.metrics.update(self.stats)
        self.stats["host_pool"]["kv_export_host_enqueued_bytes"] = 0
        self.metrics.update(self.stats)
        self.assertEqual(
            self.value("kv_export_enqueued_bytes_total", destination="host"), 100
        )
        self.stats["host_pool"]["kv_export_host_enqueued_bytes"] = 250
        self.metrics.update(self.stats)
        self.assertEqual(
            self.value("kv_export_enqueued_bytes_total", destination="host"), 250
        )
        self.assertEqual(
            self.value("kv_export_enqueued_bytes_total", destination="device"), 200
        )
        samples = [
            sample
            for family in self.registry.collect()
            for sample in family.samples
            if sample.name == "sglang:training_capture_kv_export_enqueued_bytes_total"
        ]
        self.assertEqual(len(samples), 2)
        self.assertNotIn("private_request", generate_latest(self.registry).decode())

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

    def test_operator_pause_has_its_own_gauge_and_bounded_action_counters(self):
        self.stats["admission_paused"] = True
        self.stats["counters"].update(control_pause=1, control_abort=2)
        self.metrics.update(self.stats)
        self.assertEqual(self.value("admission_paused"), 1)
        self.assertEqual(self.value("disabled"), 0)
        self.assertEqual(self.value("events_total", event="control_abort"), 2)
        self.stats["admission_paused"] = False
        self.stats["counters"]["control_resume"] = 1
        self.metrics.update(self.stats)
        self.assertEqual(self.value("admission_paused"), 0)
        self.assertEqual(self.value("events_total", event="control_resume"), 1)

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

    def test_stage_counters_are_bounded_and_repeated_snapshots_are_idempotent(self):
        timings = CaptureTimings()
        timings.observe("store_payload", 0.25)
        timings.observe("store_payload", 0.5, failed=True)
        self.stats["stage_timings"] = timings.stats()
        self.stats["stage_timings"]["private-request-id"] = {
            "calls": 999,
            "errors": 0,
            "seconds": 1,
            "max_seconds": 1,
        }
        self.metrics.update(self.stats)
        self.metrics.update(self.stats)
        self.assertEqual(self.value("stage_calls_total", stage="store_payload"), 2)
        self.assertEqual(self.value("stage_failures_total", stage="store_payload"), 1)
        self.assertEqual(self.value("stage_seconds_total", stage="store_payload"), 0.75)
        self.assertEqual(self.value("stage_max_seconds", stage="store_payload"), 0.5)
        self.assertNotIn("private-request-id", generate_latest(self.registry).decode())
        timings.observe("store_payload", 0.125)
        self.stats["stage_timings"] = timings.stats()
        self.metrics.update(self.stats)
        self.assertEqual(self.value("stage_calls_total", stage="store_payload"), 3)
        self.assertEqual(
            self.value("stage_seconds_total", stage="store_payload"), 0.875
        )
        self.assertEqual(self.value("stage_calls_total", stage="recovery_read"), 0)


class TestCaptureTimings(CustomTestCase):
    def test_records_wall_time_and_failure_without_changing_callback_result(self):
        clock = iter([1.0, 1.25, 2.0, 2.5])
        timings = CaptureTimings(clock=lambda: next(clock))
        result = object()
        self.assertIs(timings.call("store_payload", lambda: result), result)
        error = OSError("lost response")
        with self.assertRaises(OSError) as raised, timings.measure("store_payload"):
            raise error
        self.assertIs(raised.exception, error)
        self.assertEqual(
            timings.stats()["store_payload"],
            {"calls": 2, "errors": 1, "seconds": 0.75, "max_seconds": 0.5},
        )

    def test_concurrent_observations_and_detached_snapshots(self):
        timings = CaptureTimings()

        def observe():
            for _ in range(100):
                timings.observe("copy_wait", 0.25)
                self.assertEqual(set(timings.stats()), set(CaptureTimings.STAGES))

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(observe) for _ in range(4)]
            for future in futures:
                future.result(timeout=5)
        snapshot = timings.stats()
        self.assertEqual(snapshot["copy_wait"]["calls"], 400)
        self.assertEqual(snapshot["copy_wait"]["seconds"], 100)
        snapshot["copy_wait"]["calls"] = -1
        self.assertEqual(timings.stats()["copy_wait"]["calls"], 400)

    def test_unknown_stage_and_invalid_observations_do_not_grow_state(self):
        timings = CaptureTimings()
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                timings.observe("store_payload", value)
        with self.assertRaises(ValueError):
            timings.observe("private-request-id", 1)
        with self.assertRaises(ValueError), timings.measure("private-request-id"):
            self.fail("unknown stage started work")
        self.assertEqual(set(timings.stats()), set(CaptureTimings.STAGES))
        self.assertEqual(sum(v["calls"] for v in timings.stats().values()), 0)


if __name__ == "__main__":
    unittest.main()
