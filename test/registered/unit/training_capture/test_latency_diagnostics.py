"""Pause observations must preserve clocks, bounds and temporal associations."""

import asyncio
import copy
import gc
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.training_capture_benchmark_client import main
from sglang.test.training_capture_diagnostics import (
    ClientDiagnostics,
    clock_anchor,
    correlate_pauses,
    install_tokenizer_gc_observer,
    scheduler_gc_events,
    tokenizer_gc_events,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestClientDiagnostics(unittest.TestCase):
    def test_clock_sample_bounds_wall_read(self):
        with (
            patch("time.perf_counter", side_effect=[10.0, 10.004]),
            patch("time.time_ns", return_value=100_000_000_000),
        ):
            anchor = clock_anchor()
        self.assertAlmostEqual(anchor["perf_counter"], 10.002)
        self.assertAlmostEqual(anchor["uncertainty_seconds"], 0.002)
        self.assertEqual(anchor["unix_seconds"], 100.0)

    def test_real_gc_callback_is_removed_without_changing_policy(self):
        original = list(gc.callbacks)
        policy = gc.isenabled(), gc.get_threshold()
        observer = ClientDiagnostics()
        observer.start()
        try:
            gc.collect(0)
        finally:
            report = observer.close()
        self.assertEqual(gc.callbacks, original)
        self.assertEqual((gc.isenabled(), gc.get_threshold()), policy)
        self.assertTrue(any(event["kind"] == "client_gc" for event in report["events"]))
        self.assertEqual(report["incomplete_gc"], 0)
        with self.assertRaises(RuntimeError):
            observer.start()

    def test_event_bound_and_loop_deadline_are_explicit(self):
        observer = ClientDiagnostics(max_events=1)
        observer.active = True
        observer.loop = Mock()
        with patch("time.perf_counter", return_value=1.5):
            observer._tick(1.0)
            observer._tick(1.1)
        self.assertEqual(len(observer.events), 1)
        self.assertEqual(observer.dropped, 1)
        self.assertEqual(observer.events[0]["start"], 1.0)
        self.assertEqual(observer.events[0]["end"], 1.5)
        observer.loop.call_later.assert_called_with(0.01, observer._tick, 1.51)

    def test_one_loop_is_observed_and_timer_is_cancelled(self):
        observer = ClientDiagnostics()
        loop = Mock()
        observer.start()
        try:
            with patch("asyncio.get_running_loop", return_value=loop):
                observer.watch_loop()
                observer.watch_loop()
            self.assertEqual(loop.call_later.call_count, 1)
            with (
                patch("asyncio.get_running_loop", return_value=Mock()),
                self.assertRaises(RuntimeError),
            ):
                observer.watch_loop()
        finally:
            observer.close()
        loop.call_later.return_value.cancel.assert_called_once()

    def test_failed_native_cli_restores_hooks_and_retains_diagnostics(self):
        original = Mock()
        serving = SimpleNamespace(
            ASYNC_REQUEST_FUNCS={"sglang": original},
            cli_main=Mock(side_effect=ValueError("native failure")),
        )
        callbacks = list(gc.callbacks)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.json"
            argv = [
                "client",
                "--backend",
                "sglang",
                "--capture-request-records",
                str(path),
                "--capture-latency-diagnostics",
            ]
            with (
                patch.dict(
                    sys.modules, {"sglang.benchmark": SimpleNamespace(serving=serving)}
                ),
                patch.object(sys, "argv", argv),
            ):
                with self.assertRaisesRegex(ValueError, "native failure"):
                    main()
                self.assertIs(sys.argv, argv)
            report = json.loads(path.read_text())
        self.assertIs(serving.ASYNC_REQUEST_FUNCS["sglang"], original)
        self.assertEqual(gc.callbacks, callbacks)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["diagnostics"]["clock_anchors"]), 2)


class TestPauseCorrelation(unittest.TestCase):
    def setUp(self):
        self.diagnostics = {
            "schema_version": 1,
            "dropped_events": 0,
            "incomplete_gc": 0,
            "threshold_seconds": 0.02,
            "clock_anchors": [
                {"perf_counter": t, "unix_seconds": t + 1000, "uncertainty_seconds": 0}
                for t in (10, 12)
            ],
            "events": [
                {"kind": "client_gc", "start": 10.05, "end": 10.15},
                {"kind": "client_loop_lag", "start": 10.08, "end": 10.16},
            ],
        }
        self.records = [
            {
                "request_index": i,
                "trace_id": str(i),
                "start_time": start,
                "ttft": ttft,
                "latency": latency,
                "output_len": 3,
            }
            for i, start, ttft, latency in ((0, 10, 0.1, 0.2), (1, 11, 0.2, 0.4))
        ]
        self.server = scheduler_gc_events(
            "GC start: Time 1009.0 | Generation 0\n"
            "GC end: Time 1009.1 | Generation 0 | Duration: 0.1000s\n"
            "GC start: Time 1011.1 | Generation 2\n"
            "GC end: Time 1011.3 | Generation 2 | Duration: 0.2000s\n"
        )

    def summarize(self):
        return correlate_pauses(self.records, self.diagnostics, self.server, {"1"})

    def test_client_and_scheduler_spans_join_without_adding_overlapping_pauses(self):
        result = self.summarize()
        self.assertTrue(result["clock_mapping_stable"])
        self.assertEqual(result["server_events_mapped"], 2)
        events = result["long_events"]
        self.assertEqual(
            [e["overlapping_requests"] for e in events], [[0], [0], [], [1]]
        )
        first, second = result["worst_ttft"]
        self.assertTrue(first["published"])
        self.assertEqual(first["request_index"], 1)
        self.assertAlmostEqual(first["overlaps"][0]["ttft_overlap_ms"], 100)
        self.assertAlmostEqual(
            first["overlaps"][0]["after_first_token_overlap_ms"], 100
        )
        self.assertFalse(second["published"])
        self.assertEqual(len(second["overlaps"]), 2)
        self.assertAlmostEqual(second["overlaps"][0]["ttft_overlap_ms"], 50)
        self.assertAlmostEqual(
            second["overlaps"][0]["after_first_token_overlap_ms"], 50
        )
        self.assertEqual(result, self.summarize())

    def test_clock_step_disables_server_mapping_but_keeps_client_observations(self):
        self.diagnostics["clock_anchors"][1]["unix_seconds"] += 0.1
        result = self.summarize()
        self.assertFalse(result["clock_mapping_stable"])
        self.assertEqual(result["server_events_mapped"], 0)
        self.assertEqual(len(result["long_events"]), 2)
        self.assertIsNone(result["worst_ttft"][0]["start_unix_seconds"])

    def test_maximum_exposes_spikes_affecting_less_than_one_percent(self):
        self.records = [
            dict(
                self.records[1],
                request_index=index,
                trace_id=str(index),
                ttft=0.6 if index == 0 else 0.01,
                latency=0.7,
            )
            for index in range(1024)
        ]
        result = self.summarize()
        self.assertEqual(result["max_ttft_ms"], 600)
        self.assertEqual(result["ttft_over_500ms_requests"], 1)

    def test_truncation_and_out_of_window_requests_are_rejected(self):
        for field in ("dropped_events", "incomplete_gc"):
            with self.subTest(field=field):
                self.diagnostics[field] = 1
                with self.assertRaises(ValueError):
                    self.summarize()
                self.diagnostics[field] = 0
        self.records[1]["start_time"] = 12
        with self.assertRaises(ValueError):
            self.summarize()

    def test_invalid_clock_or_event_cannot_produce_a_false_join(self):
        original = copy.deepcopy(self.diagnostics)
        for target, field, value in (
            ("clock_anchors", "uncertainty_seconds", -1),
            ("clock_anchors", "unix_seconds", float("nan")),
            ("events", "end", 9),
            ("events", "start", float("inf")),
        ):
            self.diagnostics = copy.deepcopy(original)
            self.diagnostics[target][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.summarize()

    def test_scheduler_parser_reports_incomplete_logs_and_rejects_ambiguous_ranks(self):
        parsed = scheduler_gc_events(
            "GC end: Time 1000.0 | Generation 0\n"
            "GC start: Time 1000.1 | Generation 0\n"
        )
        self.assertEqual(parsed["unmatched_starts"], 1)
        self.assertEqual(parsed["unmatched_ends"], 1)
        for text in (
            "GC start: Time 1000.1 | Generation 0\nGC start: Time 1000.2 | Generation 0",
            "GC start: Time 1000.1 | Generation 0\nGC end: Time 1000.0 | Generation 0",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                scheduler_gc_events(text)


class TestTokenizerGCObservation(unittest.TestCase):
    def test_structured_event_preserves_process_clock_and_generation(self):
        with self.assertLogs("sglang.test.training_capture_diagnostics") as logs:
            callback = install_tokenizer_gc_observer(0.02)
            try:
                with (
                    patch("time.perf_counter", return_value=10.0) as mono,
                    patch("time.time", return_value=1010.0) as wall,
                ):
                    callback("start", {"generation": 2})
                    mono.return_value = 10.1
                    wall.return_value = 1010.1
                    callback("stop", {"generation": 2, "collected": 5})
            finally:
                gc.callbacks.remove(callback)
        parsed = tokenizer_gc_events("\n".join(logs.output))
        self.assertEqual(len(parsed["observers"]), 1)
        event = parsed["events"][0]
        self.assertEqual(event["generation"], 2)
        self.assertEqual(event["collected"], 5)
        self.assertAlmostEqual(event["duration_seconds"], 0.1)
        self.assertEqual(event["start_unix_seconds"], 1010.0)
        self.assertEqual(event["pid"], parsed["observers"][0]["pid"])

    def test_missing_installation_is_not_evidence_of_no_tokenizer_gc(self):
        with self.assertRaises(ValueError):
            tokenizer_gc_events("")


class TestLiveLoopObservation(unittest.IsolatedAsyncioTestCase):
    async def test_known_block_is_observed_inside_request_window(self):
        observer = ClientDiagnostics(interval=0.001)
        observer.start()
        try:
            observer.watch_loop()
            start = time.perf_counter()
            await asyncio.sleep(0.005)
            time.sleep(0.05)  # noqa: ASYNC251 - inject a known event-loop block
            ttft = time.perf_counter() - start
            await asyncio.sleep(0.005)
            latency = time.perf_counter() - start
        finally:
            diagnostics = observer.close()
        result = correlate_pauses(
            [
                {
                    "request_index": 0,
                    "trace_id": "observed",
                    "start_time": start,
                    "ttft": ttft,
                    "latency": latency,
                    "output_len": 2,
                }
            ],
            diagnostics,
            scheduler_gc_events(""),
            set(),
        )
        self.assertTrue(
            any(
                event["kind"] == "client_loop_lag"
                and event["duration_ms"] >= 40
                and event["overlapping_requests"] == [0]
                for event in result["long_events"]
            )
        )
        self.assertTrue(result["worst_ttft"][0]["overlaps"])


if __name__ == "__main__":
    unittest.main()
