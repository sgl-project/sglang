"""CPU tests for standalone benchmark-to-Chrome-trace conversion."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/convert_benchmark_to_trace.py"
spec = importlib.util.spec_from_file_location("convert_benchmark_to_trace", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class TestBenchmarkTrace(CustomTestCase):
    def setUp(self):
        self.result = {
            "backend": "sglang",
            "arrival_times": [0.25, 0.3],
            "ttfts": [0.1, 0.2],
            "itls": [[0.02, 0.03], [0.04]],
            "input_lens": [16, 32],
            "output_lens": [3, 2],
            "successes": [True, True],
        }

    def test_overlapping_requests_keep_shared_origin_and_cumulative_itls(self):
        trace = converter.convert_to_chrome_trace(self.result)
        events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
        expected = [
            (1, "TTFT", 250000, 100000),
            (1, "Output interval 1", 350000, 20000),
            (1, "Output interval 2", 370000, 30000),
            (2, "TTFT", 300000, 200000),
            (2, "Output interval 1", 500000, 40000),
        ]
        self.assertEqual(len(events), len(expected))
        for event, (tid, name, timestamp, duration) in zip(events, expected):
            self.assertEqual(
                (event["pid"], event["tid"], event["name"]), (1, tid, name)
            )
            self.assertAlmostEqual(event["ts"], timestamp)
            self.assertAlmostEqual(event["dur"], duration)
        self.assertEqual(trace["displayTimeUnit"], "ms")
        self.assertEqual(trace["metadata"]["requests_included"], 2)

    def test_output_lengths_do_not_fabricate_token_events(self):
        self.result["output_lens"] = [100, 50]
        trace = converter.convert_to_chrome_trace(self.result)
        events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0]["args"]["output_tokens"], 100)
        self.assertIn("chunks", trace["metadata"]["timing_semantics"])

    def test_zero_interval_remains_visible_without_shifting_next_event(self):
        self.result["itls"][0] = [0, 0.03]
        trace = converter.convert_to_chrome_trace(self.result)
        events = [
            event
            for event in trace["traceEvents"]
            if event.get("tid") == 1 and event["ph"] != "M"
        ]
        self.assertEqual(events[1]["ph"], "i")
        self.assertEqual(events[1]["s"], "t")
        self.assertAlmostEqual(events[1]["ts"], 350000)
        self.assertAlmostEqual(events[2]["ts"], 350000)

    def test_failed_missing_start_and_no_output_requests_are_skipped(self):
        result = {
            "arrival_times": [0, None, 0.1, 0.2],
            "ttfts": [0.1, 0.2, 0, 0.3],
            "itls": [[], [], [], []],
            "successes": [False, True, True, True],
        }
        trace = converter.convert_to_chrome_trace(result)
        self.assertEqual(trace["metadata"]["requests_total"], 4)
        self.assertEqual(trace["metadata"]["requests_included"], 1)
        self.assertEqual(
            trace["metadata"]["requests_skipped"],
            {"failed": 1, "unavailable_start": 1, "no_first_output": 1},
        )
        events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
        self.assertEqual([event["tid"] for event in events], [4])

    def test_positive_lane_ids_and_integer_shared_boundaries(self):
        self.result["arrival_times"] = [0.1234567891, 0.1276543219]
        self.result["ttfts"] = [0.0234567891, 0.0321654987]
        self.result["itls"] = [[0.0123456789, 0.00000001, 0.0216549873], [0.0234567891]]
        trace = converter.convert_to_chrome_trace(self.result)
        events = trace["traceEvents"]
        names = {
            event["tid"]: event["args"]["name"]
            for event in events
            if event["name"] == "thread_name"
        }
        self.assertEqual(names, {1: "Request 0", 2: "Request 1"})
        for tid in (1, 2):
            lane = [
                event
                for event in events
                if event.get("tid") == tid and event["ph"] != "M"
            ]
            self.assertEqual(lane[0]["args"]["request_index"], tid - 1)
            for event in lane:
                self.assertIsInstance(event["ts"], int)
                if "dur" in event:
                    self.assertIsInstance(event["dur"], int)
            for previous, current in zip(lane, lane[1:]):
                self.assertEqual(previous["ts"] + previous.get("dur", 0), current["ts"])
        first_lane = [
            event for event in events if event.get("tid") == 1 and event["ph"] != "M"
        ]
        self.assertEqual(first_lane[0]["ts"], 123457)
        self.assertEqual(first_lane[0]["dur"], 23457)
        self.assertEqual(first_lane[2]["ph"], "i")

    def test_errors_can_identify_failure_without_successes(self):
        del self.result["successes"]
        self.result["errors"] = ["request failed", ""]
        trace = converter.convert_to_chrome_trace(self.result)
        self.assertEqual(trace["metadata"]["requests_included"], 1)
        self.assertEqual(trace["metadata"]["requests_skipped"]["failed"], 1)

    def test_optional_lengths_are_not_required(self):
        del self.result["input_lens"], self.result["output_lens"]
        trace = converter.convert_to_chrome_trace(self.result)
        self.assertEqual(trace["metadata"]["requests_included"], 2)

    def test_missing_or_misaligned_arrays_are_rejected(self):
        for key in ("arrival_times", "ttfts", "itls"):
            with self.subTest(missing=key):
                result = copy.deepcopy(self.result)
                del result[key]
                with self.assertRaisesRegex(ValueError, key):
                    converter.convert_to_chrome_trace(result)
        for key in (
            "arrival_times",
            "ttfts",
            "itls",
            "input_lens",
            "output_lens",
            "successes",
            "errors",
        ):
            with self.subTest(misaligned=key):
                result = copy.deepcopy(self.result)
                result[key] = []
                with self.assertRaises(ValueError):
                    converter.convert_to_chrome_trace(result)

    def test_invalid_timing_values_are_rejected(self):
        for key in ("arrival_times", "ttfts", "itls"):
            for value in (
                -0.01,
                float("nan"),
                float("inf"),
                True,
                "0.1",
                [0.1],
                10**400,
                1e308,
            ):
                with self.subTest(key=key, value=value):
                    result = copy.deepcopy(self.result)
                    result[key][0] = [value] if key == "itls" else value
                    with self.assertRaises(ValueError):
                        converter.convert_to_chrome_trace(result)
        self.result["itls"][0] = 0.1
        with self.assertRaisesRegex(ValueError, "itls"):
            converter.convert_to_chrome_trace(self.result)

    def test_cumulative_time_overflow_is_rejected(self):
        for result in (
            {"arrival_times": [1e302], "ttfts": [1e302], "itls": [[]]},
            {"arrival_times": [0], "ttfts": [1e302], "itls": [[1e302]]},
        ):
            with self.subTest(result=result):
                with self.assertRaisesRegex(ValueError, "cumulative time"):
                    converter.convert_to_chrome_trace(result)

    def test_cli_rejects_oversized_timings_without_traceback_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            output = Path(directory) / "result_trace.json"
            for value in (10**400, 1e308):
                with self.subTest(value=value):
                    result = copy.deepcopy(self.result)
                    result["arrival_times"][0] = value
                    path.write_text(json.dumps(result), encoding="utf-8")
                    process = subprocess.run(
                        [sys.executable, "-I", "-S", str(SCRIPT), "--input", str(path)],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn("Error:", process.stderr)
                    self.assertNotIn("Traceback", process.stderr)
                    self.assertFalse(output.exists())

    def test_pretty_json_and_jsonl_run_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(self.result, indent=2), encoding="utf-8")
            self.assertEqual(converter.load_benchmark_result(path), self.result)
            second = copy.deepcopy(self.result)
            second["arrival_times"][0] = 0.5
            path.write_text(
                json.dumps(self.result) + "\n\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(converter.load_benchmark_result(path), second)
            self.assertEqual(converter.load_benchmark_result(path, 0), self.result)
            for index in (2, -3):
                with (
                    self.subTest(index=index),
                    self.assertRaisesRegex(ValueError, "out of range"),
                ):
                    converter.load_benchmark_result(path, index)

    def test_invalid_input_files_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            for contents in ("", "[]", '{"a": 1}\nnot-json\n'):
                with self.subTest(contents=contents):
                    path.write_text(contents, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        converter.load_benchmark_result(path)

    def test_cli_runs_without_site_packages_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            original = json.dumps(self.result)
            path.write_text(original, encoding="utf-8")
            # Isolated mode and no site initialization ensure no SGLang install
            # or third-party package is needed by the standalone converter.
            command = [sys.executable, "-I", "-S", str(SCRIPT), "--input", str(path)]
            process = subprocess.run(
                command, cwd=directory, capture_output=True, text=True, timeout=20
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            trace = json.loads(
                (Path(directory) / "result_trace.json").read_text(encoding="utf-8")
            )
            self.assertEqual(trace["metadata"]["requests_included"], 2)
            process = subprocess.run(
                command + ["--output", str(path)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("differ", process.stderr)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_cli_preserves_input_when_output_is_a_hardlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            output = Path(directory) / "hardlink.json"
            original = json.dumps(self.result)
            path.write_text(original, encoding="utf-8")
            try:
                output.hardlink_to(path)
            except OSError as exc:
                self.skipTest(f"Filesystem does not support hardlinks: {exc}")
            process = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(SCRIPT),
                    "--input",
                    str(path),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("differ", process.stderr)
            self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
