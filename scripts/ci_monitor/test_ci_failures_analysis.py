"""Tests for CI failure analysis and AMD job-name filtering.

Run with:
    python -m unittest discover -s scripts/ci_monitor -p 'test_ci_failures_analysis.py'
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ci_failures_analysis import (  # noqa: E402
    SGLangFailuresAnalyzer,
    _filter_legacy_amd_job_rows,
)


class TestAnalyzeTestFailuresForJob(unittest.TestCase):
    FAILED_A = "Test Summary: 1/2 passed\nFAILED:\n  test_a.py\n==========\n"
    FAILED_B = "Test Summary: 1/2 passed\nFAILED:\n  test_b.py\n==========\n"
    INCOMPLETE_B = "python3 test_b.py\nserver_args: Namespace()\n"

    def setUp(self):
        self.analyzer = SGLangFailuresAnalyzer(token="test-token")
        self.addCleanup(self.analyzer.session.close)

    def _analyze_runs(self, runs):
        # Build chronological job history from (conclusion, log text) pairs.
        recent_runs = []
        logs_by_job = {}
        for run_number, (conclusion, logs) in enumerate(runs, start=1):
            recent_runs.append(
                {
                    "job_id": run_number,
                    "run_number": run_number,
                    "job_name": "test-job",
                    "job_url": f"https://example.com/jobs/{run_number}",
                    "conclusion": conclusion,
                }
            )
            logs_by_job[run_number] = logs

        # Keep the real log parser and analyzer; avoid HTTP requests and delays.
        with (
            patch.object(
                self.analyzer, "get_job_logs", side_effect=logs_by_job.__getitem__
            ),
            patch("ci_failures_analysis.time.sleep"),
        ):
            return self.analyzer.analyze_test_failures_for_job(recent_runs)

    def test_incomplete_summary_records_unknown_not_pass(self):
        result = self._analyze_runs(
            [("failure", self.FAILED_A), ("failure", self.INCOMPLETE_B)]
        )

        self.assertIn("test_a.py", result)
        history = result["test_a.py"]["recent_runs"]
        self.assertEqual([run["run_number"] for run in history], [1, 2])
        self.assertIs(history[0]["failed"], True)
        self.assertIsNone(
            history[1]["failed"],
            "An incomplete summary does not confirm that test_a.py passed.",
        )
        self.assertEqual(history[1]["status"], "\u26aa")
        self.assertEqual(history[1]["job_url"], "https://example.com/jobs/2")

    def test_incomplete_summary_preserves_failure_streak(self):
        result = self._analyze_runs(
            [("failure", self.FAILED_A), ("failure", self.INCOMPLETE_B)]
        )

        self.assertIn("test_a.py", result)
        self.assertEqual(result["test_a.py"]["total_failures"], 1)
        self.assertEqual(
            result["test_a.py"]["current_streak"],
            1,
            "An unknown outcome must not reset the previous failure streak.",
        )

    def test_cancelled_run_keeps_previous_failure(self):
        result = self._analyze_runs([("failure", self.FAILED_A), ("cancelled", "")])

        self.assertIn(
            "test_a.py", result, "Cancellation must not hide a previous failure."
        )
        self.assertEqual(result["test_a.py"]["current_streak"], 1)
        self.assertEqual(result["test_a.py"]["total_failures"], 1)
        self.assertEqual(
            [run["failed"] for run in result["test_a.py"]["recent_runs"]],
            [True, None],
        )

    def test_skipped_run_keeps_previous_failure(self):
        result = self._analyze_runs([("failure", self.FAILED_A), ("skipped", "")])

        self.assertIn(
            "test_a.py", result, "Skipping a run must not hide a previous failure."
        )
        self.assertEqual(result["test_a.py"]["current_streak"], 1)
        self.assertEqual(result["test_a.py"]["total_failures"], 1)
        self.assertIsNone(result["test_a.py"]["recent_runs"][-1]["failed"])

    def test_missing_logs_keep_previous_failure(self):
        result = self._analyze_runs([("failure", self.FAILED_A), ("failure", "")])

        self.assertIn(
            "test_a.py", result, "Missing logs must not hide a previous failure."
        )
        self.assertEqual(result["test_a.py"]["current_streak"], 1)
        self.assertEqual(result["test_a.py"]["total_failures"], 1)
        self.assertIsNone(result["test_a.py"]["recent_runs"][-1]["failed"])

    def test_multiple_unknown_runs_keep_previous_failures(self):
        result = self._analyze_runs(
            [
                ("failure", self.FAILED_A),
                ("failure", self.FAILED_A),
                ("cancelled", ""),
                ("skipped", ""),
                ("cancelled", ""),
            ]
        )

        self.assertIn("test_a.py", result)
        self.assertEqual(result["test_a.py"]["current_streak"], 2)
        self.assertEqual(result["test_a.py"]["total_failures"], 2)
        self.assertEqual(
            [run["failed"] for run in result["test_a.py"]["recent_runs"]],
            [True, True, None, None, None],
        )

    def test_consecutive_failures_increment_streak(self):
        result = self._analyze_runs(
            [("failure", self.FAILED_A), ("failure", self.FAILED_A)]
        )

        self.assertEqual(result["test_a.py"]["current_streak"], 2)
        self.assertEqual(result["test_a.py"]["total_failures"], 2)
        self.assertEqual(
            [run["failed"] for run in result["test_a.py"]["recent_runs"]],
            [True, True],
        )

    def test_unknown_between_failures_does_not_count_as_failure(self):
        result = self._analyze_runs(
            [
                ("failure", self.FAILED_A),
                ("cancelled", ""),
                ("failure", self.FAILED_A),
            ]
        )

        self.assertEqual(result["test_a.py"]["current_streak"], 2)
        self.assertEqual(result["test_a.py"]["total_failures"], 2)
        self.assertEqual(
            [run["failed"] for run in result["test_a.py"]["recent_runs"]],
            [True, None, True],
        )

    def test_success_after_unknown_resets_streak(self):
        result = self._analyze_runs(
            [("failure", self.FAILED_A), ("cancelled", ""), ("success", "")]
        )

        self.assertEqual(result["test_a.py"]["current_streak"], 0)
        self.assertEqual(result["test_a.py"]["total_failures"], 1)
        self.assertEqual(
            [run["failed"] for run in result["test_a.py"]["recent_runs"]],
            [True, None, False],
        )

    def test_complete_summary_for_other_test_resets_streak(self):
        result = self._analyze_runs(
            [("failure", self.FAILED_A), ("failure", self.FAILED_B)]
        )

        self.assertEqual(result["test_a.py"]["current_streak"], 0)
        self.assertEqual(result["test_a.py"]["total_failures"], 1)
        self.assertIs(result["test_a.py"]["recent_runs"][-1]["failed"], False)
        self.assertEqual(result["test_a.py"]["recent_runs"][-1]["status"], "\u2705")
        self.assertEqual(result["test_b.py"]["current_streak"], 1)
        self.assertEqual(result["test_b.py"]["total_failures"], 1)

    def test_unknown_runs_without_failure_return_no_summary_marker(self):
        result = self._analyze_runs([("cancelled", ""), ("skipped", "")])

        self.assertEqual(result, {"_no_test_summary": True})


class TestFilterLegacyAmdJobRows(unittest.TestCase):
    def test_drops_legacy_names_and_nested_utilities(self):
        rows = {
            "stage-b-test-1-gpu-small-amd-rocm720 (linux-mi300-1gpu-sglang, 0)": {},
            "nightly-accuracy-2-gpu-rocm720 (rocm724)": {},
            "nightly-accuracy-2-gpu-rocm724": {},
            "nightly-test-1-gpu-unit (rocm724)": {},
            "call-pr-test-amd-rocm720 / call-pr-test-amd-extra-rocm720 / extra-a-test-1-gpu-small-amd (linux-mi300-1gpu-sglang)": {},
            "wait-for-stage-a-amd": {},
            "call-pr-test-amd-extra / pr-test-amd-extra-finish": {},
            "call-pr-test-amd-extra / call-gate / pr-gate": {},
        }

        self.assertEqual(_filter_legacy_amd_job_rows(rows), {})

    def test_keeps_current_flavors_and_nested_callers_separate(self):
        new_success = {"current_streak": 0}
        rows = {
            "stage-b-test-1-gpu-small-amd (rocm724, linux-mi300-1gpu-sglang, 0)": new_success,
            "nightly-accuracy-2-gpu (rocm720, linux-mi300-2gpu-sglang)": {
                "current_streak": 1
            },
            "call-pr-test-amd-rocm720 / stage-c-test-4-gpu-amd (rocm724, linux-mi300-4gpu-sglang, 0)": {
                "current_streak": 0
            },
            "call-pr-test-amd-rocm720 / call-pr-test-amd-extra / extra-a-test-1-gpu-small-amd (rocm724, linux-mi300-1gpu-sglang)": {
                "current_streak": 0
            },
        }

        filtered = _filter_legacy_amd_job_rows(rows)

        self.assertEqual(set(filtered), set(rows))
        self.assertIs(
            filtered[
                "stage-b-test-1-gpu-small-amd (rocm724, linux-mi300-1gpu-sglang, 0)"
            ],
            new_success,
        )


if __name__ == "__main__":
    unittest.main()
