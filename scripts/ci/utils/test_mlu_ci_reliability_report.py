"""Offline tests for the MLU CI report using GitHub API and artifact fixtures."""

import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlu_ci_reliability_report as report  # noqa: E402


class TestMLUCIReport(unittest.TestCase):
    def test_workflow_runs_walk_past_1000_with_boundary_overlap(self):
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=7)
        runs = [
            {
                "id": index,
                "created_at": report.isoformat(end - timedelta(seconds=index // 2)),
            }
            for index in range(1105)
        ]
        ranges = []

        def batch(api, path, key):
            lower, upper = parse_qs(urlsplit(path).query)["created"][0].split("..")
            ranges.append((lower, upper))
            return [run for run in runs if lower <= run["created_at"] <= upper][:1000]

        with patch.object(report, "paginate", side_effect=batch):
            result = report.list_workflow_runs(
                None, "example/sglang", "pr-test-mlu.yml", start, end, "all"
            )
        self.assertEqual([run["id"] for run in result], list(range(2, 1105)))
        self.assertEqual(
            ranges,
            [
                (report.isoformat(start), report.isoformat(end)),
                (report.isoformat(start), runs[999]["created_at"]),
            ],
        )

    def test_workflow_runs_reject_stalled_cursor(self):
        start = datetime(2026, 10, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=7)
        runs = [
            {"id": index, "created_at": report.isoformat(start)}
            for index in range(1000)
        ]
        with (
            patch.object(report, "paginate", return_value=runs),
            self.assertRaisesRegex(RuntimeError, "1000-result limit"),
        ):
            report.list_workflow_runs(
                None, "example/sglang", "pr-test-mlu.yml", start, end, "all"
            )

    def test_successful_run_produces_json_and_markdown(self):
        prefix = "/repos/example/sglang/actions"
        responses = {
            f"{prefix}/workflows/pr-test-mlu.yml/runs": {
                "workflow_runs": [
                    {
                        "id": 42,
                        "run_attempt": 1,
                        "created_at": "2026-10-06T10:00:00Z",
                        "event": "pull_request_target",
                    }
                ]
            },
            f"{prefix}/runs/42/attempts/1/jobs": {
                "jobs": [
                    {
                        "id": 101,
                        "name": "MLU / pr-test-1-mlu",
                        "runner_group_name": "mlu-ci",
                        "runner_name": "mlu-runner-1",
                        "status": "completed",
                        "conclusion": "success",
                        "created_at": "2026-10-06T10:00:00Z",
                        "started_at": "2026-10-06T10:01:00Z",
                        "completed_at": "2026-10-06T10:03:00Z",
                    }
                ]
            },
            f"{prefix}/runs/42/artifacts": {
                "artifacts": [
                    {
                        "id": 201,
                        "name": "mlu-ci-result-pr-42-1-pr-test-1-mlu",
                        "expired": False,
                    }
                ]
            },
        }
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(
                "result-task.json",
                json.dumps(
                    {"status": "success", "failure_type": "", "failure_stage": ""}
                ),
            )

        def request(path, **kwargs):
            if urlsplit(path).path == f"{prefix}/artifacts/201/zip":
                return archive.getvalue()
            return json.dumps(responses[urlsplit(path).path]).encode()

        with tempfile.TemporaryDirectory() as directory:
            markdown_path = Path(directory) / "report.md"
            json_path = Path(directory) / "report.json"
            argv = [
                "report",
                "--repo",
                "example/sglang",
                "--days",
                "7",
                "--end-time",
                "2026-10-07T00:00:00Z",
                "--output",
                str(markdown_path),
                "--json-output",
                str(json_path),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}),
                patch.object(report.GitHubApi, "request", side_effect=request),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(report.main(), 0)
            summary = json.loads(json_path.read_text())
            markdown = markdown_path.read_text()

        self.assertEqual(summary["counts"]["mlu_job_attempts"], 1)
        self.assertEqual(summary["counts"]["result_metadata"], 1)
        self.assertEqual(summary["classified_reliability"], 1.0)
        self.assertEqual(summary["metadata_coverage_for_assigned_jobs"], 1.0)
        self.assertEqual(summary["collection_errors"], [])
        self.assertEqual(summary["latency_seconds"]["queue_median"], 60)
        self.assertEqual(summary["latency_seconds"]["runtime_median"], 120)
        self.assertIn("<!-- mlu-ci-reliability-report -->", markdown)
        self.assertIn("| `success` | `success` | 1 |", markdown)
        self.assertIn("**100.0%**", markdown)

    def test_failed_jobs_rerun_skips_carried_over_executions(self):
        start = datetime(2026, 10, 6, tzinfo=timezone.utc)
        single = {
            "id": 101,
            "name": "MLU / one",
            "runner_group_name": "mlu-ci",
            "runner_name": "mlu-runner",
            "status": "completed",
            "started_at": "2026-10-06T10:01:00Z",
            "completed_at": "2026-10-06T10:03:00Z",
        }
        dual = {**single, "id": 102, "name": "MLU / two"}
        rerun = {**dual, "id": 202, "completed_at": "2026-10-06T10:05:00Z"}
        jobs = [[single, dual], [{**single, "id": 201}, rerun]]
        runs = [{"id": 42, "run_attempt": 2}]
        artifacts = [
            {"id": index, "name": f"mlu-ci-result-pr-42-{attempt}-{suite}"}
            for index, (attempt, suite) in enumerate(
                ((1, "one"), (1, "two"), (2, "two"))
            )
        ]

        def request(path, **kwargs):
            status = "failed" if path.endswith("/artifacts/1/zip") else "success"
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("result-task.json", json.dumps({"status": status}))
            return archive.getvalue()

        api = report.GitHubApi("test-token")
        with (
            patch.object(report, "list_workflow_runs", return_value=runs),
            patch.object(report, "list_attempt_jobs", side_effect=jobs),
            patch.object(report, "paginate", return_value=artifacts),
            patch.object(api, "request", side_effect=request),
        ):
            records, errors = report.collect_records(
                api, "a/b", "mlu.yml", "MLU / ", start, start + timedelta(days=1)
            )
        self.assertEqual(
            [(r["attempt"], r["job_id"]) for r in records],
            [(1, 101), (1, 102), (2, 202)],
        )
        self.assertEqual(
            [r["metadata"] for r in records],
            [{"status": "success"}, {"status": "failed"}, {"status": "success"}],
        )
        self.assertEqual(errors, [])

    def test_mixed_results_keep_missing_metadata_out_of_reliability(self):
        outcomes = [
            ("success", {"status": "success"}),
            (
                "failure",
                {
                    "status": "failed",
                    "failure_type": "infrastructure",
                    "failure_stage": "external_task",
                },
            ),
            (
                "failure",
                {
                    "status": "failed",
                    "failure_type": "test",
                    "failure_stage": "pytest",
                },
            ),
            ("failure", None),
        ]
        records = [
            {
                "run_id": run_id,
                "attempt": 1,
                "run_url": f"https://github.com/example/sglang/actions/runs/{run_id}",
                "job_url": None,
                "runner_assigned": True,
                "job_status": "completed",
                "job_conclusion": conclusion,
                "artifact_uploaded": metadata is not None,
                "artifact_expired": False,
                "metadata_error": None,
                "metadata": metadata,
            }
            for run_id, (conclusion, metadata) in enumerate(outcomes, start=1)
        ]
        summary = report.build_summary(
            records,
            [],
            "example/sglang",
            "pr-test-mlu.yml",
            datetime(2026, 10, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 8, tzinfo=timezone.utc),
        )

        self.assertEqual(summary["counts"]["mlu_job_attempts"], 4)
        self.assertEqual(summary["counts"]["result_metadata"], 3)
        self.assertEqual(summary["counts"]["infrastructure_results"], 1)
        self.assertAlmostEqual(summary["classified_reliability"], 2 / 3)
        self.assertEqual(summary["metadata_coverage_for_assigned_jobs"], 0.75)
        self.assertCountEqual(
            summary["classifications"],
            [
                {"failure_type": kind, "failure_stage": stage, "count": 1}
                for kind, stage in (
                    ("success", "success"),
                    ("infrastructure", "external_task"),
                    ("test", "pytest"),
                    ("missing_metadata", "no_result_metadata"),
                )
            ],
        )
        markdown = report.render_markdown(summary)
        self.assertIn("Classified non-infrastructure reliability: **66.7%**", markdown)
        self.assertIn(
            "Metadata coverage among completed runner-assigned jobs: **75.0%**",
            markdown,
        )
        self.assertIn(
            "[run 4 attempt 1](https://github.com/example/sglang/actions/runs/4): "
            "metadata missing, conclusion `failure`",
            markdown,
        )

    def test_empty_report_has_no_ratios_and_renders_markdown(self):
        summary = report.build_summary(
            [],
            [],
            "example/sglang",
            "pr-test-mlu.yml",
            datetime(2026, 10, 1, tzinfo=timezone.utc),
            datetime(2026, 10, 8, tzinfo=timezone.utc),
        )

        self.assertTrue(all(count == 0 for count in summary["counts"].values()))
        self.assertIsNone(summary["classified_reliability"])
        self.assertIsNone(summary["metadata_coverage_for_assigned_jobs"])
        self.assertEqual(summary["classifications"], [])
        self.assertEqual(summary["records"], [])
        self.assertTrue(
            all(value is None for value in summary["latency_seconds"].values())
        )
        markdown = report.render_markdown(summary)
        self.assertIn("MLU job attempts: **0**", markdown)
        self.assertIn("Classified non-infrastructure reliability: **n/a**", markdown)
        self.assertIn(
            "Metadata coverage among completed runner-assigned jobs: **n/a**",
            markdown,
        )
        self.assertIn("| `n/a` | `n/a` | 0 |", markdown)

    def test_success_status_requires_exact_match(self):
        for status, expected in (
            ("success", ("success", "success")),
            ("unsuccessful", ("unknown", "unsuccessful")),
        ):
            with self.subTest(status=status):
                self.assertEqual(
                    report.result_bucket({"metadata": {"status": status}}), expected
                )


if __name__ == "__main__":
    unittest.main()
