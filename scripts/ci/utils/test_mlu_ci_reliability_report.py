import importlib.util
import io
import json
import tempfile
import unittest
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("mlu_ci_reliability_report.py")
SPEC = importlib.util.spec_from_file_location("mlu_ci_reliability_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result_archive(payload):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        bundle.writestr("nested/result-task.json", json.dumps(payload))
    return output.getvalue()


def record(
    *,
    metadata,
    assigned=True,
    conclusion="success",
    run_id=1,
    queue_seconds=30,
    runtime_seconds=120,
):
    return {
        "run_id": run_id,
        "attempt": 1,
        "run_url": f"https://github.example/runs/{run_id}",
        "job_url": f"https://github.example/jobs/{run_id}",
        "runner_assigned": assigned,
        "job_status": "completed",
        "job_conclusion": conclusion,
        "queue_seconds": queue_seconds if assigned else None,
        "runtime_seconds": runtime_seconds if assigned else None,
        "artifact_uploaded": metadata is not None,
        "artifact_expired": False,
        "metadata": metadata,
        "metadata_error": "",
    }


class MluCiReliabilityReportTest(unittest.TestCase):
    def test_workflow_query_filters_pull_request_target_events(self):
        api = mock.Mock()
        api.get_json.return_value = {"workflow_runs": []}
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = datetime(2026, 7, 8, tzinfo=timezone.utc)

        self.assertEqual(
            MODULE.list_workflow_runs(
                api,
                "sgl-project/sglang",
                "pr-test-mlu.yml",
                start,
                end,
                "pull_request_target",
            ),
            [],
        )

        requested_path = api.get_json.call_args.args[0]
        self.assertIn("event=pull_request_target", requested_path)

    def test_cross_host_artifact_redirect_does_not_forward_token(self):
        request = urllib.request.Request(
            "https://api.github.com/repos/o/r/actions/artifacts/1/zip",
            headers={"Authorization": "Bearer secret"},
        )
        redirected = MODULE.StripCrossHostAuthorization().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://results-receiver.actions.githubusercontent.com/archive.zip",
        )

        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_reads_structured_result_without_extracting_archive(self):
        payload = {
            "status": "error",
            "failure_type": "timeout",
            "failure_stage": "mlu_resource_queue",
        }

        self.assertEqual(MODULE.read_result_metadata(result_archive(payload)), payload)

    def test_rejects_artifact_without_result_metadata(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as bundle:
            bundle.writestr("jenkins-task.log", "test output")

        with self.assertRaisesRegex(ValueError, "does not contain"):
            MODULE.read_result_metadata(output.getvalue())

    def test_job_name_matching_does_not_accept_similar_jobs(self):
        self.assertTrue(MODULE.job_name_matches("pr-test-mlu", "pr-test-mlu"))
        self.assertTrue(
            MODULE.job_name_matches("PR Test (MLU) / pr-test-mlu", "pr-test-mlu")
        )
        self.assertFalse(MODULE.job_name_matches("pr-test-mlu-extra", "pr-test-mlu"))

    def test_collection_joins_job_attempt_with_result_artifact(self):
        run = {
            "id": 42,
            "run_attempt": 1,
            "html_url": "https://github.example/runs/42",
            "event": "pull_request",
            "created_at": "2026-07-01T00:00:00Z",
        }
        jobs = [
            {
                "id": 99,
                "name": "pr-test-mlu",
                "status": "completed",
                "conclusion": "success",
                "created_at": "2026-07-01T00:00:00Z",
                "started_at": "2026-07-01T00:01:00Z",
                "completed_at": "2026-07-01T00:04:00Z",
                "runner_name": "mlu-pr-host-123",
                "html_url": "https://github.example/jobs/99",
            },
            {
                "id": 100,
                "name": "pr-test-mlu",
                "conclusion": "skipped",
            },
        ]
        artifacts = [
            {
                "id": 7,
                "name": "mlu-ci-result-pr-42-1",
                "created_at": "2026-07-01T00:04:00Z",
                "expired": False,
            }
        ]
        payload = {
            "status": "error",
            "failure_type": "infrastructure",
            "failure_stage": "jenkins_connect",
        }
        api = mock.Mock()
        api.request.return_value = result_archive(payload)
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = datetime(2026, 7, 8, tzinfo=timezone.utc)

        with mock.patch.object(
            MODULE, "list_workflow_runs", return_value=[run]
        ), mock.patch.object(
            MODULE, "list_attempt_jobs", return_value=jobs
        ), mock.patch.object(
            MODULE, "paginate", return_value=artifacts
        ):
            records, errors = MODULE.collect_records(
                api,
                "sgl-project/sglang",
                "pr-test-mlu.yml",
                "pr-test-mlu",
                start,
                end,
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["metadata"], payload)
        self.assertEqual(records[0]["queue_seconds"], 60)
        self.assertEqual(records[0]["runtime_seconds"], 180)
        api.request.assert_called_once_with(
            "/repos/sgl-project/sglang/actions/artifacts/7/zip",
            accept="application/vnd.github+json",
        )

    def test_summary_keeps_missing_metadata_visible(self):
        records = [
            record(metadata={"status": "success"}, run_id=1),
            record(
                metadata={
                    "status": "error",
                    "failure_type": "infrastructure",
                    "failure_stage": "jenkins_connect",
                },
                conclusion="success",
                run_id=2,
                queue_seconds=90,
                runtime_seconds=240,
            ),
            record(metadata=None, conclusion="failure", run_id=3),
            record(
                metadata=None,
                assigned=False,
                conclusion="cancelled",
                run_id=4,
            ),
        ]
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = datetime(2026, 7, 8, tzinfo=timezone.utc)

        summary = MODULE.build_summary(
            records, [], "sgl-project/sglang", "pr-test-mlu.yml", start, end
        )
        markdown = MODULE.render_markdown(summary)

        self.assertEqual(summary["counts"]["mlu_job_attempts"], 4)
        self.assertEqual(summary["counts"]["runner_assigned"], 3)
        self.assertEqual(summary["counts"]["runner_unassigned"], 1)
        self.assertEqual(summary["counts"]["job_completed"], 4)
        self.assertEqual(summary["counts"]["job_cancelled"], 1)
        self.assertEqual(summary["counts"]["unclaimed_completed"], 1)
        self.assertEqual(summary["counts"]["result_metadata"], 2)
        self.assertEqual(summary["classified_reliability"], 0.5)
        self.assertAlmostEqual(summary["metadata_coverage_for_assigned_jobs"], 2 / 3)
        self.assertTrue(markdown.startswith("<!-- mlu-ci-reliability-report -->\n"))
        self.assertIn("`infrastructure` | `jenkins_connect` | 1", markdown)
        self.assertIn("run 3 attempt 1", markdown)
        self.assertIn("runner not assigned", markdown)

    def test_report_outputs_are_json_serializable(self):
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = datetime(2026, 7, 8, tzinfo=timezone.utc)
        summary = MODULE.build_summary(
            [record(metadata={"status": "success"})],
            [],
            "sgl-project/sglang",
            "pr-test-mlu.yml",
            start,
            end,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(json.loads(path.read_text())["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
