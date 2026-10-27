import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
HANDLER_PATH = ROOT / "scripts/ci/utils/slash_command_handler.py"

requests_stub = types.ModuleType("requests")
github_stub = types.ModuleType("github")


class _Auth:
    @staticmethod
    def Token(token):
        return token


github_stub.Auth = _Auth
github_stub.Github = object
sys.modules.setdefault("requests", requests_stub)
sys.modules.setdefault("github", github_stub)
sys.modules.setdefault("runner_configs", types.ModuleType("runner_configs"))

spec = importlib.util.spec_from_file_location("slash_command_handler", HANDLER_PATH)
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)


class ResolveTagRunCiPermissionsTest(unittest.TestCase):
    def test_explicit_permission_skips_repository_lookup(self):
        repo = Mock()

        result = handler.resolve_tag_run_ci_permissions(
            repo, "configured-user", {"can_tag_run_ci_label": True}
        )

        self.assertTrue(result["can_tag_run_ci_label"])
        repo.get_collaborator_permission.assert_not_called()

    def test_write_or_higher_repository_role_is_authorized(self):
        for permission in ("write", "maintain", "admin"):
            with self.subTest(permission=permission):
                repo = Mock()
                repo.get_collaborator_permission.return_value = permission

                result = handler.resolve_tag_run_ci_permissions(
                    repo, "maintainer", None
                )

                self.assertTrue(result["can_tag_run_ci_label"])

    def test_lower_repository_role_is_not_authorized(self):
        for permission in ("none", "read", "triage"):
            with self.subTest(permission=permission):
                repo = Mock()
                repo.get_collaborator_permission.return_value = permission

                result = handler.resolve_tag_run_ci_permissions(
                    repo, "external-user", None
                )

                self.assertNotIn("can_tag_run_ci_label", result)

    def test_lookup_failure_fails_closed(self):
        repo = Mock()
        repo.get_collaborator_permission.side_effect = RuntimeError("API unavailable")

        result = handler.resolve_tag_run_ci_permissions(repo, "unknown-user", None)

        self.assertNotIn("can_tag_run_ci_label", result)


class RerunFailedCiTest(unittest.TestCase):
    def run_record(
        self,
        run_id=1,
        workflow_id=10,
        conclusion="failure",
        status="completed",
        path=".github/workflows/pr-test.yml",
        jobs=None,
    ):
        run = Mock()
        run.id = run_id
        run.workflow_id = workflow_id
        run.conclusion = conclusion
        run.status = status
        run.path = path
        run.name = path.rsplit("/", 1)[-1]
        run.jobs.return_value = []
        for job_name, job_conclusion in jobs or ():
            job = Mock()
            job.name = job_name
            job.conclusion = job_conclusion
            run.jobs.return_value.append(job)
        return run

    def invoke(self, runs):
        repo = Mock()
        repo.get_workflow_runs.return_value = runs
        pr = Mock()
        pr.head.sha = "head-sha"
        pr.get_files.return_value = []
        self.comment = Mock()
        result = handler.handle_rerun_failed_ci(
            repo,
            pr,
            self.comment,
            {"can_rerun_failed_ci": True},
        )
        repo.get_workflow_runs.assert_called_once_with(head_sha="head-sha")
        return result

    def test_rerun_covers_all_unsuccessful_conclusions(self):
        for conclusion in ("failure", "skipped", "cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                run = self.run_record(conclusion=conclusion)
                self.assertTrue(self.invoke([run]))
                if conclusion == "skipped":
                    run.rerun.assert_called_once_with()
                    run.rerun_failed_jobs.assert_not_called()
                else:
                    run.rerun_failed_jobs.assert_called_once_with()
                    run.rerun.assert_not_called()

    def test_only_latest_run_per_workflow_and_recoverable_states_are_used(self):
        older = self.run_record(run_id=1, conclusion="failure")
        newer = self.run_record(run_id=2, conclusion="success")
        active = self.run_record(run_id=3, workflow_id=11, status="in_progress")
        approval = self.run_record(
            run_id=4, workflow_id=12, conclusion="action_required"
        )

        self.assertFalse(self.invoke([older, newer, active, approval]))
        for run in (older, newer, active, approval):
            run.rerun.assert_not_called()
            run.rerun_failed_jobs.assert_not_called()

    def test_ignored_ci_events_do_not_hide_failed_pipeline(self):
        for ignored_name in ("CI Ignored Event", "PPU E2E Ignored Event"):
            with self.subTest(ignored_name=ignored_name):
                failed = self.run_record(
                    run_id=1,
                    path=".github/workflows/ci.yaml",
                    jobs=[("CI Result", "success")],
                )
                ignored = self.run_record(
                    run_id=2,
                    conclusion="success",
                    path=".github/workflows/ci.yaml",
                    jobs=[(ignored_name, "success")],
                )

                self.assertTrue(self.invoke([failed, ignored]))
                failed.rerun.assert_called_once_with()
                failed.rerun_failed_jobs.assert_not_called()
                ignored.rerun.assert_not_called()
                ignored.rerun_failed_jobs.assert_not_called()

    def test_noop_extra_event_does_not_hide_failed_pipeline(self):
        failed = self.run_record(
            run_id=1,
            path=".github/workflows/pr-test-extra.yml",
            jobs=[("check-changes / check-changes", "success")],
        )
        ignored = self.run_record(
            run_id=2,
            conclusion="success",
            path=".github/workflows/pr-test-extra.yml",
            jobs=[("check-changes / check-changes", "skipped")],
        )

        self.assertTrue(self.invoke([failed, ignored]))
        failed.rerun_failed_jobs.assert_called_once_with()
        failed.rerun.assert_not_called()
        ignored.rerun.assert_not_called()
        ignored.rerun_failed_jobs.assert_not_called()

    def test_unsuccessful_extra_run_with_skipped_selector_is_retried(self):
        for conclusion in ("cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                run = self.run_record(
                    conclusion=conclusion,
                    path=".github/workflows/pr-test-extra.yml",
                    jobs=[("check-changes / check-changes", "skipped")],
                )

                self.assertTrue(self.invoke([run]))
                run.rerun_failed_jobs.assert_called_once_with()
                run.rerun.assert_not_called()

    def test_newer_meaningful_ci_run_still_supersedes_failure(self):
        for status, conclusion in (("completed", "success"), ("in_progress", None)):
            with self.subTest(status=status, conclusion=conclusion):
                failed = self.run_record(
                    run_id=1,
                    path=".github/workflows/ci.yaml",
                    jobs=[("CI Result", "success")],
                )
                newer = self.run_record(
                    run_id=2,
                    status=status,
                    conclusion=conclusion,
                    path=".github/workflows/ci.yaml",
                    jobs=[("CI Result", "success")],
                )

                self.assertFalse(self.invoke([failed, newer]))
                failed.rerun.assert_not_called()
                failed.rerun_failed_jobs.assert_not_called()
                newer.rerun.assert_not_called()
                newer.rerun_failed_jobs.assert_not_called()

    def test_selective_rerun_rejection_falls_back_to_full_rerun(self):
        for rejection in ("exception", "false"):
            with self.subTest(rejection=rejection):
                run = self.run_record(conclusion="cancelled")
                if rejection == "exception":
                    run.rerun_failed_jobs.side_effect = RuntimeError("no failed jobs")
                else:
                    run.rerun_failed_jobs.return_value = False

                self.assertTrue(self.invoke([run]))
                run.rerun_failed_jobs.assert_called_once_with()
                run.rerun.assert_called_once_with()

    def test_rejected_rerun_does_not_report_success(self):
        for path in (".github/workflows/pr-test.yml", ".github/workflows/ci.yaml"):
            with self.subTest(path=path):
                run = self.run_record(conclusion="cancelled", path=path)
                run.rerun_failed_jobs.return_value = False
                run.rerun.return_value = False

                self.assertFalse(self.invoke([run]))
                run.rerun.assert_called_once_with()
                self.comment.create_reaction.assert_not_called()

    def test_ppu_pipeline_always_requires_a_full_rerun(self):
        for conclusion in ("failure", "skipped", "cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                run = self.run_record(
                    conclusion=conclusion, path=".github/workflows/ci.yaml"
                )
                self.assertTrue(self.invoke([run]))
                run.rerun.assert_called_once_with()
                run.rerun_failed_jobs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
