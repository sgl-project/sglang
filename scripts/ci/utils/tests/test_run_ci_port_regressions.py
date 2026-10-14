"""Guard authorization integration without reverting newer rerun behavior."""

import re
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_run_ci_permissions import ROOT, handler


class RunCiPortRegressionTest(unittest.TestCase):
    def invoke_main(self, command, permission="write", author="other"):
        repo = Mock()
        repo.get_collaborator_permission.return_value = permission
        pr = repo.get_pull.return_value
        pr.user.login = author
        env = {
            "GITHUB_TOKEN": "test-token",
            "REPO_FULL_NAME": "owner/repo",
            "PR_NUMBER": "1",
            "COMMENT_ID": "2",
            "COMMENT_BODY": command,
            "USER_LOGIN": "commenter",
        }
        with (
            patch.object(handler, "get_env_var", side_effect=env.__getitem__),
            patch.object(handler, "load_permissions", return_value=None),
            patch.object(handler, "Github") as github,
            patch.object(handler.time, "sleep"),
            patch.object(
                handler, "handle_rerun_failed_ci", return_value=False
            ) as rerun,
            patch.object(handler, "handle_rerun_test") as rerun_test,
        ):
            github.return_value.get_repo.return_value = repo
            handler.main()
        return SimpleNamespace(repo=repo, pr=pr, rerun=rerun, rerun_test=rerun_test)

    def test_both_tag_commands_resolve_repository_permission(self):
        for command in ("/tag-run-ci-label", "/tag-and-rerun-ci"):
            with self.subTest(command=command):
                result = self.invoke_main(command)
                result.repo.get_collaborator_permission.assert_called_once_with(
                    "commenter"
                )
                result.pr.add_to_labels.assert_called_once_with("run-ci")
                if command == "/tag-and-rerun-ci":
                    permissions = result.rerun.call_args.args[3]
                    self.assertTrue(permissions["can_tag_run_ci_label"])
                    self.assertNotIn("can_rerun_failed_ci", permissions)

    def test_authorship_alone_does_not_authorize_labels_or_individual_tests(self):
        result = self.invoke_main(
            "/tag-and-rerun-ci", permission="read", author="commenter"
        )
        result.pr.add_to_labels.assert_not_called()
        permissions = result.rerun.call_args.args[3]
        self.assertTrue(permissions["can_rerun_failed_ci"])
        self.assertNotIn("can_tag_run_ci_label", permissions)
        self.assertNotIn("can_rerun_test", permissions)

    def test_unlisted_user_still_reaches_existing_rerun_test_gate(self):
        result = self.invoke_main("/rerun-test test_example.py")
        result.repo.get_collaborator_permission.assert_not_called()
        result.rerun_test.assert_called_once()
        self.assertEqual(result.rerun_test.call_args.args[3], {})
        self.assertEqual(result.rerun_test.call_args.args[4], ["test_example.py"])

    def test_repository_permission_only_adds_tag_capability_without_mutation(self):
        repo = Mock()
        repo.get_collaborator_permission.return_value = "maintain"
        configured = {"can_rerun_failed_ci": False, "cooldown_interval_minutes": 60}
        result = handler.resolve_tag_run_ci_permissions(repo, "maintainer", configured)
        self.assertEqual(result, {**configured, "can_tag_run_ci_label": True})
        self.assertNotIn("can_tag_run_ci_label", configured)

    def test_active_newer_run_prevents_rerunning_superseded_failure(self):
        repo, pr, comment = Mock(), Mock(), Mock()
        pr.get_files.return_value = []
        older = Mock(id=1, workflow_id=10, status="completed", conclusion="failure")
        newer = Mock(id=2, workflow_id=10, status="in_progress", conclusion=None)
        repo.get_workflow_runs.return_value = [newer, older]
        self.assertFalse(
            handler.handle_rerun_failed_ci(
                repo, pr, comment, {"can_rerun_failed_ci": True}
            )
        )
        older.rerun.assert_not_called()
        older.rerun_failed_jobs.assert_not_called()
        newer.rerun.assert_not_called()
        newer.rerun_failed_jobs.assert_not_called()

    def test_cancelled_and_timed_out_runs_keep_failed_jobs_fallback(self):
        for conclusion in ("cancelled", "timed_out"):
            with self.subTest(conclusion=conclusion):
                repo, pr, comment = Mock(), Mock(), Mock()
                pr.get_files.return_value = []
                run = Mock(
                    id=1, workflow_id=10, status="completed", conclusion=conclusion
                )
                run.rerun_failed_jobs.side_effect = RuntimeError("no failed jobs")
                repo.get_workflow_runs.return_value = [run]
                self.assertTrue(
                    handler.handle_rerun_failed_ci(
                        repo, pr, comment, {"can_rerun_failed_ci": True}
                    )
                )
                run.rerun_failed_jobs.assert_called_once()
                run.rerun.assert_called_once()


class PpuWorkflowPortRegressionTest(unittest.TestCase):
    def job(self, workflow, job):
        text = (ROOT / ".github/workflows" / workflow).read_text()
        match = re.search(
            rf"^  {re.escape(job)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", text, re.M | re.S
        )
        self.assertIsNotNone(match, job)
        return match.group(1)

    def test_full_ci_depends_only_on_registration(self):
        ci = self.job("ci.yaml", "ci")
        self.assertRegex(ci, r"(?m)^    needs: registration-validation$")
        self.assertNotIn("human-review", ci)
        self.assertIn("github.event_name == 'pull_request'", ci)
        human_review = self.job("ci.yaml", "human-review")
        self.assertIn("needs: [registration-validation, ai-code-review]", human_review)

    def test_ppu_stages_remain_serial_kubernetes_jobs(self):
        for stage, devices, dependency in (
            ("a", 1, "check-changes"),
            ("b", 1, "check-changes, stage-a-test-1-gpu-ppu"),
            ("c", 2, "check-changes, stage-b-test-1-gpu-ppu"),
        ):
            suite = f"stage-{stage}-test-{devices}-gpu-ppu"
            with self.subTest(stage=stage):
                job = self.job("pr-test-ppu.yml", suite)
                self.assertIn(f"needs: [{dependency}]", job)
                self.assertIn("runs-on: k8s-runner-group-cpu-flytiger", job)
                self.assertIn("uses: flytiger-eco/ppu-distributed-action@main", job)
                self.assertIn(f"nproc_per_node: {devices}", job)
                self.assertIn(f"run_suite.py --hw ppu --suite {suite}", job)
                self.assertEqual(job.count("ref: ${{ inputs.ref || github.sha }}"), 2)
                if stage != "a":
                    self.assertIn("!failure()", job)
                    self.assertIn("!cancelled()", job)

    def test_workflow_uses_sdk_220_and_describes_actual_coverage(self):
        workflow = (ROOT / ".github/workflows/pr-test-ppu.yml").read_text()
        self.assertIn("PPU_SDK_VERSION: 2.2.0", workflow)
        self.assertIn("sdk2.2.0-pytorch2.11.0", workflow)
        self.assertNotIn("test_anthropic_server.py", workflow)
        self.assertNotIn("27 no-runtime", workflow)
        self.assertIn("No tests are currently registered to stage-c", workflow)
        self.assertNotRegex(workflow, r"(?m)^  pull_request:")


if __name__ == "__main__":
    unittest.main()
