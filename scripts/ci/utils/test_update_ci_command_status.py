"""Offline regression tests for CI command replies and workflow-run linkage."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import update_ci_command_status as status  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def run(**changes):
    values = dict(
        id=42,
        name="PR Test (MLU)",
        html_url="https://github.com/o/r/actions/runs/42",
        run_attempt=2,
        status="completed",
        conclusion="success",
    )
    values.update(changes)
    return SimpleNamespace(**values)


class TestCICommandStatus(unittest.TestCase):
    def test_rerun_does_not_show_previous_success(self):
        self.assertIn("⏳", status.status_line(run(), 3))
        self.assertIn("queued", status.status_line(run(), 3))
        self.assertIn(
            "🔄", status.status_line(run(run_attempt=3, status="in_progress"), 3)
        )
        self.assertIn("✅", status.status_line(run(run_attempt=3), 3))

    def test_terminal_statuses_remain_distinct(self):
        for conclusion, icon in [
            ("success", "✅"),
            ("failure", "❌"),
            ("cancelled", "🚫"),
            ("timed_out", "❌"),
            ("skipped", "⏭️"),
        ]:
            with self.subTest(conclusion=conclusion):
                line = status.status_line(run(conclusion=conclusion), 2)
                self.assertIn(icon, line)
                self.assertIn(conclusion, line)

    def test_refresh_uses_live_attempt_and_preserves_other_lines(self):
        # A late event for attempt 1 must use the current attempt 3, including
        # when another workflow's event caused this PR-wide refresh.
        body = (
            "Details\n"
            + status.status_line(run(run_attempt=1, status="queued"), 1)
            + "\n"
        )
        comment = Mock(
            body=body, user=SimpleNamespace(login="github-actions[bot]", type="Bot")
        )
        forged = Mock(body=body, user=SimpleNamespace(login="contributor", type="User"))
        repo = Mock()
        repo.get_issue.return_value.get_comments.return_value = [comment, forged]
        repo.get_workflow_run.return_value = run(run_attempt=3, conclusion="cancelled")
        status.refresh_comments(repo, 12)
        updated = comment.edit.call_args.args[0]
        self.assertEqual(
            updated,
            "Details\n"
            + status.status_line(run(run_attempt=3, conclusion="cancelled"), 1)
            + "\n",
        )
        forged.edit.assert_not_called()
        comment.body = updated
        comment.edit.reset_mock()
        status.refresh_comments(repo, 12)
        comment.edit.assert_not_called()

    def test_all_runs_in_reply_are_reconciled(self):
        first, second = run(), run(id=43, conclusion="failure")
        comment = Mock(
            body="\n".join(
                status.status_line(r, 2)
                for r in [run(status="queued"), run(id=43, status="queued")]
            ),
            user=SimpleNamespace(login="github-actions[bot]", type="Bot"),
        )
        repo = Mock()
        repo.get_issue.return_value.get_comments.return_value = [comment]
        repo.get_workflow_run.side_effect = lambda n: {42: first, 43: second}[n]
        status.refresh_comments(repo, 12)
        self.assertEqual(
            comment.edit.call_args.args[0],
            "\n".join(status.status_line(r, 2) for r in [first, second]),
        )

    def test_deleted_run_does_not_block_current_results(self):
        from github import GithubException

        old = status.status_line(run(id=41, status="queued"), 1)
        current = status.status_line(run(status="queued"), 2)
        comment = Mock(
            body=old + "\n" + current,
            user=SimpleNamespace(login="github-actions[bot]", type="Bot"),
        )
        repo = Mock()
        repo.get_issue.return_value.get_comments.return_value = [comment]
        repo.get_workflow_run.side_effect = [GithubException(404), run()]
        status.refresh_comments(repo, 12)
        self.assertEqual(
            comment.edit.call_args.args[0], old + "\n" + status.status_line(run(), 2)
        )

    def test_fork_pr_lookup_and_non_pr_events(self):
        repo = Mock()
        repo.get_pulls.return_value = [SimpleNamespace(number=12)]
        event = {
            "workflow_run": {
                "event": "pull_request_target",
                "pull_requests": [],
                "head_repository": {"owner": {"login": "contributor"}},
                "head_branch": "feature",
            }
        }
        self.assertEqual(status.linked_pr_numbers(repo, event), [12])
        repo.get_pulls.assert_called_once_with(state="all", head="contributor:feature")
        event["workflow_run"]["event"] = "schedule"
        self.assertEqual(status.linked_pr_numbers(repo, event), [])
        event["workflow_run"].update(
            event="pull_request", pull_requests=[{"number": 13}]
        )
        self.assertEqual(status.linked_pr_numbers(repo, event), [13])

    def test_dispatch_requires_valid_pr(self):
        self.assertEqual(
            status.linked_pr_numbers(None, {"inputs": {"pr_number": "12"}}), [12]
        )
        for value in ["", "0", "12abc", "-1"]:
            with self.assertRaises(ValueError):
                status.linked_pr_numbers(None, {"inputs": {"pr_number": value}})

    def test_handler_marks_existing_and_rerun_attempts(self):
        import slash_command_handler as handler

        path = ".github/workflows/pr-test-mlu.yml"
        for state in ("queued", "completed"):
            with self.subTest(state=state):
                current = run(status=state)
                with (
                    patch.object(handler, "pr_workflow_runs", return_value=[current]),
                    patch.object(
                        handler, "latest_run_per_workflow", return_value=[current]
                    ),
                    patch.object(handler, "workflow_file", return_value=path),
                    patch.object(
                        handler, "_rerun_workflow_run", return_value=(True, "rerun")
                    ),
                ):
                    acted, message = handler._rerun_tier_workflows(None, None, [path])
                self.assertTrue(acted)
                minimum = 3 if state == "completed" else 2
                self.assertIn(f"<!--ci-run:42:{minimum}-->", message)
                self.assertIn("queued", message)

    def test_listener_covers_registered_command_workflows(self):
        import json

        tiers = json.loads((ROOT / "scripts/ci/opt_in_ci_tiers.json").read_text())
        names = {
            yaml.safe_load((ROOT / path).read_text())["name"]
            for tier in tiers.values()
            for path in tier["workflows"]
        }
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/ci-command-status.yml").read_text()
        )
        # PyYAML treats the unquoted YAML 1.1 key 'on' as True.
        self.assertEqual(set(workflow[True]["workflow_run"]["workflows"]), names)


if __name__ == "__main__":
    unittest.main()
