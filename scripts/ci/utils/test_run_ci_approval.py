"""Unit tests for run_ci_approval.py, with fake GitHub objects and no network."""

import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ci_approval as rca  # noqa: E402
from github import GithubException  # noqa: E402

PR_TEST = ".github/workflows/pr-test.yml"
PR_TEST_EXTRA = ".github/workflows/pr-test-extra.yml"
HEAD = "a" * 40
# 2026-11-23T00:00:00Z; 90 days earlier is 2026-08-25.
NOW = 1795392000


def _time(hour):
    return datetime(2026, 11, 23, hour, tzinfo=timezone.utc)


class FakeRepo:
    def __init__(self, permissions=None, merged_prs=0, statuses=(), runs=(), head=HEAD):
        self.full_name = "sgl-project/sglang"
        self.permissions = permissions or {}
        self.merged_prs = merged_prs
        self.statuses = list(statuses)
        self.runs = list(runs)
        self.head = head
        self.queries = []
        self.requester = SimpleNamespace(requestJsonAndCheck=self._request)

    def _request(self, verb, url, parameters):
        self.queries.append(parameters["q"])
        return {}, {"total_count": self.merged_prs}

    def get_collaborator_permission(self, login):
        permission = self.permissions.get(login, 404)
        if isinstance(permission, int):
            raise GithubException(permission, None, None)
        return permission

    def get_commit(self, sha):
        return SimpleNamespace(get_statuses=lambda: self.statuses)

    def get_workflow_runs(self, head_sha):
        return list(self.runs)

    def get_pull(self, number):
        return SimpleNamespace(head=SimpleNamespace(sha=self.head))


def _pr(repo_id=1, ref="feature"):
    return SimpleNamespace(
        number=7,
        head=SimpleNamespace(sha=HEAD, ref=ref, repo=SimpleNamespace(id=repo_id)),
    )


def _status(context=rca.APPROVAL_CONTEXT, state="success", login=rca.APPROVAL_CREATOR):
    creator = SimpleNamespace(login=login) if login else None
    return SimpleNamespace(context=context, state=state, creator=creator)


def _job(name, status="completed", conclusion="success", skipped=False):
    steps = (
        [SimpleNamespace(name=rca.GATE_SKIP_STEP, conclusion="success")]
        if skipped
        else []
    )
    return SimpleNamespace(name=name, status=status, conclusion=conclusion, steps=steps)


class FakeRun:
    """A workflow run whose (run status, jobs) advance one state per jobs() call."""

    def __init__(
        self,
        path=PR_TEST,
        states=None,
        run_id=1,
        created_at=None,
        event="pull_request_target",
        head_branch="feature",
        repo_id=1,
        rerun_accepted=True,
    ):
        self.id = run_id
        self.workflow_id = path
        self.path = path
        self.name = os.path.basename(path)
        self.html_url = f"https://example.invalid/runs/{run_id}"
        self.event = event
        self.head_branch = head_branch
        self.head_repository = SimpleNamespace(id=repo_id)
        self.created_at = created_at
        self.states = list(states or [("completed", [])])
        self.status = self.states[0][0]
        self.rerun_accepted = rerun_accepted
        self.reruns = 0

    def jobs(self):
        self.status, jobs = self.states[0]
        if len(self.states) > 1:
            self.states.pop(0)
        return jobs

    def rerun(self):
        self.reruns += 1
        return self.rerun_accepted


GATE = "call-gate / pr-gate"


class TestCanApprove(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("time.time", return_value=NOW)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_write_access_approves_without_activity(self):
        for permission in rca.APPROVER_REPO_PERMISSIONS:
            with self.subTest(permission=permission):
                repo = FakeRepo(permissions={"u": permission})
                self.assertEqual(rca.can_approve(repo, "u", None), (True, ""))
                self.assertEqual(repo.queries, [])

    def test_no_write_access_and_no_permission_entry(self):
        for permission in ("read", "triage", 403, 404):
            with self.subTest(permission=permission):
                repo = FakeRepo(permissions={"u": permission}, merged_prs=5)
                ok, reason = rca.can_approve(repo, "u", {"can_rerun_failed_ci": True})
                self.assertFalse(ok)
                self.assertIn("neither write access nor can_tag_run_ci_label", reason)
                self.assertEqual(repo.queries, [])

    def test_server_error_is_raised(self):
        repo = FakeRepo(permissions={"u": 500})
        with self.assertRaises(GithubException):
            rca.can_approve(repo, "u", {"can_tag_run_ci_label": True})

    def test_ci_permissions_user_needs_a_recent_merged_pr(self):
        perms = {"can_tag_run_ci_label": True}
        repo = FakeRepo(permissions={"u": "read"}, merged_prs=1)
        self.assertEqual(rca.can_approve(repo, "u", perms), (True, ""))
        self.assertEqual(
            repo.queries,
            ["repo:sgl-project/sglang is:pr is:merged author:u merged:>=2026-08-25"],
        )

        repo = FakeRepo(permissions={"u": 404}, merged_prs=0)
        ok, reason = rca.can_approve(repo, "u", perms)
        self.assertFalse(ok)
        self.assertIn("no PR merged into sgl-project/sglang since 2026-08-25", reason)


class TestIsApproved(unittest.TestCase):
    def test_bot_success_status_with_label(self):
        repo = FakeRepo(statuses=[_status(context="other"), _status()])
        self.assertTrue(rca.is_approved(repo, HEAD, {"run-ci"}))

    def test_rejected_statuses(self):
        for status in (
            _status(login="someone"),
            _status(login=None),
            _status(state="failure"),
            _status(context="ci/other"),
        ):
            with self.subTest(status=status):
                repo = FakeRepo(statuses=[status])
                self.assertFalse(rca.is_approved(repo, HEAD, {"run-ci"}))

    def test_removed_label_revokes(self):
        repo = FakeRepo(statuses=[_status()])
        self.assertFalse(rca.is_approved(repo, HEAD, {"run-ci-nv-extra"}))


class TestApprovalWorkflowsFor(unittest.TestCase):
    def test_extra_needs_its_label(self):
        self.assertEqual(rca.approval_workflows_for({"run-ci"}), [PR_TEST])
        self.assertEqual(
            rca.approval_workflows_for({"run-ci", "run-ci-nv-extra"}),
            [PR_TEST, PR_TEST_EXTRA],
        )


class TestPrWorkflowRuns(unittest.TestCase):
    def test_other_fork_or_branch_is_dropped(self):
        mine = FakeRun(run_id=1)
        other_fork = FakeRun(run_id=2, repo_id=2)
        other_branch = FakeRun(run_id=3, head_branch="main")
        push = FakeRun(run_id=4, event="push", head_branch="main", repo_id=2)
        repo = FakeRepo(runs=[mine, other_fork, other_branch, push])
        self.assertEqual(list(rca.pr_workflow_runs(repo, _pr(), HEAD)), [mine, push])


class TestHeadSeenBy(unittest.TestCase):
    def test_push_clock(self):
        comment_at = _time(12)
        cases = {
            "pushed before the comment": ([_time(11)], True),
            "pushed after the comment": ([_time(13)], False),
            "re-pushed after the comment": ([_time(10), _time(13)], False),
            "no run yet": ([], False),
        }
        for case, (created, expected) in cases.items():
            with self.subTest(case=case):
                runs = [FakeRun(run_id=i, created_at=t) for i, t in enumerate(created)]
                # Another workflow's run says nothing about the push.
                runs.append(FakeRun(path=PR_TEST_EXTRA, run_id=99, created_at=_time(1)))
                repo = FakeRepo(runs=runs)
                self.assertEqual(
                    rca.head_seen_by(repo, _pr(), HEAD, comment_at), expected
                )


class TestRerunGatedRuns(unittest.TestCase):
    def setUp(self):
        self.sleep = mock.patch("time.sleep").start()
        self.addCleanup(mock.patch.stopall)

    def _rerun(self, repo, paths=(PR_TEST,)):
        return rca.rerun_gated_runs(repo, _pr(), HEAD, list(paths))

    def test_skipped_gate_is_rerun_once_completed(self):
        run = FakeRun(
            states=[
                ("in_progress", [_job(GATE, status="in_progress")]),
                ("in_progress", [_job(GATE, skipped=True)]),
                ("completed", [_job(GATE, skipped=True)]),
            ]
        )
        [(acted, message)] = self._rerun(FakeRepo(runs=[run]))
        self.assertTrue(acted)
        self.assertIn("Re-running", message)
        self.assertEqual(run.reruns, 1)
        self.assertEqual(self.sleep.call_count, 2)

    def test_passed_gate_needs_nothing(self):
        run = FakeRun(states=[("in_progress", [_job(GATE)])])
        [(acted, message)] = self._rerun(FakeRepo(runs=[run]))
        self.assertFalse(acted)
        self.assertIn("already passed its gate", message)
        self.assertEqual(run.reruns, 0)

    def test_failed_gate_is_not_rerun(self):
        run = FakeRun(states=[("completed", [_job(GATE, conclusion="failure")])])
        [(acted, message)] = self._rerun(FakeRepo(runs=[run]))
        self.assertFalse(acted)
        self.assertIn("gate failed", message)
        self.assertEqual(run.reruns, 0)

    def test_no_run(self):
        [(acted, message)] = self._rerun(FakeRepo())
        self.assertFalse(acted)
        self.assertIn("No `pr-test.yml` run exists", message)
        self.sleep.assert_not_called()

    def test_moved_head_is_not_rerun(self):
        run = FakeRun(states=[("completed", [_job(GATE, skipped=True)])])
        [(acted, message)] = self._rerun(FakeRepo(runs=[run], head="b" * 40))
        self.assertFalse(acted)
        self.assertIn("head moved", message)
        self.assertEqual(run.reruns, 0)

    def test_refused_rerun_is_reported(self):
        run = FakeRun(
            states=[("completed", [_job(GATE, skipped=True)])], rerun_accepted=False
        )
        [(acted, message)] = self._rerun(FakeRepo(runs=[run]))
        self.assertFalse(acted)
        self.assertIn("Failed to re-run", message)

    def test_wait_is_bounded(self):
        mock.patch(
            "time.monotonic",
            side_effect=[0, rca.GATE_MAX_WAIT_SECONDS - 1, rca.GATE_MAX_WAIT_SECONDS],
        ).start()
        run = FakeRun(states=[("in_progress", [_job(GATE, status="in_progress")])])
        [(acted, message)] = self._rerun(FakeRepo(runs=[run]))
        self.assertFalse(acted)
        self.assertIn("Timed out", message)
        self.assertEqual(self.sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
