"""Unit tests for run_ci_approval.py, with fake GitHub objects and no network."""

import os
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ci_approval as rca  # noqa: E402
from github import GithubException  # noqa: E402

PR_TEST = ".github/workflows/pr-test.yml"
HEAD = "a" * 40
GATE = "call-gate / pr-gate"
PR = NS(number=7, head=NS(sha=HEAD, ref="feature", repo=NS(id=1)))


def _at(hour):
    return datetime(2026, 11, 23, hour, tzinfo=timezone.utc)


class FakeRepo:
    full_name = "sgl-project/sglang"

    def __init__(self, permissions=None, statuses=(), runs=()):
        self.permissions = permissions or {}
        self.statuses = list(statuses)
        self.runs = list(runs)

    def get_collaborator_permission(self, login):
        permission = self.permissions.get(login, 404)
        if isinstance(permission, int):
            raise GithubException(permission, None, None)
        return permission

    def get_commit(self, sha):
        return NS(get_statuses=lambda: self.statuses)

    def get_workflow_runs(self, head_sha):
        return list(self.runs)

    def get_pull(self, number):
        return PR


class FakeRun:
    """A pr-test.yml run of PR whose (status, jobs) advance one state per jobs() call."""

    def __init__(self, states=(("completed", ()),), created_at=None, run_id=1):
        self.id = run_id
        self.workflow_id = self.path = PR_TEST
        self.name = os.path.basename(PR_TEST)
        self.html_url = f"https://example.invalid/runs/{run_id}"
        self.event = "pull_request_target"
        self.head_branch = "feature"
        self.head_repository = NS(id=1)
        self.created_at = created_at
        self.states = list(states)
        self.status = self.states[0][0]
        self.reruns = 0

    def jobs(self):
        self.status, jobs = self.states[0]
        if len(self.states) > 1:
            self.states.pop(0)
        return list(jobs)

    def rerun(self):
        self.reruns += 1
        return True


def _gate(status="completed", skipped=False):
    steps = [NS(name=rca.GATE_SKIP_STEP, conclusion="success")] if skipped else []
    return NS(name=GATE, status=status, conclusion="success", steps=steps)


def _status(state="success", login=rca.APPROVAL_CREATOR):
    creator = NS(login=login) if login else None
    return NS(context=rca.APPROVAL_CONTEXT, state=state, creator=creator)


class TestCanApprove(unittest.TestCase):
    def test_write_access_or_ci_permissions_entry_approves(self):
        tag = {"can_tag_run_ci_label": True}
        cases = [
            *(((p, None), True) for p in rca.APPROVER_REPO_PERMISSIONS),
            (("read", tag), True),
            ((404, tag), True),
            (("read", {"can_rerun_failed_ci": True}), False),
            ((404, None), False),
        ]
        for (permission, perms), expected in cases:
            with self.subTest(permission=permission, perms=perms):
                repo = FakeRepo(permissions={"u": permission})
                self.assertEqual(rca.can_approve(repo, "u", perms)[0], expected)


class TestIsApproved(unittest.TestCase):
    def test_label_plus_approver_author_or_bot_status(self):
        cases = [
            ("maint", [], {"run-ci"}, True),
            ("maint", [], set(), False),
            ("listed", [], {"run-ci"}, True),
            ("ext", [_status()], {"run-ci"}, True),
            ("ext", [], {"run-ci"}, False),
            ("ext", [_status(login="someone")], {"run-ci"}, False),
            ("ext", [_status(login=None)], {"run-ci"}, False),
            ("ext", [_status(state="failure")], {"run-ci"}, False),
            ("ext", [_status()], {"run-ci-nv-extra"}, False),
        ]
        listed = {"listed": {"can_tag_run_ci_label": True}}
        with mock.patch.object(rca, "load_permissions", side_effect=listed.get):
            for author, statuses, labels, expected in cases:
                with self.subTest(author=author, statuses=statuses, labels=labels):
                    repo = FakeRepo(
                        permissions={"maint": "write", "listed": "read", "ext": 404},
                        statuses=statuses,
                    )
                    self.assertEqual(
                        rca.is_approved(repo, HEAD, labels, author), expected
                    )


class TestHeadSeenBy(unittest.TestCase):
    def test_head_must_be_pushed_before_the_comment(self):
        cases = {
            "pushed before": ([_at(11)], True),
            "pushed after": ([_at(13)], False),
            "re-pushed after": ([_at(10), _at(13)], False),
            "no run yet": ([], False),
        }
        for case, (created, expected) in cases.items():
            with self.subTest(case=case):
                runs = [FakeRun(created_at=t, run_id=i) for i, t in enumerate(created)]
                repo = FakeRepo(runs=runs)
                self.assertEqual(rca.head_seen_by(repo, PR, HEAD, _at(12)), expected)


class TestRerunGatedRuns(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_skipped_gate_is_rerun_once_completed(self):
        run = FakeRun(
            states=[
                ("in_progress", [_gate(status="in_progress")]),
                ("in_progress", [_gate(skipped=True)]),
                ("completed", [_gate(skipped=True)]),
            ]
        )
        [(acted, _)] = rca.rerun_gated_runs(FakeRepo(runs=[run]), PR, HEAD, [PR_TEST])
        self.assertTrue(acted)
        self.assertEqual(run.reruns, 1)

    def test_passed_gate_is_not_rerun(self):
        run = FakeRun(states=[("in_progress", [_gate()])])
        [(acted, _)] = rca.rerun_gated_runs(FakeRepo(runs=[run]), PR, HEAD, [PR_TEST])
        self.assertFalse(acted)
        self.assertEqual(run.reruns, 0)


if __name__ == "__main__":
    unittest.main()
