"""CPU-only contracts for persistent PPU commands and deferred full reruns."""

import copy
import io
import json
import os
import subprocess
import sys
import unittest
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci/ppu"))
import pr_e2e_commands as commands

FIXTURE_NOW = datetime.now(timezone.utc).replace(microsecond=0)


class MemoryGitHub(commands.GitHub):
    """In-memory API, including genuine ZIP plan artifacts and receipt parsing."""

    def __init__(self, pr=None):
        self.repository = "example/sglang"
        self.pr = pr or {
            "number": 123,
            "head": {
                "sha": "a" * 40,
                "ref": "test/branch",
                "repo": {"full_name": self.repository},
            },
            "base": {
                "sha": "b" * 40,
                "ref": "base",
                "repo": {"full_name": self.repository},
            },
            "labels": [{"name": "run-ci"}],
            "state": "open",
            "draft": False,
        }
        self.permission = "write"
        self.notes = []
        self.calls = []
        self.runs = []
        self.plans = {}
        self.fail_rerun = False
        now = FIXTURE_NOW
        self.now = now.isoformat().replace("+00:00", "Z")
        self.handler = {
            "path": commands.HANDLER,
            "event": "issue_comment",
            "head_branch": "main",
            "head_repository": {"full_name": self.repository},
            "run_attempt": 1,
            "status": "in_progress",
            "run_started_at": (now - timedelta(minutes=1)).isoformat(),
        }

    def metadata(self):
        return {"default_branch": "main"}

    def handler_run(self, run_id, attempt):
        return self.handler

    def get(self, path):
        if path == f"pulls/{self.pr['number']}":
            return copy.deepcopy(self.pr)
        if path.startswith("collaborators/"):
            return {"permission": self.permission}
        if path.startswith("actions/runs/"):
            return copy.deepcopy(
                next(r for r in self.runs if r["id"] == int(path.split("/")[2]))
            )
        raise AssertionError(path)

    def comments(self, number):
        assert number == self.pr["number"]
        return copy.deepcopy(self.notes)

    def post(self, path, data):
        self.calls.append((path, copy.deepcopy(data)))
        if path.endswith("/comments"):
            note = {
                "id": max((c["id"] for c in self.notes), default=0) + 1,
                "body": data["body"],
                "user": {"id": 999, "login": "github-actions[bot]", "type": "Bot"},
                "created_at": self.now,
                "updated_at": self.now,
            }
            self.notes.append(note)
            return copy.deepcopy(note)
        if path.endswith("/rerun"):
            if self.fail_rerun:
                raise TimeoutError("Simulated ambiguous submission failure")
            run = next(r for r in self.runs if r["id"] == int(path.split("/")[2]))
            run["status"] = "queued"
            run["run_attempt"] += 1
            return None
        raise AssertionError(path)

    def pages(self, path, key=None):
        if path.startswith("actions/workflows/"):
            return copy.deepcopy(self.runs)
        if path.startswith("actions/runs/") and "/artifacts?" in path:
            run_id = int(path.split("/")[2])
            name = path.split("name=")[1]
            attempt = int(name.rsplit("-", 1)[1])
            if (run_id, attempt) not in self.plans:
                return []
            return [{"name": name, "expired": False, "id": run_id * 100 + attempt}]
        if path.startswith("commits/"):
            return [copy.deepcopy(self.pr)]
        raise AssertionError(path)

    def request(self, path, data=None, binary=False):
        assert binary
        artifact = int(path.split("/")[2])
        plan = self.plans[(artifact // 100, artifact % 100)]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("selection.json", json.dumps(plan))
        return buffer.getvalue()

    def add_command(self, body="/run-ppu-e2e test-bf16", number=10):
        comment = {
            "id": number,
            "body": body,
            "user": {"id": 2, "login": "maintainer", "type": "User"},
            "created_at": self.now,
            "updated_at": self.now,
        }
        self.notes.append(comment)
        return comment

    def accept(self, tokens=("test-bf16",), number=10):
        body = "/run-ppu-e2e " + " ".join(tokens) if tokens else "/clear-ppu-e2e"
        source = self.add_command(body, number)
        request = commands.command_request(self, source)
        with mock.patch.dict(
            os.environ, {"GITHUB_RUN_ID": "700", "GITHUB_RUN_ATTEMPT": "1"}
        ):
            commands.write_receipt(self, self.pr["number"], "request", request=request)
        return commands.resolve_request(self, self.pr["number"])

    def add_run(self, status="completed", run_id=42, request=None):
        run = {
            "id": run_id,
            "run_attempt": 1,
            "path": commands.PIPELINE,
            "event": "pull_request",
            "head_sha": self.pr["head"]["sha"],
            "head_repository": self.pr["head"]["repo"],
            "head_branch": self.pr["head"]["ref"],
            "pull_requests": [{"number": self.pr["number"]}],
            "status": status,
            "conclusion": "success",
            "created_at": self.now,
        }
        self.runs.append(run)
        self.plans[(run_id, 1)] = {
            "pr": self.pr["number"],
            "head_sha": self.pr["head"]["sha"],
            "base_sha": self.pr["base"]["sha"],
            "run_id": str(run_id),
            "attempt": "1",
            "request": request,
            "labels": [label["name"] for label in self.pr["labels"]],
            "state": self.pr["state"],
            "draft": self.pr["draft"],
        }
        return run

    def submissions(self):
        return [path for path, _ in self.calls if path.endswith("/rerun")]


class CommandStateTest(unittest.TestCase):
    def test_exact_command_grammar(self):
        self.assertEqual(
            commands.parse_command(" /run-ppu-e2e test-qwen test-w4a8 test-qwen "),
            ["test-qwen", "test-w4a8"],
        )
        self.assertEqual(commands.parse_command("/clear-ppu-e2e"), [])
        self.assertIsNone(commands.parse_command("An ordinary comment"))
        for body in (
            "/run-ppu-e2e",
            "/run-ppu-e2e test-unknown",
            "/run-ppu-e2e test-perf",
            "/run-ppu-e2e test-bf16; echo unsafe",
            "/run-ppu-e2e-extra test-bf16",
            "/run-ppu-e2e TEST-BF16",
            "/clear-ppu-e2e test-qwen",
            "> /run-ppu-e2e test-bf16",
            "`/run-ppu-e2e test-bf16`",
            "please /run-ppu-e2e test-bf16",
            "/run-ppu-e2e test-bf16\n/clear-ppu-e2e",
        ):
            with self.subTest(body=body), self.assertRaises(ValueError):
                commands.parse_command(body)

    def test_replace_clear_and_persist_across_revisions(self):
        api = MemoryGitHub()
        first = api.accept()
        self.assertEqual(commands.resolve_request(api, 123), first)
        second = api.accept(("test-qwen", "test-w4a8"), 20)
        api.pr["head"]["sha"] = "d" * 40
        self.assertEqual(commands.resolve_request(api, 123), second)
        cleared = api.accept((), 30)
        self.assertEqual(commands.resolve_request(api, 123), cleared)
        self.assertEqual(cleared["selectors"], [])
        self.assertEqual(commands.resolve_request(api, 123, first), first)

    def test_out_of_order_receipts_choose_original_comment_order(self):
        api = MemoryGitHub()
        latest = api.accept(("test-qwen",), 20)
        api.accept(("test-bf16",), 10)
        self.assertEqual(commands.resolve_request(api, 123), latest)

    def test_deleted_edited_or_revoked_request_never_restores_older_selection(self):
        for mutation in ("deleted", "edited", "revoked"):
            api = MemoryGitHub()
            api.accept()
            latest = api.accept(("test-qwen",), 20)
            if mutation == "deleted":
                api.notes = [n for n in api.notes if n["id"] != latest["comment_id"]]
            elif mutation == "edited":
                next(n for n in api.notes if n["id"] == latest["comment_id"])[
                    "body"
                ] = "/clear-ppu-e2e"
            else:
                api.permission = "read"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                commands.resolve_request(api, 123)

    def test_permission_and_bot_rejection(self):
        for permission in ("none", "read", "triage"):
            api = MemoryGitHub()
            api.permission = permission
            with self.subTest(permission=permission), self.assertRaises(ValueError):
                commands.command_request(api, api.add_command())
        for permission in ("write", "maintain", "admin"):
            api = MemoryGitHub()
            api.permission = permission
            self.assertTrue(commands.command_request(api, api.add_command()))
        api = MemoryGitHub()
        source = api.add_command()
        source["user"]["type"] = "Bot"
        with self.assertRaises(ValueError):
            commands.command_request(api, source)
        source["user"]["type"] = "User"
        source["updated_at"] = "edited"
        with self.assertRaises(ValueError):
            commands.command_request(api, source)

    def test_receipt_trust_boundary(self):
        for field, value in (
            ("path", ".github/workflows/unrelated.yml"),
            ("event", "workflow_dispatch"),
            ("head_branch", "attacker"),
            ("head_repository", {"full_name": "attacker/fork"}),
            ("run_attempt", 2),
            ("run_started_at", "2099-01-01T00:00:00Z"),
        ):
            api = MemoryGitHub()
            api.accept()
            api.handler[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                commands.resolve_request(api, 123)
        api = MemoryGitHub()
        api.accept()
        receipt = api.notes[-1]
        receipt["user"] = {"type": "User", "login": "attacker"}
        self.assertIsNone(commands.resolve_request(api, 123))
        receipt["user"] = {"type": "Bot", "login": "github-actions[bot]"}
        receipt["updated_at"] = "edited"
        with self.assertRaises(ValueError):
            commands.resolve_request(api, 123)

    def test_completed_handler_time_bound_and_conflicting_receipts_fail_closed(self):
        api = MemoryGitHub()
        api.accept()
        api.handler.update(status="completed", updated_at=api.now)
        self.assertIsNotNone(commands.resolve_request(api, 123))
        api.handler["updated_at"] = "2000-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "postdates"):
            commands.resolve_request(api, 123)
        api.handler["updated_at"] = api.now
        forged = copy.deepcopy(api.notes[-1])
        forged["id"] += 1
        value = json.loads(forged["body"][len(commands.MARKER) :])
        value["request"]["selectors"] = ["test-qwen"]
        forged["body"] = commands.MARKER + json.dumps(value)
        api.notes.append(forged)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            commands.resolve_request(api, 123)

    def test_new_valid_command_recovers_deleted_source_without_restoring_history(self):
        api = MemoryGitHub()
        api.accept()
        old = api.accept(("test-qwen",), 20)
        api.notes = [note for note in api.notes if note["id"] != old["comment_id"]]
        with self.assertRaises(ValueError):
            commands.resolve_request(api, 123)
        new = api.accept((), 30)
        self.assertEqual(commands.resolve_request(api, 123), new)

    def test_permission_api_failure_is_not_an_empty_selection(self):
        api = MemoryGitHub()
        api.accept()
        with mock.patch.object(
            api, "get", side_effect=RuntimeError("offline")
        ), self.assertRaises(RuntimeError):
            commands.resolve_request(api, 123)

    def test_pagination_and_fail_closed_bound(self):
        api = object.__new__(commands.GitHub)
        with mock.patch.object(
            api, "get", side_effect=[[{}] * 100, [{"id": 101}]]
        ) as get:
            self.assertEqual(len(api.comments(123)), 101)
            self.assertIn("page=2", get.call_args.args[0])
        with mock.patch.object(api, "get", return_value=[{}] * 100), self.assertRaises(
            ValueError
        ):
            api.comments(123)

    def test_redirect_strips_token_on_cross_host_and_rejects_http(self):
        request = urllib.request.Request(
            "https://api.github.com/example",
            headers={"Authorization": "Bearer fixture"},
        )
        redirected = commands.SafeRedirect().redirect_request(
            request, None, 302, "", {}, "https://storage.example/artifact"
        )
        self.assertIsNone(redirected.get_header("Authorization"))
        with self.assertRaises(ValueError):
            commands.SafeRedirect().redirect_request(
                request, None, 302, "", {}, "http://storage.example/artifact"
            )


class RerunSchedulingTest(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(
            os.environ, {"GITHUB_RUN_ID": "700", "GITHUB_RUN_ATTEMPT": "1"}
        )
        patch.start()
        self.addCleanup(patch.stop)
        self.api = MemoryGitHub()
        self.request = self.api.accept()
        self.run = self.api.add_run()

    def schedule(self):
        return commands.schedule(self.api, self.api.pr, self.request)

    def test_completed_run_gets_one_full_rerun_and_no_cancel_or_label_writes(self):
        self.assertIn("rerun all jobs", self.schedule())
        self.assertEqual(self.api.submissions(), ["actions/runs/42/rerun"])
        self.assertEqual(self.run["run_attempt"], 2)
        self.schedule()
        self.assertEqual(len(self.api.submissions()), 1)
        self.assertFalse(
            any(
                "/cancel" in p or "/labels" in p or "rerun-failed" in p
                for p, _ in self.api.calls
            )
        )

    def test_active_and_queued_runs_defer_until_completion(self):
        for state in ("queued", "in_progress", "waiting", "pending"):
            self.run["status"] = state
            self.assertIn("finish before", self.schedule())
            self.assertEqual(self.api.submissions(), [])
        self.run["status"] = "completed"
        self.schedule()
        self.assertEqual(len(self.api.submissions()), 1)

    def test_completed_rerun_does_not_loop(self):
        self.schedule()
        self.run["status"] = "completed"
        self.api.plans[(42, 2)] = {
            **self.api.plans[(42, 1)],
            "attempt": "2",
            "request": self.request,
        }
        for _ in range(3):
            self.schedule()
        self.assertEqual(len(self.api.submissions()), 1)

    def test_normal_run_already_consumed_request_no_duplicate(self):
        self.api.plans[(42, 1)]["request"] = self.request
        self.assertIn("already consumed", self.schedule())
        self.assertEqual(self.api.submissions(), [])

    def test_multiple_pending_commands_coalesce_to_latest(self):
        self.run["status"] = "in_progress"
        self.api.add_command("/run-ppu-e2e test-qwen", 20)
        self.api.add_command("/clear-ppu-e2e", 30)
        commands.reconcile(self.api, 123)
        self.assertEqual(commands.resolve_request(self.api, 123)["comment_id"], 30)
        self.assertEqual(self.api.submissions(), [])
        self.run["status"] = "completed"
        commands.reconcile(self.api, 123)
        self.assertEqual(len(self.api.submissions()), 1)
        intents = [r for r in commands.receipts(self.api, 123) if r["kind"] == "rerun"]
        self.assertEqual(intents[0]["request_id"], 30)

    def test_stale_base_head_missing_artifact_and_old_runs_are_not_rerun(self):
        for mutation in ("head", "base", "artifact", "old"):
            api = MemoryGitHub()
            request = api.accept()
            run = api.add_run()
            if mutation == "head":
                api.pr["head"]["sha"] = "new"
            elif mutation == "base":
                api.pr["base"]["sha"] = "new"
            elif mutation == "artifact":
                api.plans.clear()
            else:
                run["created_at"] = "2020-01-01T00:00:00Z"
            with self.subTest(mutation=mutation):
                commands.schedule(api, api.pr, request)
                self.assertEqual(api.submissions(), [])

    def test_authorization_and_platform_approval_are_not_bypassed(self):
        for field, value in (("labels", []), ("draft", True), ("state", "closed")):
            old = self.api.pr[field]
            self.api.pr[field] = value
            self.schedule()
            self.assertEqual(self.api.submissions(), [])
            self.api.pr[field] = old
        self.run["conclusion"] = "action_required"
        self.assertIn("approval", self.schedule())
        self.assertEqual(self.api.submissions(), [])

    def test_clear_can_refresh_gate_without_run_ci(self):
        self.request = self.api.accept((), 20)
        self.api.pr["labels"] = []
        self.schedule()
        self.assertEqual(len(self.api.submissions()), 1)

    def test_ambiguous_api_failure_records_intent_and_does_not_retry(self):
        self.api.fail_rerun = True
        with self.assertRaises(TimeoutError):
            self.schedule()
        self.api.fail_rerun = False
        self.schedule()
        self.assertEqual(len(self.api.submissions()), 1)
        records = commands.receipts(self.api, 123)
        self.assertTrue(
            any(r["kind"] == "rerun-result" and r["accepted"] is False for r in records)
        )

    def test_artifact_attempt_mismatch_and_zip_inventory_rejected(self):
        self.api.plans[(42, 1)]["attempt"] = "0"
        with self.assertRaises(ValueError):
            commands.attempt_plan(self.api, self.run)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../selection.json", "{}")
        with mock.patch.object(
            self.api, "request", return_value=buffer.getvalue()
        ), self.assertRaises(ValueError):
            commands.attempt_plan(self.api, self.run)

    def test_unrelated_workflow_or_pr_cannot_be_rerun(self):
        for field, value in (
            ("path", "other.yml"),
            ("event", "push"),
            ("pull_requests", [{"number": 999}]),
        ):
            old = self.run[field]
            self.run[field] = value
            self.schedule()
            self.assertEqual(self.api.submissions(), [])
            self.run[field] = old

    def test_empty_pr_association_uses_api_fallback_and_exact_source_ref(self):
        self.run["pull_requests"] = []
        self.assertIn("rerun all jobs", self.schedule())
        api = MemoryGitHub()
        request = api.accept()
        run = api.add_run()
        run["pull_requests"] = []
        run["head_branch"] = "other-branch"
        commands.schedule(api, api.pr, request)
        self.assertEqual(api.submissions(), [])
        run["head_branch"] = api.pr["head"]["ref"]
        original = api.pages
        with mock.patch.object(
            api,
            "pages",
            side_effect=lambda path, key=None: (
                [] if path.startswith("commits/") else original(path, key)
            ),
        ):
            commands.schedule(api, api.pr, request)
        self.assertEqual(api.submissions(), [])

    def test_older_approval_failure_does_not_block_newer_approved_run(self):
        old = self.api.add_run(run_id=41)
        old["conclusion"] = "action_required"
        self.assertIn("rerun all jobs", self.schedule())

    def test_previous_unauthorized_plan_does_not_consume_hardware_request(self):
        self.api.plans[(42, 1)].update(request=self.request, labels=[])
        self.assertIn("rerun all jobs", self.schedule())

    def test_newer_ignored_label_run_without_plan_is_not_rerun(self):
        self.api.add_run(run_id=43)
        del self.api.plans[(43, 1)]
        self.assertIn("rerun all jobs", self.schedule())
        self.assertEqual(self.api.submissions(), ["actions/runs/42/rerun"])

    def test_scheduling_rereads_revision_authorization_and_attempt(self):
        for field, value in (
            ("head", {"sha": "new"}),
            ("base", {"sha": "new"}),
            ("draft", True),
            ("state", "closed"),
            ("labels", []),
            ("run_attempt", 2),
            ("status", "queued"),
            ("path", "other.yml"),
        ):
            api = MemoryGitHub()
            request = api.accept()
            api.add_run()
            original = api.get

            def changed(path):
                result = original(path)
                if (path.startswith("pulls/") and field in api.pr) or (
                    path.startswith("actions/runs/") and field not in api.pr
                ):
                    result[field] = value
                return result

            with self.subTest(field=field), mock.patch.object(
                api, "get", side_effect=changed
            ), self.assertRaises(ValueError):
                commands.schedule(api, api.pr, request)
            self.assertEqual(api.submissions(), [])

    def test_new_command_or_new_ci_during_scheduling_prevents_submission(self):
        for mutation in ("command", "run"):
            api = MemoryGitHub()
            request = api.accept()
            api.add_run()
            original = api.get

            def changed(path):
                if path.startswith("pulls/"):
                    if mutation == "command":
                        api.accept(("test-qwen",), 20)
                    else:
                        api.add_run(status="queued", run_id=43)
                return original(path)

            with self.subTest(mutation=mutation), mock.patch.object(
                api, "get", side_effect=changed
            ), self.assertRaises(ValueError):
                commands.schedule(api, api.pr, request)
            self.assertEqual(api.submissions(), [])

    def test_completion_event_resolves_only_current_revision(self):
        event = {
            "repository": {"full_name": self.api.repository},
            "action": "completed",
            "workflow_run": {"id": 42},
        }
        self.assertEqual(commands.resolve_event(self.api, event, "workflow_run"), 123)
        self.api.pr["head"]["sha"] = "new"
        self.assertIsNone(commands.resolve_event(self.api, event, "workflow_run"))
        self.run["event"] = "push"
        with self.assertRaises(ValueError):
            commands.resolve_event(self.api, event, "workflow_run")


class CommandWorkflowTest(unittest.TestCase):
    def test_handler_is_default_branch_only_with_separate_nonpreempting_scheduler(self):
        workflow = yaml.safe_load((ROOT / commands.HANDLER).read_text())
        events = workflow.get("on", workflow.get(True))
        self.assertEqual(events["issue_comment"]["types"], ["created"])
        self.assertEqual(events["workflow_run"]["types"], ["completed"])
        jobs = workflow["jobs"]
        self.assertFalse(jobs["schedule"]["concurrency"]["cancel-in-progress"])
        self.assertIn(
            "needs.resolve.outputs.pr", jobs["schedule"]["concurrency"]["group"]
        )
        for job in jobs.values():
            for step in job["steps"]:
                if step.get("uses") == "actions/checkout@v4":
                    self.assertEqual(
                        step["with"]["ref"],
                        "${{ github.event.repository.default_branch }}",
                    )
                    self.assertFalse(step["with"]["persist-credentials"])
                self.assertNotIn("${{", step.get("run", ""))
                self.assertNotIn("download-artifact", step.get("uses", ""))

    def test_label_events_never_preempt_and_nested_workflows_are_run_scoped(self):
        ci = yaml.safe_load((ROOT / commands.PIPELINE).read_text())
        template = ci["concurrency"]["cancel-in-progress"]
        cases = [
            ("pull_request", "synchronize", 1, True),
            ("pull_request", "labeled", 1, False),
            ("pull_request", "unlabeled", 1, False),
            ("pull_request", "synchronize", 2, False),
            ("pull_request_review", "submitted", 1, False),
        ]
        script = "const x=JSON.parse(process.argv[1]); console.log(JSON.stringify(x.events.map(github => Function('github', 'return ('+x.expression+');')(github))));"
        result = subprocess.run(
            [
                "node",
                "-e",
                script,
                json.dumps(
                    {
                        "expression": template.removeprefix("${{ ").removesuffix(" }}"),
                        "events": [
                            {"event_name": e, "event": {"action": a}, "run_attempt": n}
                            for e, a, n, _ in cases
                        ],
                    }
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(
            json.loads(result.stdout), [expected for _, _, _, expected in cases]
        )
        nested = yaml.safe_load(
            (ROOT / ".github/workflows/pr-test-ppu.yml").read_text()
        )["concurrency"]
        self.assertFalse(nested["cancel-in-progress"])
        self.assertIn("github.run_id", nested["group"])
        self.assertIn("github.run_attempt", nested["group"])


if __name__ == "__main__":
    unittest.main()
