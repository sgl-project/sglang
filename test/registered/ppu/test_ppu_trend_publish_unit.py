"""临时bare remote中确定性复现发布竞争，不使用真实GitHub凭证。"""

import ast
import importlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_ppu_quality_trend_unit import KITS, PACKAGE, answer_report, register_cpu_ci
from test_ppu_trend_pipeline_unit import SCRIPTS, row, save

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def git(root, *args):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    return subprocess.check_output(
        ["git", "-C", str(root), *args], env=env, stderr=subprocess.PIPE, text=True
    ).strip()


class TestPublishing(unittest.TestCase):
    def setUp(self):
        self.assertTrue((SCRIPTS / "trend_publish.py").is_file(), "缺少隔离发布协议")
        self.mod = importlib.import_module("trend_publish")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.remote = self.root / "remote.git"
        self.remote.mkdir()
        git(self.remote, "init", "--bare")
        self.repo = self.root / "publisher"
        self.repo.mkdir()
        git(self.repo, "init", "-b", "nightly-test-data")
        self.add_row(self.repo, row(run="1"))
        git(self.repo, "remote", "add", "origin", str(self.remote))
        git(self.repo, "push", "-u", "origin", "nightly-test-data")
        self.rival = self.root / "rival"
        git(
            self.root,
            "clone",
            "--branch",
            "nightly-test-data",
            str(self.remote),
            str(self.rival),
        )
        self.output = self.root / "output"

    def add_row(self, repo, value):
        p = value["provenance"]
        save(
            repo / "data" / value["test_id"],
            value,
            f"2026-09-24-{p['github_run_id']}-{p['github_run_attempt']}.jsonl",
        )
        git(repo, "add", "data")
        git(repo, "commit", "-m", "fixture row")

    def publish(self, day="24", **kwargs):
        return self.mod.publish(
            self.repo,
            mode="derived",
            output=self.output,
            as_of_date="2026-09-" + day,
            revision="b" * 40,
            **kwargs,
        )

    def remote_json(self):
        return json.loads(git(self.remote, "show", "nightly-test-data:trend.json"))

    def test_shared_publication_recomputes_after_rival_push(self):
        original = self.mod._push
        calls = []

        def competing_push(repo):
            if not calls:
                self.add_row(self.rival, row(run="2", score=0.8))
                git(self.rival, "push", "origin", "nightly-test-data")
            calls.append(True)
            return original(repo)

        with mock.patch.object(
            self.mod, "_push", side_effect=competing_push
        ), mock.patch.object(self.mod.time, "sleep"):
            result = self.publish()
        self.assertEqual(result["status"], "published")
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.remote_json()["series"][0]["n_measured"], 2)
        self.assertEqual(
            git(self.remote, "ls-tree", "--name-only", "nightly-test-data"),
            "data\ntrend.json",
        )

    def test_idempotency_and_old_date_cannot_overwrite_new(self):
        self.publish()
        before = git(self.remote, "rev-parse", "nightly-test-data")
        self.assertEqual(self.publish()["status"], "unchanged")
        self.assertEqual(before, git(self.remote, "rev-parse", "nightly-test-data"))
        self.assertEqual(self.publish(day="23")["status"], "superseded")
        self.assertEqual(self.remote_json()["as_of_date"], "2026-09-24")

    def test_exhausted_push_preserves_remote_and_candidate_is_not_published(self):
        self.publish()
        before = git(self.remote, "rev-parse", "nightly-test-data")
        with mock.patch.object(
            self.mod, "_push", return_value=False
        ), mock.patch.object(self.mod.time, "sleep"):
            with self.assertRaises(RuntimeError):
                self.publish(day="25")
        self.assertEqual(git(self.remote, "rev-parse", "nightly-test-data"), before)
        status = json.loads((self.output / "publication.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertTrue((self.output / "trend.json").is_file())

    def test_invalid_input_preserves_last_published_view(self):
        self.publish()
        git(self.rival, "pull", "--ff-only")
        bad = row(run="2")
        bad["metrics"]["score"] = True
        self.add_row(self.rival, bad)
        git(self.rival, "push", "origin", "nightly-test-data")
        before = git(self.remote, "show", "nightly-test-data:trend.json")
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(
            git(self.remote, "show", "nightly-test-data:trend.json"), before
        )

    def test_shell_wrappers_publish_rows_then_snapshot(self):
        incoming = self.root / "incoming"
        save(incoming, row(run="3"))
        env = {
            **os.environ,
            "INCOMING_DIR": str(incoming),
            "TREND_OUTPUT_DIR": str(self.output),
            "TREND_TOOL_SHA": "b" * 40,
            "TREND_AS_OF_DATE": "2026-09-24",
            "GITHUB_RUN_ID": "3",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_STEP_SUMMARY": str(self.root / "step-summary"),
        }
        for script in ("publish_trend_rows.sh", "publish_derived_trend.sh"):
            result = subprocess.run(
                ["bash", str(SCRIPTS / script)],
                cwd=self.repo,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.remote_json()["series"][0]["n_measured"], 2)
        self.assertIn("published_commit:", (self.root / "step-summary").read_text())
        self.assertEqual(
            json.loads((self.output / "publication.json").read_text())["status"],
            "published",
        )

    def test_three_schemas_survive_publication_without_private_evidence(self):
        incoming = self.root / "incoming"
        quality = importlib.import_module(PACKAGE + ".quality_trend")
        report = answer_report()
        report["provenance"].update(github_run_id="3", github_run_attempt="1")
        report["cases"][0]["verdict"] = "failed"
        answer = incoming / "answer"
        answer.mkdir(parents=True)
        quality.write_quality_trend(report, answer)
        (answer / "result.raw.json").write_text("PRIVATE_SENTINEL")
        save(incoming / "accuracy", row(run="3", score=0.0))
        # 执行实际Perf行转换和序列化函数，避免手写fixture遗漏旧schema字段。
        tree = ast.parse((KITS / "perf_eval_kit.py").read_text())
        nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in {"trend_points", "render_trend_jsonl"}
            )
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name)
                    and t.id
                    in {"TREND_PROVENANCE_FIELDS", "TREND_POINT_SCHEMA_VERSION"}
                    for t in node.targets
                )
            )
        ]
        scope = {"Any": object, "json": json}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "perf-fixture", "exec"),
            scope,
        )
        perf_report = {
            "test_id": "perf-test",
            "config_digest": "c" * 64,
            "generated_at": "2026-09-24T01:00:00Z",
            "provenance": {"github_run_id": "3"},
            "measurements": [
                {
                    "id": "prefill",
                    "input_len": 4096,
                    "output_len": 1,
                    "num_prompts": 1,
                    "concurrency": 1,
                    "source_case": "fixture",
                    "tc_name": "prefill",
                    "status": "measured",
                    "reason_code": None,
                    "metrics": {"ttft_mean_ms": 10},
                }
            ],
        }
        perf = incoming / "perf"
        perf.mkdir()
        raw = scope["render_trend_jsonl"](perf_report)
        (perf / "trend.jsonl").write_text(raw)
        self.mod.publish(
            self.repo, mode="rows", incoming=incoming, run_id="3", attempt="1"
        )
        self.publish()
        summary = self.remote_json()
        self.assertEqual(
            {s["key"]["schema_version"] for s in summary["series"]},
            {
                "ppu-perf-trend-point/v1",
                "ppu-answer-trend-point/v1",
                "ppu-accuracy-trend-point/v1",
            },
        )
        archived = git(
            self.remote, "show", "nightly-test-data:data/perf-test/2026-09-24-3-1.jsonl"
        )
        self.assertEqual(archived + "\n", raw)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(summary))
        self.assertNotIn(
            "raw.json",
            git(self.remote, "ls-tree", "-r", "--name-only", "nightly-test-data"),
        )

    def test_output_symlink_does_not_overwrite_external_file(self):
        self.output.mkdir()
        outside = self.root / "protected"
        outside.write_text("untouched")
        (self.output / "publication.json").symlink_to(outside)
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(outside.read_text(), "untouched")

    def test_run_context_projects_callee_results_without_leaking_outputs(self):
        # 运行上下文只进入运行summary，用于提醒“结束不等于报告齐全”，
        # 并记录本次global运行身份；确定性trend.json不含这些字段。
        needs = {
            "accuracy": {"result": "success", "outputs": {"leak": "SECRET"}},
            "perf-8": {"result": "failure"},
            "answer-8": {"result": "skipped"},
            "prepare": {"result": "weird-status"},
        }
        text = self.mod.run_context(
            needs,
            source_sha="a" * 40,
            workflow_sha="c" * 40,
            tool_sha="b" * 40,
            executed_at="2026-09-24T12:00:00Z",
        )
        self.assertIn("以各原始发布job的对账为准", text)
        self.assertIn("| accuracy | success |", text)
        self.assertIn("| perf-8 | failure |", text)
        self.assertIn("| answer-8 | skipped |", text)
        # 未知枚举投影为 unknown，绝不回显任意状态或 outputs。
        self.assertIn("| prepare | unknown |", text)
        self.assertNotIn("weird-status", text)
        self.assertNotIn("SECRET", text)
        self.assertIn("源码 SHA：" + "a" * 40, text)
        self.assertIn("workflow SHA：" + "c" * 40, text)
        self.assertIn("工具 SHA：" + "b" * 40, text)
        self.assertIn("执行 UTC 时间：2026-09-24T12:00:00Z", text)

    def test_source_context_changes_only_summary_not_snapshot(self):
        before = None
        commit = None
        for source_sha in ("a" * 40, "d" * 40):
            summary = self.root / ("summary-" + source_sha)
            result = subprocess.run(
                ["bash", str(SCRIPTS / "publish_derived_trend.sh")],
                cwd=self.repo,
                env={
                    **os.environ,
                    "TREND_OUTPUT_DIR": str(self.output),
                    "TREND_TOOL_SHA": "b" * 40,
                    "TREND_AS_OF_DATE": "2026-09-24",
                    "RUN_SOURCE_SHA": source_sha,
                    "RUN_WORKFLOW_SHA": "c" * 40,
                    "GITHUB_STEP_SUMMARY": str(summary),
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            text = summary.read_text()
            self.assertIn("源码 SHA：" + source_sha, text)
            self.assertIn("workflow SHA：" + "c" * 40, text)
            self.assertIn("工具 SHA：" + "b" * 40, text)
            snapshot = (self.output / "trend.json").read_bytes()
            if before is not None:
                self.assertEqual(snapshot, before)
                self.assertEqual(
                    git(self.remote, "rev-parse", "nightly-test-data"), commit
                )
            before = snapshot
            commit = git(self.remote, "rev-parse", "nightly-test-data")

    def test_failure_after_staging_leaves_no_stale_worktree(self):
        before = git(self.repo, "worktree", "list", "--porcelain")
        with mock.patch.object(
            self.mod, "_commit", side_effect=RuntimeError("fixture")
        ):
            with self.assertRaises(RuntimeError):
                self.publish()
        self.assertEqual(git(self.repo, "worktree", "list", "--porcelain"), before)

    def test_raw_writers_preserve_both_independent_paths(self):
        incoming = self.root / "incoming"
        save(incoming, row(run="3"))
        original = self.mod._push
        calls = []

        def competing_push(repo):
            if not calls:
                self.add_row(self.rival, row(run="2"))
                git(self.rival, "push", "origin", "nightly-test-data")
            calls.append(True)
            return original(repo)

        with mock.patch.object(
            self.mod, "_push", side_effect=competing_push
        ), mock.patch.object(self.mod.time, "sleep"):
            result = self.mod.publish(
                self.repo, mode="rows", incoming=incoming, run_id="3", attempt="1"
            )
        self.assertEqual(result["status"], "published")
        self.assertEqual(
            len(
                git(
                    self.remote,
                    "ls-tree",
                    "-r",
                    "--name-only",
                    "nightly-test-data:data",
                ).splitlines()
            ),
            3,
        )
        result = self.mod.publish(
            self.repo, mode="rows", incoming=incoming, run_id="3", attempt="1"
        )
        self.assertEqual(result["status"], "unchanged")


if __name__ == "__main__":
    unittest.main()
