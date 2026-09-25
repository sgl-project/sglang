"""趋势归档及派生的离线协议测试；不访问真实远端。"""

import copy
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_ppu_quality_trend_unit import PACKAGE, ROOT, accuracy_report, register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCRIPTS = ROOT / "scripts/ci/ppu"
sys.path.insert(0, str(SCRIPTS))


def row(run="123", attempt="1", day="24", score=0.5):
    mod = importlib.import_module(PACKAGE + ".quality_trend")
    report = accuracy_report()
    report["generated_at"] = f"2026-09-{day}T00:00:00Z"
    report["provenance"].update(github_run_id=run, github_run_attempt=attempt)
    report["measurements"][0].update(score=score, ratio=score / 0.5)
    return mod.accuracy_points(report)[0]


def save(root, value, name="trend.jsonl"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(value) + "\n")
    return path


class TestStaging(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.incoming = self.root / "incoming"
        self.incoming.mkdir()
        self.env = {
            **os.environ,
            "INCOMING_DIR": str(self.incoming),
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "9",
        }

    def stage(self):
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "stage_trend_rows.py")],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def test_original_attempt_and_duplicate_artifacts_are_idempotent(self):
        value = row(attempt="2")
        save(self.incoming / "one", value)
        save(self.incoming / "two", value)
        result = self.stage()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        path = self.root / "data/accuracy-test/2026-09-24-123-2.jsonl"
        self.assertEqual(len(path.read_text().splitlines()), 1)
        self.assertEqual(self.stage().returncode, 0)
        changed = copy.deepcopy(value)
        changed["metrics"]["score"] = 0.2
        save(self.incoming / "two", changed)
        self.assertNotEqual(self.stage().returncode, 0)
        self.assertEqual(json.loads(path.read_text()), value)

    def test_validation_is_transactional_and_does_not_leak_input(self):
        save(self.incoming / "one", row())
        invalid = row()
        invalid["test_id"] = "../../PRIVATE_SENTINEL"
        save(self.incoming / "two", invalid)
        result = self.stage()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "data").exists())
        self.assertNotIn("PRIVATE_SENTINEL", result.stdout + result.stderr)

    def test_symlinks_and_conversion_markers_block_publication(self):
        outside = save(self.root / "outside", row())
        (self.incoming / "trend.jsonl").symlink_to(outside)
        self.assertNotEqual(self.stage().returncode, 0)
        (self.incoming / "trend.jsonl").unlink()
        save(self.incoming, row())
        (self.incoming / "trend-error.json").write_text("{}")
        self.assertNotEqual(self.stage().returncode, 0)

    def test_target_symlink_and_cross_run_are_rejected(self):
        save(self.incoming, row(run="456"))
        self.assertNotEqual(self.stage().returncode, 0)
        save(self.incoming, row())
        (self.root / "outside").mkdir()
        (self.root / "data").symlink_to(self.root / "outside", target_is_directory=True)
        self.assertNotEqual(self.stage().returncode, 0)
        self.assertEqual(list((self.root / "outside").iterdir()), [])

    def test_accounting_separates_missing_reports_and_unmeasured_points(self):
        value = row()
        value.update(
            status="unmeasured",
            metrics=None,
            reason_code="execution_failed",
            quality_verdict="not_evaluated",
            quality_reason_codes=[],
        )
        save(self.incoming, value)
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / "account_for_trend_reports.py")],
            env={
                **self.env,
                "NEEDS_JSON": json.dumps(
                    {"one": {"result": "failure"}, "two": {"result": "success"}}
                ),
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unmeasured=1", result.stdout)
        self.assertIn("report_gap=1", result.stdout)
        self.assertNotIn("every entry", result.stdout)

    def test_legacy_multiline_order_is_idempotent_without_rewriting_archive(self):
        first, second = row(), row()
        first["measurement_id"] = "z"
        second["measurement_id"] = "a"
        path = save(self.incoming, first)
        content = json.dumps(first) + "\n" + json.dumps(second) + "\n"
        path.write_text(content)
        target = self.root / "data/accuracy-test/2026-09-24-123-1.jsonl"
        target.parent.mkdir(parents=True)
        target.write_text(content)
        result = self.stage()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(target.read_text(), content)

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        path = save(self.incoming, row())
        for text in (
            '{"schema_version":"x","schema_version":"y"}',
            path.read_text().replace('"score": 0.5', '"score": NaN'),
        ):
            path.write_text(text)
            self.assertNotEqual(self.stage().returncode, 0)
        self.assertFalse((self.root / "data").exists())

    def test_accuracy_collector_keeps_trend_but_not_predictions(self):
        nas = self.root / "nas"
        save(nas, row())
        (nas / "trend-error.json").write_text("{}")
        save(nas / "evalscope/predictions", {"secret": "PRIVATE_SENTINEL"}, "raw.json")
        result = subprocess.run(
            ["bash", str(SCRIPTS / "collect_accuracy_evidence.sh")],
            env={
                **self.env,
                "ENTRY": "fixture",
                "ACCURACY_RESULTS_ON_RUNNER": str(nas),
                "GITHUB_WORKSPACE": str(self.root),
                "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        dest = self.root / "ppu-accuracy-artifacts"
        self.assertTrue((dest / "trend.jsonl").is_file())
        self.assertTrue((dest / "trend-error.json").is_file())
        self.assertFalse((dest / "evalscope/predictions").exists())
        self.assertTrue((nas / "trend.jsonl").exists())


class TestDerivation(unittest.TestCase):
    def setUp(self):
        self.assertTrue((SCRIPTS / "derive_trend.py").is_file(), "缺少确定性派生器")
        self.mod = importlib.import_module("derive_trend")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def archive(self, value):
        p = value["provenance"]
        date = (
            importlib.import_module(PACKAGE + ".trend_contract")
            .utc(value["generated_at"])
            .date()
        )
        return save(
            self.root / "data" / value["test_id"],
            value,
            f"{date}-{p['github_run_id']}-{p['github_run_attempt']}.jsonl",
        )

    def derive(self, day="24"):
        return self.mod.derive(
            self.root / "data",
            as_of_date="2026-09-" + day,
            input_data_tree="a" * 40,
            derivation_revision="b" * 40,
        )

    def test_newest_invalid_attempt_never_falls_back(self):
        self.archive(row(attempt="1", score=0.9))
        invalid = row(attempt="2")
        invalid.update(
            status="unmeasured",
            metrics=None,
            reason_code="execution_failed",
            quality_verdict="not_evaluated",
            quality_reason_codes=[],
        )
        self.archive(invalid)
        (series,) = self.derive()["series"]
        self.assertEqual(series["n_measured"], 0)
        self.assertEqual(series["n_unmeasured"], 1)
        self.assertEqual(series["latest_observation"]["attempt"], "2")

    def test_utc_window_stats_and_full_digest_history(self):
        for run, day, score in (
            ("1", "10", 0.1),
            ("2", "11", 0.2),
            ("3", "22", 0.4),
            ("4", "23", 0.6),
            ("5", "25", 0.8),
        ):
            self.archive(row(run=run, day=day, score=score))
        old = row(run="6", day="01")
        old["config_digest"] = "f" * 64
        self.archive(old)
        result = self.derive()
        active = next(
            s for s in result["series"] if s["key"]["config_digest"] == "e" * 64
        )
        metric = active["metrics"]["score"]
        self.assertEqual(metric["stats"]["n"], 3)
        self.assertAlmostEqual(metric["stats"]["p50"], 0.4)
        self.assertAlmostEqual(metric["stats"]["p90"], 0.56)
        self.assertAlmostEqual(metric["stats"]["mad"], 0.2)
        self.assertEqual(metric["observed_days"], 3)
        self.assertEqual(len(result["digest_history"]), 2)
        self.assertEqual(result, self.derive())
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))

    def test_sample_cap_does_not_limit_coverage_counts(self):
        for n in range(30):
            self.archive(row(run=str(n + 1), day=str(11 + n % 14), score=n / 100))
        (series,) = self.derive()["series"]
        self.assertEqual(series["n_measured"], 30)
        self.assertEqual(series["metrics"]["score"]["stats"]["n"], 16)
        self.assertEqual(series["metrics"]["score"]["observed_days"], 14)

    def test_zero_median_and_insufficient_samples(self):
        self.archive(row(run="1", score=0))
        metric = self.derive()["series"][0]["metrics"]["score"]
        self.assertEqual(metric["stats"]["status"], "insufficient_samples")
        for n in (2, 3):
            self.archive(row(run=str(n), score=0))
        metric = self.derive()["series"][0]["metrics"]["score"]
        self.assertIsNone(metric["deviation"])

    def test_renderer_shows_invalid_latest_and_escapes_strings(self):
        self.assertTrue((SCRIPTS / "render_trend.py").is_file(), "缺少Markdown渲染器")
        self.archive(row())
        summary = self.derive()
        summary["series"][0]["key"]["test_id"] = "<script>|[link](evil)"
        rendered = importlib.import_module("render_trend").render(summary)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("[link](evil)", rendered)
        self.assertIn("1/14", rendered)
        self.assertIn(
            "https://github.com/flytiger-eco/sglang-for-sail/actions/runs/123", rendered
        )
        self.assertIn("质量失败", rendered)
        self.assertIn("insufficient_samples", rendered)

    def test_legacy_republication_and_future_attempt_do_not_bias_snapshot(self):
        value = row(run="5", day="23")
        value = {
            key: value[key]
            for key in (
                "test_id",
                "measurement_id",
                "config_digest",
                "generated_at",
                "status",
                "reason_code",
                "provenance",
            )
        }
        value.update(
            schema_version="ppu-perf-trend-point/v1", metrics={"ttft_mean_ms": 10}
        )
        value["provenance"].pop("github_run_attempt")
        save(self.root / "data/accuracy-test", value, "2026-09-23-5-1.jsonl")
        save(self.root / "data/accuracy-test", value, "2026-09-23-5-2.jsonl")
        future = copy.deepcopy(value)
        future.update(
            generated_at="2026-09-25T00:00:00Z",
            status="failed",
            reason_code="execution_failed",
            metrics=None,
        )
        save(self.root / "data/accuracy-test", future, "2026-09-25-5-3.jsonl")
        (series,) = self.derive()["series"]
        self.assertEqual(series["n_measured"], 1)
        self.assertEqual(series["latest_observation"]["attempt"], "1")

    def test_legacy_partial_metrics_remain_unmeasured_and_extras_are_not_republished(
        self,
    ):
        value = row()
        value.update(
            schema_version="ppu-perf-trend-point/v1",
            status="failed",
            reason_code="incomplete_requests",
            metrics={"ttft_mean_ms": 1.0},
            config_components={"answer": "PRIVATE_SENTINEL"},
            quality_verdict="PRIVATE_SENTINEL",
            metric_name="PRIVATE_SENTINEL",
        )
        self.archive(value)
        summary = self.derive()
        self.assertEqual(summary["series"][0]["n_unmeasured"], 1)
        self.assertEqual(
            summary["series"][0]["metrics"]["ttft_mean_ms"]["stats"]["n"], 0
        )
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(summary))

    def test_metric_statistics_use_independent_valid_points_and_latest_invalid(self):
        for run in ("1", "2", "3"):
            value = row(run=run)
            if run == "2":
                value["metrics"].pop("ratio")
            self.archive(value)
        last = row(run="4")
        last.update(
            status="unmeasured",
            reason_code="execution_failed",
            metrics=None,
            quality_verdict="not_evaluated",
            quality_reason_codes=[],
        )
        self.archive(last)
        (series,) = self.derive()["series"]
        self.assertEqual(series["metrics"]["score"]["stats"]["n"], 3)
        self.assertEqual(series["metrics"]["ratio"]["stats"]["n"], 2)
        self.assertIsNone(series["metrics"]["score"]["deviation"])
        self.assertEqual(series["latest_observation"]["run_id"], "4")

    def test_conflicting_identity_and_filename_are_rejected(self):
        value = row()
        path = self.archive(value)
        changed = copy.deepcopy(value)
        changed["metrics"]["score"] = 0.8
        path.write_text(json.dumps(value) + "\n" + json.dumps(changed) + "\n")
        with self.assertRaises(ValueError):
            self.derive()
        path.write_text(json.dumps({**value, "test_id": "other"}) + "\n")
        with self.assertRaises(ValueError):
            self.derive()


if __name__ == "__main__":
    unittest.main()
