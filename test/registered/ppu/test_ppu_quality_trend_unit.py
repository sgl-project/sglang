"""质量趋势契约：低分不丢失、缺测不补零、公开字段最小化。"""

import ast
import copy
import importlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KITS = ROOT / "python/sglang/test/kits"
# 仅加载纯stdlib叶子模块，不导入sglang的硬件初始化入口。
PACKAGE = "ppu_trend_test_kits"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(KITS)]
    sys.modules[PACKAGE] = package
spec = importlib.util.spec_from_file_location(
    "trend_registry", ROOT / "python/sglang/test/ci/ci_register.py"
)
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)
register_cpu_ci = registry.register_cpu_ci
register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def provenance():
    return {
        "test_config_id": "answer-test",
        "test_config_sha256": "a" * 64,
        "github_run_id": "123",
        "github_run_attempt": "2",
        "served_model_name": "reviewed-model",
        "source_revision": "b" * 40,
        "evaluation": {"token": "PRIVATE_SENTINEL"},
        "node": "PRIVATE_SENTINEL",
        "package_versions": {"torch": "2.11.0", "private": "PRIVATE_SENTINEL"},
    }


def answer_report():
    return {
        "schema_version": "ppu-answer-result/v1",
        "created_at": "2026-09-24T01:00:00+08:00",
        "dataset": {"sha256": "c" * 64},
        "quality_profile": {"sha256": "d" * 64},
        "provenance": provenance(),
        "judge": {"mode": "disabled"},
        "summary": {"total": 2},
        "cases": [
            {
                "case_id": name,
                "verdict": "passed",
                "returned_model": "reviewed-model",
                "sample_id": name,
                "findings": [],
                "raw_response": "PRIVATE_SENTINEL",
            }
            for name in ("one", "two")
        ],
    }


def accuracy_report():
    return {
        "schema_version": "ppu-accuracy-result/v1",
        "test_id": "accuracy-test",
        "generated_at": "2026-09-24T00:00:00Z",
        "config_digest": "e" * 64,
        "provenance": provenance(),
        "measurements": [
            {
                "id": "gsm8k",
                "status": "measured",
                "reason_code": None,
                "score": 0.5,
                "ratio": 1.0,
                "metric_name": "AverageAccuracy",
                "primary_metric": "AverageAccuracy",
                "samples": 20,
                "expected_samples": 20,
                "baseline": 0.5,
                "min_ratio": 0.9,
                "max_ratio": 1.1,
                "execution": {
                    "requested": 20,
                    "succeeded": 20,
                    "errored": 0,
                    "incomplete": False,
                },
            }
        ],
    }


class TestQualityTrend(unittest.TestCase):
    def setUp(self):
        self.assertTrue((KITS / "quality_trend.py").is_file(), "缺少质量趋势转换器")
        self.mod = importlib.import_module(PACKAGE + ".quality_trend")
        self.contract = importlib.import_module(PACKAGE + ".trend_contract")

    def test_accuracy_keeps_low_zero_and_suspicious_high_scores(self):
        for score, reason in (
            (0.5, None),
            (0.0, "below_baseline"),
            (1.0, "above_baseline"),
        ):
            with self.subTest(score=score):
                report = accuracy_report()
                record = report["measurements"][0]
                record.update(
                    score=score,
                    ratio=score / 0.5,
                    reason_code=reason,
                    status="failed" if reason else "measured",
                )
                (row,) = self.mod.accuracy_points(report)
                self.assertEqual(row["status"], "measured")
                self.assertEqual(row["metrics"]["score"], score)
                self.assertEqual(
                    row["quality_verdict"], "failed" if reason else "passed"
                )
                self.assertIsNone(row["reason_code"])

    def test_accuracy_missing_inputs_are_not_quality_scores(self):
        for change in (
            {"score": None},
            {"metric_name": "other"},
            {"samples": 19},
            {"execution": {"errored": 1}},
            {"execution": None},
        ):
            report = accuracy_report()
            report["measurements"][0].update(change)
            (row,) = self.mod.accuracy_points(report)
            self.assertEqual(row["status"], "unmeasured")
            self.assertIsNone(row["metrics"])
            self.assertEqual(row["quality_verdict"], "not_evaluated")

    def test_accuracy_no_baseline_does_not_invent_verdict(self):
        report = accuracy_report()
        report["measurements"][0].update(baseline=None, ratio=None)
        (row,) = self.mod.accuracy_points(report)
        self.assertEqual(row["metrics"], {"score": 0.5})
        self.assertEqual(row["quality_verdict"], "not_evaluated")

    def test_answer_wrong_repeated_truncated_are_measured(self):
        for reason in (
            "fact_rule_failed",
            "periodic_fragment_repeat",
            "finish_reason_length",
        ):
            report = answer_report()
            report["cases"][0].update(
                verdict="failed",
                failure_class="candidate_failed",
                findings=[{"action": "hard_fail", "reason_code": reason}],
            )
            rows = self.mod.answer_points(report)
            self.assertEqual(rows[0]["metrics"], {"passed": 0})
            self.assertEqual(rows[-1]["metrics"], {"pass_rate": 0.5})
            self.assertEqual(rows[0]["quality_reason_codes"], [reason])

    def test_answer_infrastructure_missing_does_not_shrink_denominator(self):
        for change in (
            {"failure_class": "server_error"},
            {"failure_class": "runner_error"},
            {"returned_model": "other"},
            {"returned_model": None},
        ):
            report = answer_report()
            report["cases"][0].update(change)
            rows = self.mod.answer_points(report)
            self.assertIsNone(rows[0]["metrics"])
            self.assertIsNone(rows[-1]["metrics"])
            self.assertEqual(
                rows[-1]["counts"], {"total": 2, "measured": 1, "unmeasured": 1}
            )

    def test_answer_all_wrong_is_valid_zero(self):
        report = answer_report()
        for case in report["cases"]:
            case.update(
                verdict="failed",
                failure_class="candidate_failed",
                findings=[{"action": "hard_fail", "reason_code": "fact_rule_failed"}],
            )
        self.assertEqual(
            self.mod.answer_points(report)[-1]["metrics"], {"pass_rate": 0}
        )

    def test_digest_covers_content_not_key_order(self):
        report = answer_report()
        digest = self.mod.answer_points(report)[0]["config_digest"]
        reordered = json.loads(json.dumps(report, sort_keys=True))
        self.assertEqual(self.mod.answer_points(reordered)[0]["config_digest"], digest)
        for field in ("dataset", "quality_profile"):
            changed = copy.deepcopy(report)
            changed[field]["sha256"] = "f" * 64
            self.assertNotEqual(
                self.mod.answer_points(changed)[0]["config_digest"], digest
            )
            del changed[field]["sha256"]
            with self.assertRaises(ValueError):
                self.mod.answer_points(changed)

    def test_privacy_and_utc_original_identity(self):
        rows = self.mod.answer_points(answer_report())
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(rows))
        self.assertEqual(rows[0]["generated_at"], "2026-09-23T17:00:00+00:00")
        self.assertEqual(rows[0]["provenance"]["github_run_attempt"], "2")
        for row in rows:
            self.contract.validate_row(row)

    def test_validation_rejects_malformed_and_extra_fields(self):
        row = self.mod.answer_points(answer_report())[0]
        for change in (
            {"schema_version": "ppu-answer-trend-point/v99"},
            {"generated_at": "2026-02-30T00:00:00Z"},
            {"generated_at": "2026-09-24T00:00:00"},
            {"test_id": "../../escape"},
            {"config_digest": "abc"},
            {"raw_response": "PRIVATE_SENTINEL"},
            {"metrics": {"passed": True}},
            {"metrics": {"passed": float("nan")}},
            {"metrics": {"passed": float("inf")}},
            {"metrics": {"unexpected": 1}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.contract.validate_row({**row, **change})

    def test_conversion_error_preserves_reports_and_is_sanitized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "result.json").write_text("original")
            report = answer_report()
            report["dataset"]["sha256"] = "PRIVATE_SENTINEL"
            self.mod.write_quality_trend(report, root)
            self.assertEqual((root / "result.json").read_text(), "original")
            self.assertFalse((root / "trend.jsonl").exists())
            self.assertNotIn(
                "PRIVATE_SENTINEL", (root / "trend-error.json").read_text()
            )

    def test_conversion_recovery_replaces_stale_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = answer_report()
            self.mod.write_quality_trend(good, root)
            self.mod.write_quality_trend({}, root)
            self.assertFalse((root / "trend.jsonl").exists())
            self.assertTrue((root / "trend-error.json").exists())
            self.mod.write_quality_trend(good, root)
            self.assertTrue((root / "trend.jsonl").exists())
            self.assertFalse((root / "trend-error.json").exists())

    def test_case_counts_cannot_smuggle_unvalidated_objects(self):
        value = self.mod.answer_points(answer_report())[0]
        value["counts"] = {"answer": "PRIVATE_SENTINEL"}
        with self.assertRaises(ValueError):
            self.contract.validate_row(value)

    def test_both_report_writers_emit_trends_after_primary_reports(self):
        for name, fixture in (("answer", answer_report), ("accuracy", accuracy_report)):
            tree = ast.parse((KITS / (name + "_eval_kit.py")).read_text())
            writer = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "write_report_files"
            )
            scope = {
                "Any": object,
                "Path": Path,
                "json": json,
                "redact_report": lambda report: report,
                "render_summary": lambda report: "summary",
                "render_junit": lambda report: b"junit",
                "write_quality_trend": self.mod.write_quality_trend,
            }
            exec(
                compile(ast.Module(body=[writer], type_ignores=[]), "writer", "exec"),
                scope,
            )
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                scope["write_report_files"](fixture(), root)
                self.assertTrue((root / "trend.jsonl").is_file(), name)
                self.assertTrue((root / "junit.xml").is_file())

    def test_writer_outputs_all_points_and_refuses_missing_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            self.mod.write_quality_trend(answer_report(), Path(temp))
            self.assertEqual(
                len((Path(temp) / "trend.jsonl").read_text().splitlines()), 3
            )
        report = answer_report()
        del report["provenance"]["github_run_attempt"]
        with self.assertRaises(ValueError):
            self.mod.answer_points(report)


if __name__ == "__main__":
    unittest.main()
