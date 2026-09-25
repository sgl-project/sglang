"""CPU检查发布工具信任边界、artifact分区和既有调度DAG。"""

import fnmatch
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_ppu_quality_trend_unit import ROOT, register_cpu_ci

try:
    import yaml
except ImportError:
    yaml = None

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
WORKFLOWS = ROOT / ".github/workflows"
NAMES = [
    "perf",
    "perf-16",
    "perf-32",
    "pd-perf-glm52",
    "pd-perf-qwen35",
    "answer",
    "answer-16",
    "answer-32",
    "accuracy",
]


def load(path):
    return yaml.safe_load(path.read_text())


@unittest.skipUnless(yaml, "workflow结构验证需要PyYAML")
class TestTrendWorkflows(unittest.TestCase):
    def check_publisher(self, job, mode):
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        self.assertEqual(job["permissions"], {"contents": "write", "actions": "read"})
        self.assertIn("!cancelled()", job["if"])
        self.assertEqual(job["env"]["TREND_TOOL_SHA"], "${{ vars.PPU_TREND_TOOL_SHA }}")
        self.assertNotIn("runner.", str(job["env"]))
        steps = job["steps"]
        validation = next(s for s in steps if s.get("id") == "tool-version")
        self.assertLess(
            steps.index(validation), next(i for i, s in enumerate(steps) if "uses" in s)
        )
        for value in ("", "v0.5.18", "a" * 39, "a" * 40 + "\nevil"):
            result = subprocess.run(
                ["bash", "-c", validation["run"]],
                env={**os.environ, "TREND_TOOL_SHA": value},
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
        source = next(s for s in steps if s.get("with", {}).get("path") == "source")
        self.assertEqual(source["with"]["ref"], "${{ env.TREND_TOOL_SHA }}")
        self.assertIs(source["with"]["persist-credentials"], False)
        self.assertFalse(any("inputs.ref" in str(s) for s in steps))
        data = next(s for s in steps if s.get("with", {}).get("path") == "series")
        self.assertEqual(data["with"]["ref"], "nightly-test-data")
        self.assertEqual(data["with"]["fetch-depth"], 0)
        self.assertTrue(
            any("publish_" + mode + ".sh" in s.get("run", "") for s in steps)
        )
        self.assertTrue(
            any(
                s.get("uses") == "actions/upload-artifact@v4"
                and s.get("if") == "always()"
                for s in steps
            )
        )

    def test_all_nine_raw_publishers_preserve_measurement_jobs(self):
        for name in NAMES:
            with self.subTest(name=name):
                path = WORKFLOWS / ("test-ppu-" + name + ".yml")
                workflow = load(path)
                self.assertTrue(
                    "publish-trend-rows" in workflow["jobs"], "缺少原始发布job"
                )
                job = workflow["jobs"]["publish-trend-rows"]
                self.check_publisher(job, "trend_rows")
                measuring = {
                    k: v
                    for k, v in workflow["jobs"].items()
                    if k != "publish-trend-rows"
                }
                self.assertEqual(set(job["needs"]), set(measuring))
                self.assertEqual(workflow["permissions"], {"contents": "read"})
                for measurement in measuring.values():
                    self.assertEqual(
                        measurement.get("permissions", workflow["permissions"])[
                            "contents"
                        ],
                        "read",
                    )
                    self.assertNotIn("publish_derived_trend", str(measurement))

    def test_artifact_patterns_are_disjoint_and_accounting_is_explicit(self):
        patterns = []
        for name in NAMES:
            jobs = load(WORKFLOWS / ("test-ppu-" + name + ".yml"))["jobs"]
            self.assertTrue("publish-trend-rows" in jobs, "缺少原始发布job")
            steps = jobs["publish-trend-rows"]["steps"]
            pattern = next(
                s["with"]["pattern"]
                for s in steps
                if s.get("uses") == "actions/download-artifact@v4"
            )
            patterns.append(pattern)
            account = next(
                s for s in steps if "account_for_trend_reports.py" in s.get("run", "")
            )
            self.assertIn("steps.trusted.outcome == 'success'", account["if"])
            self.assertIn("NEEDS_JSON", account["env"])
        examples = [
            "ppu-perf-k8s-glm52",
            "ppu-perf-k8s-16",
            "ppu-perf-k8s-32",
            "ppu-pd-perf-k8s-16-glm52-fp8",
            "ppu-pd-perf-k8s-16-qwen35-fp8",
            "ppu-answer-k8s-qwen3.8-27b",
            "ppu-answer-k8s-16-kimi26",
            "ppu-answer-k8s-32",
            "ppu-accuracy-k8s-glm52-smoke",
        ]
        for index, example in enumerate(examples):
            artifact = example + "-${{ github.run_id }}-${{ github.run_attempt }}"
            self.assertEqual(
                [i for i, p in enumerate(patterns) if fnmatch.fnmatchcase(artifact, p)],
                [index],
            )

    def test_main_has_only_one_cpu_derived_publisher_without_dag_changes(self):
        main = os.environ.get("PPU_TREND_MAIN_ROOT")
        if not main:
            self.skipTest("跨分支检查需要PPU_TREND_MAIN_ROOT")
        root = Path(main)
        path = root / ".github/workflows/nightly-test-ppu-global.yml"
        workflow = load(path)
        self.assertTrue("publish-derived-trend" in workflow["jobs"], "缺少唯一派生job")
        job = workflow["jobs"]["publish-derived-trend"]
        self.check_publisher(job, "derived_trend")
        # 运行上下文需要callee结果与本次global运行身份，仅写运行summary。
        publish = next(
            s for s in job["steps"] if "publish_derived_trend.sh" in s.get("run", "")
        )
        self.assertEqual(publish["env"]["NEEDS_JSON"], "${{ toJSON(needs) }}")
        self.assertEqual(publish["env"]["RUN_SOURCE_SHA"], "${{ github.sha }}")
        self.assertEqual(
            publish["env"]["RUN_WORKFLOW_SHA"], "${{ github.workflow_sha }}"
        )
        remaining = {
            k: v for k, v in workflow["jobs"].items() if k != "publish-derived-trend"
        }
        expected = [
            "prepare",
            "pd-glm52",
            "pd-qwen35",
            "perf-8",
            "perf-16",
            "perf-32",
            "answer-8",
            "answer-16",
            "answer-32",
            "accuracy",
        ]
        self.assertEqual(set(remaining), set(expected))
        self.assertEqual(set(job["needs"]), set(expected))
        for index, name in enumerate(expected[1:], 1):
            boundary = 1 if index < 3 else 3 if index < 6 else 6 if index < 9 else 9
            self.assertEqual(remaining[name]["needs"], expected[:boundary])
            self.assertTrue(remaining[name]["uses"].endswith("@v0.5.18"))
            self.assertEqual(
                remaining[name]["with"]["ref"],
                "${{ needs.prepare.outputs.source_sha }}",
            )
        self.assertEqual(
            job["if"], "${{ !cancelled() && needs.prepare.result == 'success' }}"
        )
        self.assertEqual(
            workflow["concurrency"],
            {"group": "ppu-nightly-btv15", "cancel-in-progress": False, "queue": "max"},
        )

    def test_answer_collectors_preserve_root_and_rank_layouts(self):
        for variant in ("", "-16", "-32"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                nas = root / "nas"
                (nas / "ranks/rank-1").mkdir(parents=True)
                (nas / "trend.jsonl").write_text("root trend\n")
                (nas / "trend-error.json").write_text("{}\n")
                (nas / "ranks/rank-1/environment.json").write_text("PRIVATE_SENTINEL")
                if variant:
                    workflow = load(WORKFLOWS / ("test-ppu-answer" + variant + ".yml"))
                    job = next(iter(workflow["jobs"].values()))
                    script = next(
                        s["run"]
                        for s in job["steps"]
                        if s.get("name") == "Collect Answer evidence"
                    )
                    command = ["bash", "-c", script]
                else:
                    command = [
                        "bash",
                        str(ROOT / "scripts/ci/ppu/collect_answer_evidence.sh"),
                    ]
                result = subprocess.run(
                    command,
                    env={
                        **os.environ,
                        "ENTRY": "fixture",
                        "SUITE": "fixture",
                        "ANSWER_RESULTS_ON_RUNNER": str(nas),
                        "GITHUB_WORKSPACE": str(root),
                        "GITHUB_STEP_SUMMARY": str(root / "summary"),
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                destination = root / "ppu-answer-artifacts"
                for file in (
                    "trend.jsonl",
                    "trend-error.json",
                    "ranks/rank-1/environment.json",
                ):
                    self.assertEqual(
                        (nas / file).read_bytes(), (destination / file).read_bytes()
                    )

    def test_legacy_answer_caller_grants_only_call_jobs_write_permission(self):
        workflow = load(WORKFLOWS / "nightly-test-ppu-answer.yml")
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        for job in workflow["jobs"].values():
            self.assertEqual(
                job.get("permissions"), {"contents": "write", "actions": "read"}
            )


if __name__ == "__main__":
    unittest.main()
