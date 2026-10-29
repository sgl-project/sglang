"""CPU-only regression checks for AMD registration and workflow dispatch wiring."""

import ast
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REGISTRY = load_module("ci_register", "python/sglang/test/ci/ci_register.py")
LINT = load_module("check_registered_tests", "scripts/lint/check_registered_tests.py")


def suite_namespace():
    # Exercise the real catalog/validator without importing the GPU harness.
    names = {
        "PER_COMMIT_SUITES",
        "NIGHTLY_SUITES",
        "OTHER_SUITES",
        "_SUITE_CHECKED_BACKENDS",
        "_valid_suites_by_backend",
        "validate_all_suites",
    }
    tree = ast.parse((ROOT / "test/run_suite.py").read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in node.targets
            )
        )
    ]
    namespace = {
        "HWBackend": REGISTRY.HWBackend,
        "CIRegistry": REGISTRY.CIRegistry,
        "List": list,
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "run_suite.py", "exec"),
        namespace,
    )
    return namespace


def workflow(name):
    return yaml.safe_load(
        (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
    )


class TestAMDRegistration(unittest.TestCase):
    def test_lint_rejects_an_undeclared_amd_pr_suite(self):
        # The old hook accepted this exact orphan registration without error.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_orphan.py"
            path.write_text(
                'register_amd_ci(est_time=2, suite="stage-a-test-cpu-amd")\n'
            )
            output = io.StringIO()
            with patch.object(LINT.glob, "glob", return_value=[str(path)]):
                with redirect_stdout(output):
                    result = LINT.main()
            self.assertEqual(result, 1)
            self.assertIn("stage-a-test-cpu-amd", output.getvalue())

    def test_rejects_orphan_pr_suites(self):
        validate = suite_namespace()["validate_all_suites"]
        for suite in ("stage-a-test-cpu-amd", "base-b-test-2-gpu-large"):
            with self.subTest(suite=suite), self.assertRaisesRegex(ValueError, suite):
                validate(
                    [
                        SimpleNamespace(
                            backend=REGISTRY.HWBackend.AMD,
                            nightly=False,
                            filename="regression.py",
                            effective_suite=suite,
                        )
                    ]
                )

    def test_accepts_pr_suites_and_legacy_nightly(self):
        namespace = suite_namespace()
        suites = namespace["PER_COMMIT_SUITES"][REGISTRY.HWBackend.AMD]
        rows = [
            SimpleNamespace(
                backend=REGISTRY.HWBackend.AMD,
                nightly=False,
                filename="regression.py",
                effective_suite=suite,
            )
            for suite in suites
        ]
        rows.append(
            SimpleNamespace(
                backend=REGISTRY.HWBackend.AMD,
                nightly=True,
                filename="legacy.py",
                effective_suite="nightly-perf-8-gpu-gpt-oss",
            )
        )
        namespace["validate_all_suites"](rows)
        self.assertEqual(LINT.declared_amd_pr_suites(), set(suites))

    def test_all_amd_pr_registrations_have_dispatch_jobs(self):
        jobs = set(workflow("pr-test-amd.yml")["jobs"])
        jobs.update(workflow("pr-test-amd-extra.yml")["jobs"])
        errors = []
        for path in (ROOT / "test/registered").rglob("*.py"):
            if path.name in ("__init__.py", "conftest.py"):
                continue
            rows, _ = REGISTRY.ut_parse_one_file(str(path))
            errors.extend(
                f"{path}: {row.effective_suite}"
                for row in rows
                if (
                    row.backend == REGISTRY.HWBackend.AMD
                    and not row.nightly
                    and row.effective_suite not in jobs
                )
            )
        self.assertEqual(errors, [])

    def test_fp8_kv_files_select_the_existing_nightly_job(self):
        for name in ("basic", "mtp"):
            path = ROOT / f"test/registered/amd/test_deepseek_v3_{name}_kv_fp8.py"
            rows, _ = REGISTRY.ut_parse_one_file(str(path))
            self.assertEqual(
                [(r.effective_suite, r.nightly) for r in rows],
                [("nightly-amd-8-gpu", True)],
            )
        job = workflow("nightly-test-amd.yml")["jobs"]["nightly-8-gpu"]
        self.assertTrue(
            any(
                "--suite nightly-amd-8-gpu --nightly" in step.get("run", "")
                for step in job["steps"]
            )
        )


class TestAMDWorkflows(unittest.TestCase):
    def test_multi_shard_lanes_are_not_serialized(self):
        jobs = workflow("pr-test-amd.yml")["jobs"]
        for name in (
            "stage-b-test-1-gpu-large-amd",
            "multimodal-gen-test-1-gpu-amd",
            "multimodal-gen-test-2-gpu-amd",
        ):
            with self.subTest(job=name):
                strategy = jobs[name]["strategy"]
                self.assertGreater(len(strategy["matrix"]["part"]), 1)
                self.assertGreater(strategy["max-parallel"], 1)

    def test_stage_b_wait_cannot_hit_hosted_job_cap(self):
        jobs = workflow("pr-test-amd.yml")["jobs"]
        gate = jobs["wait-for-stage-b-amd"]
        inputs = next(s["with"] for s in gate["steps"] if s.get("id") == "wait")
        self.assertLess(int(inputs["max-wait-minutes"]), 360)
        self.assertEqual(inputs["fail-on-timeout"], "false")
        finish = jobs["pr-test-amd-finish"]
        for spec in json.loads(inputs["jobs"]):
            job_name = spec["prefix"]
            self.assertIn(job_name, finish["needs"])
            self.assertEqual(
                spec["expected_count"],
                len(jobs[job_name]["strategy"]["matrix"]["part"]),
            )
        self.assertIn('"failure"', finish["steps"][0]["run"])
        self.assertIn('"cancelled"', finish["steps"][0]["run"])

    def test_monitor_queries_and_reports_standalone_extra(self):
        jobs = workflow("amd-ci-job-monitor.yml")["jobs"]
        select = next(
            s["run"]
            for s in jobs["fetch-actions-data"]["steps"]
            if s.get("id") == "select-workflows"
        )
        # Both filtered and full scans must include the standalone workflow.
        self.assertEqual(select.count("pr-test-amd-extra.yml"), 2)
        reports = jobs["extra-ci-reports"]
        self.assertIn("extra_jobs", reports["strategy"]["matrix"]["job_name"])
        self.assertTrue(
            any(
                '--workflow "pr-test-amd-extra.yml"' in s.get("run", "")
                for s in reports["steps"]
            )
        )
        fleet = jobs["runner-fleet-report"]["steps"][-1]["run"]
        self.assertIn("pr-test-amd-extra.yml", fleet)


if __name__ == "__main__":
    unittest.main()
