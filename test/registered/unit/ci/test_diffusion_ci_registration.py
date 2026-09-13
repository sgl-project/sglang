import importlib
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]


class TestDiffusionCIRegistration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stack = ExitStack()
        cls.addClassCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {"USE_NPU_CONFIGS": "0"}))
        stack.enter_context(patch.dict(sys.modules))
        parser_dir = _REPO_ROOT / "scripts/ci/utils/diffusion"
        stack.enter_context(patch.object(sys, "path", [str(parser_dir), *sys.path]))
        for name in (
            "diffusion_case_parser",
            "compute_diffusion_partitions",
            "verify_diffusion_coverage",
        ):
            sys.modules.pop(name, None)
        cls.parser = importlib.import_module("diffusion_case_parser")
        cls.planner = importlib.import_module("compute_diffusion_partitions")
        cls.coverage = importlib.import_module("verify_diffusion_coverage")

    def test_llada_cases_reach_plans_and_coverage_with_full_estimates(self):
        runner = _REPO_ROOT / self.parser.RUN_SUITE_REL_PATH
        config = self.parser.resolve_case_config_path(_REPO_ROOT, runner)
        suites = self.parser.collect_diffusion_suites(
            config, runner, _REPO_ROOT / self.parser.BASELINE_REL_PATH
        )
        coverage = self.coverage.get_expected_cases(_REPO_ROOT)
        for degree in (1, 2):
            suite = f"{degree}-gpu"
            expected = {
                f"llada_image_turbo_fp8_{mode}_sp{degree}" for mode in ("t2i", "edit")
            }
            with self.subTest(suite=suite):
                cases = suites[suite].cases
                ids = [case.case_id for case in cases]
                self.assertEqual(expected.intersection(ids), expected)
                self.assertTrue(expected.issubset(coverage[suite]))
                for case in cases:
                    if case.case_id in expected:
                        self.assertEqual(case.est_time, 600.0)
                        self.assertEqual(ids.count(case.case_id), 1)
                for count in (1, 3, 8):
                    partitions = self.planner.partition_items_by_lpt(
                        self.planner.build_partition_items(suites[suite]), count
                    )
                    plan = self.planner.build_partition_plan(suite, partitions)
                    planned_ids = [
                        case_id
                        for partition in plan["partitions"]
                        for case_id in partition["case_ids"]
                    ]
                    for case_id in expected:
                        self.assertEqual(planned_ids.count(case_id), 1)

    def test_static_estimate_fallback_and_baseline_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "cases.py"
            config.write_text(
                'ONE_GPU_CASES = [DiffusionTestCase("explicit", estimated_full_test_time_s=600.0), '
                'DiffusionTestCase("fallback")]\n'
            )
            runner = root / "runner.py"
            runner.write_text("STANDALONE_FILES = {}\n")
            baseline = root / "baseline.json"
            for scenario, expected in (
                (None, 600.0),
                ({"expected_e2e_ms": 60000.0}, 600.0),
                ({"estimated_full_test_time_s": 420.0}, 420.0),
            ):
                with self.subTest(scenario=scenario):
                    scenarios = {} if scenario is None else {"explicit": scenario}
                    baseline.write_text(json.dumps({"scenarios": scenarios}))
                    suites = self.parser.collect_diffusion_suites(
                        config, runner, baseline
                    )
                    estimates = {
                        case.case_id: case.est_time for case in suites["1-gpu"].cases
                    }
                    self.assertEqual(estimates["explicit"], expected)
                    self.assertEqual(
                        estimates["fallback"], self.parser.DEFAULT_EST_TIME_SECONDS
                    )


if __name__ == "__main__":
    unittest.main()
