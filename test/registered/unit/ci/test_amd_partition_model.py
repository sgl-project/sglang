import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = load_script("amd_partition_model", "scripts/ci/amd/build_partition_model.py")
AMD_SUITES = load_script("amd_suites", "scripts/ci/amd_suites.py")
RUNNER = MODEL.runner_name("mi300")


def timing_line(*, filename, elapsed, passed=True):
    return "2026-11-13T07:46:20Z " + json.dumps(
        {"file": filename, "passed": passed, "elapsed": elapsed}
    )


class TestAmdPartitionModel(CustomTestCase):
    def test_checked_in_models_key_a_suite_an_amd_job_runs(self):
        # run_suite.py silently falls back to est_time when the key does not match.
        models = sorted(MODEL.MODEL_DIR.glob("partition-model-*-*.json"))
        self.assertTrue(models)
        workflow = (REPO_ROOT / ".github/workflows/pr-test-amd.yml").read_text()
        self.assertIn(
            f"--suite {MODEL.SUITE} --auto-partition-id ${{{{ matrix.part }}}} "
            "--auto-partition-size 8 --timeout-per-file 2400 --partition-model-file "
            "/sglang-checkout/scripts/ci/amd/partition-model-"
            "${{ inputs.runner_arch || 'mi300' }}-${{ matrix.rocm_version }}.json",
            workflow,
        )
        self.assertIn(MODEL.SUITE, AMD_SUITES.load_suites())
        for path in models:
            with self.subTest(model=path.name):
                model = json.loads(path.read_text())
                self.assertEqual(list(model["est"]), [MODEL.SUITE])
                runner_arch, rocm_version = path.stem.split("-")[2:]
                self.assertEqual(
                    model["metadata"]["runner"], MODEL.runner_name(runner_arch)
                )
                self.assertEqual(model["metadata"]["rocm_version"], rocm_version)

    def test_timings_ignore_failed_tests_and_unrelated_output(self):
        filename = "/sglang-checkout/test/registered/example.py"
        log = "\n".join(
            [
                "not a metric",
                timing_line(filename=filename, elapsed=12, passed=False),
                timing_line(filename=filename, elapsed=8),
            ]
        )
        self.assertEqual(
            MODEL.parse_test_timings(log), {"test/registered/example.py": 8}
        )

    def test_invalid_values_and_paths_cannot_poison_estimates(self):
        filename = "/sglang-checkout/test/registered/example.py"
        for elapsed in [True, "30", -1, float("nan")]:
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                MODEL.parse_test_timings(
                    timing_line(filename=filename, elapsed=elapsed)
                )
        for filename in ["/tmp/test.py", "test/registered/../../secret.py"]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                MODEL.parse_test_timings(timing_line(filename=filename, elapsed=4))

    def test_median_model_is_suite_scoped_and_preserves_sparse_samples(self):
        result = MODEL.build_model(
            traces=[
                {"test/registered/a.py": 10, "test/registered/b.py": 8},
                {"test/registered/a.py": 12},
                {"test/registered/a.py": 100},
            ],
            source_runs=[1, 2, 3],
            runner_arch="mi300",
            rocm_version="rocm10",
        )
        self.assertEqual(
            result["est"],
            {MODEL.SUITE: {"test/registered/a.py": 12, "test/registered/b.py": 8}},
        )
        self.assertEqual(
            result["metadata"]["samples"],
            {"test/registered/a.py": 3, "test/registered/b.py": 1},
        )
        self.assertEqual(result["metadata"]["source_runs"], [1, 2, 3])
        for traces, runs in [([], []), ([{}], [1]), ([{"a": 1}], [])]:
            with self.subTest(traces=traces), self.assertRaises(ValueError):
                MODEL.build_model(
                    traces=traces,
                    source_runs=runs,
                    runner_arch="mi300",
                    rocm_version="rocm10",
                )
        with self.assertRaises(ValueError):
            MODEL.build_model(
                traces=[{"test/registered/a.py": 1}] * 2,
                source_runs=[1, 1],
                runner_arch="mi300",
                rocm_version="rocm10",
            )

    def test_incomplete_or_failed_shards_are_rejected(self):
        for jobs in [
            [],
            [
                {
                    "id": 1,
                    "name": f"{MODEL.SUITE} (rocm10, {RUNNER}, 0)",
                    "conclusion": "failure",
                }
            ],
        ]:
            with (
                self.subTest(jobs=jobs),
                patch.object(MODEL, "github_api", return_value={"jobs": jobs}),
            ):
                with self.assertRaises(ValueError):
                    MODEL.collect_run_timings(
                        run_id=1,
                        runner_arch="mi300",
                        rocm_version="rocm10",
                        expected_shards=1,
                    )

    def test_other_rocm_versions_and_runners_do_not_mix_into_the_model(self):
        jobs = [
            {
                "id": 1,
                "name": f"{MODEL.SUITE} (rocm10, {RUNNER}, 0)",
                "conclusion": "success",
            },
            {
                "id": 2,
                "name": f"{MODEL.SUITE} (rocm720, {RUNNER}, 0)",
                "conclusion": "success",
            },
            {
                "id": 3,
                "name": f"{MODEL.SUITE} (rocm10, linux-mi325-1gpu-sglang, 0)",
                "conclusion": "success",
            },
        ]
        responses = [
            {"jobs": jobs},
            timing_line(filename="test/registered/example.py", elapsed=42),
        ]
        with patch.object(MODEL, "github_api", side_effect=responses):
            result = MODEL.collect_run_timings(
                run_id=1, runner_arch="mi300", rocm_version="rocm10", expected_shards=1
            )
        self.assertEqual(result, {"test/registered/example.py": 42})

    def test_duplicate_files_across_shards_are_rejected(self):
        jobs = [
            {
                "id": part,
                "name": f"{MODEL.SUITE} (rocm10, {RUNNER}, {part})",
                "conclusion": "success",
            }
            for part in range(2)
        ]
        record = timing_line(filename="test/registered/example.py", elapsed=4)
        with patch.object(
            MODEL, "github_api", side_effect=[{"jobs": jobs}, record, record]
        ):
            with self.assertRaises(ValueError):
                MODEL.collect_run_timings(
                    run_id=1,
                    runner_arch="mi300",
                    rocm_version="rocm10",
                    expected_shards=2,
                )


if __name__ == "__main__":
    unittest.main()
