import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/ci/amd/build_partition_model.py"
SPEC = importlib.util.spec_from_file_location("amd_partition_model", SCRIPT)
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def timing_line(filename, elapsed, passed=True):
    return "2026-11-13T07:46:20Z " + json.dumps(
        {"file": filename, "passed": passed, "elapsed": elapsed}
    )


class TestAmdPartitionModel(CustomTestCase):
    def test_timings_ignore_failed_tests_and_unrelated_output(self):
        filename = "/sglang-checkout/test/registered/example.py"
        log = "\n".join(
            ["not a metric", timing_line(filename, 12, False), timing_line(filename, 8)]
        )
        self.assertEqual(
            MODEL.parse_test_timings(log), {"test/registered/example.py": 8}
        )

    def test_key_order_whitespace_and_color_do_not_hide_timings(self):
        record = json.dumps(
            {"elapsed": 12, "passed": True, "file": "test/registered/example.py"},
            separators=(",", ": "),
        )
        log = f"2026-11-13T07:46:20Z \x1b[32m{record}\x1b[0m"
        self.assertEqual(
            MODEL.parse_test_timings(log), {"test/registered/example.py": 12}
        )

    def test_invalid_values_and_paths_cannot_poison_estimates(self):
        filename = "/sglang-checkout/test/registered/example.py"
        for elapsed in [True, "30", -1, float("inf"), float("nan")]:
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                MODEL.parse_test_timings(timing_line(filename, elapsed))
        for filename in ["/tmp/test.py", "test/registered/../../secret.py"]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                MODEL.parse_test_timings(timing_line(filename, 4))

    def test_zero_duration_does_not_create_a_zero_cost_shard(self):
        self.assertEqual(
            MODEL.parse_test_timings(timing_line("test/registered/example.py", 0)),
            {"test/registered/example.py": 1},
        )

    def test_median_model_is_suite_scoped_and_preserves_sparse_samples(self):
        result = MODEL.build_model(
            [
                {"test/registered/a.py": 10, "test/registered/b.py": 8},
                {"test/registered/a.py": 12},
                {"test/registered/a.py": 100},
            ],
            [1, 2, 3],
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
                MODEL.build_model(traces, runs)
        with self.assertRaises(ValueError):
            MODEL.build_model([{"test/registered/a.py": 1}] * 2, [1, 1])

    def test_incomplete_or_failed_shards_are_rejected(self):
        for jobs in [
            [],
            [
                {
                    "id": 1,
                    "name": f"{MODEL.SUITE} (rocm10, {MODEL.RUNNER}, 0)",
                    "conclusion": "failure",
                }
            ],
        ]:
            with (
                self.subTest(jobs=jobs),
                patch.object(MODEL, "github_api", return_value={"jobs": jobs}),
            ):
                with self.assertRaises(ValueError):
                    MODEL.collect_run_timings(1, expected_shards=1)

    def test_other_rocm_versions_and_runners_do_not_mix_into_the_model(self):
        jobs = [
            {
                "id": 1,
                "name": f"{MODEL.SUITE} (rocm10, {MODEL.RUNNER}, 0)",
                "conclusion": "success",
            },
            {
                "id": 2,
                "name": f"{MODEL.SUITE} (rocm720, {MODEL.RUNNER}, 0)",
                "conclusion": "success",
            },
            {
                "id": 3,
                "name": f"{MODEL.SUITE} (rocm10, linux-mi325-1gpu-sglang, 0)",
                "conclusion": "success",
            },
        ]
        responses = [{"jobs": jobs}, timing_line("test/registered/example.py", 42)]
        with patch.object(MODEL, "github_api", side_effect=responses):
            result = MODEL.collect_run_timings(1, expected_shards=1)
        self.assertEqual(result, {"test/registered/example.py": 42})

    def test_duplicate_files_across_shards_are_rejected(self):
        jobs = [
            {
                "id": part,
                "name": f"{MODEL.SUITE} (rocm10, {MODEL.RUNNER}, {part})",
                "conclusion": "success",
            }
            for part in range(2)
        ]
        record = timing_line("test/registered/example.py", 4)
        with patch.object(
            MODEL, "github_api", side_effect=[{"jobs": jobs}, record, record]
        ):
            with self.assertRaises(ValueError):
                MODEL.collect_run_timings(1, expected_shards=2)


if __name__ == "__main__":
    unittest.main()
