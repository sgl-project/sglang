"""Unit tests for ci_auto_bisect log parsing (no GitHub or Claude API calls).

python -m unittest discover -s scripts/ci_monitor -p 'test_ci_auto_bisect.py'
"""

import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _mod in ("anthropic", "requests"):
    try:
        __import__(_mod)
    except ImportError:
        sys.modules[_mod] = types.ModuleType(_mod)

import ci_auto_bisect as bisect  # noqa: E402

# Excerpt of XPU diffusion job 135735999144: 7 pytest attempts, 2 cases fail all.
FIXTURE = os.path.join(HERE, "fixtures", "xpu_diffusion_pytest_job.txt")
NODE = "test_server_1_gpu.py::TestDiffusionServerOneGpu::test_diffusion_generation[{}]"


def _read_fixture() -> str:
    with open(FIXTURE, encoding="utf-8") as f:
        return f.read()


class TestPytestSummary(unittest.TestCase):
    def test_xpu_diffusion_log_keeps_node_ids(self):
        summary = bisect.parse_test_summary(_read_fixture())
        self.assertEqual(
            [t["test_id"] for t in summary["failed_tests"]],
            [
                NODE.format("flux_2_klein_image_t2i"),
                NODE.format("flux_2_klein_base_image_t2i"),
            ],
        )
        self.assertEqual(
            {t["test_file"] for t in summary["failed_tests"]}, {"test_server_1_gpu.py"}
        )
        # Counts come from the final attempt, not attempt 1 (2 failed, 2 passed)
        self.assertEqual((summary["passed"], summary["total"]), (0, 2))

    def test_item_recovered_on_retry_is_not_reported(self):
        logs = "\n".join(
            [
                "Starting pytest attempt 1/7: initial pass for 2 assigned item(s)",
                "=========================== short test summary info ============================",
                "FAILED sglang/multimodal_gen/test/server/test_a.py::test_x[flaky]",
                "FAILED sglang/multimodal_gen/test/server/test_a.py::test_x[broken] - AssertionError: boom",
                "======= 2 failed, 3 passed in 10.00s =======",
                "Starting pytest attempt 2/7: retry failed items for 2 assigned item(s)",
                "=========================== short test summary info ============================",
                "FAILED sglang/multimodal_gen/test/server/test_a.py::test_x[broken] - AssertionError: boom",
                "======= 1 failed, 1 passed in 5.00s =======",
                "================================ Pytest Tail Summary ================================",
                "Starting pytest attempt 1/7: initial pass for 1 assigned item(s)",
                "======= 1 passed in 2.00s =======",
                "================================ Pytest Tail Summary ================================",
                "Starting pytest attempt 1/7: initial pass for 1 assigned item(s)",
                "=========================== short test summary info ============================",
                "ERROR sglang/multimodal_gen/test/unit/test_b.py::test_y",
                "======= 1 error in 1.00s =======",
                "================================ Pytest Tail Summary ================================",
            ]
        )
        summary = bisect.parse_test_summary(logs)
        self.assertEqual(
            [t["test_id"] for t in summary["failed_tests"]],
            ["test_a.py::test_x[broken]", "test_b.py::test_y"],
        )
        self.assertEqual((summary["passed"], summary["total"]), (2, 4))

    def test_unittest_summary_unchanged(self):
        logs = "\n".join(
            [
                "Test Summary: 3/4 passed",
                "✗ FAILED:",
                "  registered/core/test_foo.py (exit code 1)",
                "=" * 20,
            ]
        )
        summary = bisect.parse_test_summary(logs)
        self.assertEqual((summary["passed"], summary["total"]), (3, 4))
        self.assertEqual(summary["failed_tests"][0]["test_file"], "test_foo.py")
        self.assertNotIn("test_id", summary["failed_tests"][0])


class TestPytestErrorSignature(unittest.TestCase):
    def test_error_excerpt_is_per_case(self):
        logs = _read_fixture()
        base = bisect.extract_error_signature(
            logs, "test_server_1_gpu.py", NODE.format("flux_2_klein_base_image_t2i")
        )
        self.assertTrue(
            base.startswith(
                "_ TestDiffusionServerOneGpu.test_diffusion_generation[flux_2_klein_base_image_t2i] _"
            )
        )
        self.assertIn("AssertionError: Validation failed", base)
        self.assertNotIn("[flux_2_klein_image_t2i]", base)


class TestFailureTargetLabel(unittest.TestCase):
    def test_label_prefers_node_id(self):
        kwargs = dict(
            job_name="multimodal-gen-test-1-gpu-xpu",
            test_file="test_server_1_gpu.py",
            hardware="Intel XPU",
            current_streak=2,
            first_failure_sha="",
            last_failure_sha="",
            first_failure_date="",
            last_failure_date="",
            first_failure_job_url="",
            last_failure_job_url="",
            first_failure_job_id=None,
            last_failure_job_id=None,
        )
        self.assertEqual(
            bisect.FailureTarget(**kwargs).test_label, "test_server_1_gpu.py"
        )
        node = NODE.format("flux_2_klein_image_t2i")
        self.assertEqual(bisect.FailureTarget(test_id=node, **kwargs).test_label, node)


if __name__ == "__main__":
    unittest.main()
