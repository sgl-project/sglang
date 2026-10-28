"""Regressions for attributing CI failures from completed pytest attempts."""

import importlib.util
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts/ci_monitor/ci_failures_analysis.py"
)
_SPEC = importlib.util.spec_from_file_location("ci_failures_analysis", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TestCIFailuresAnalysis(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.analyzer = _MODULE.SGLangFailuresAnalyzer("")
        self.addCleanup(self.analyzer.session.close)

    def test_pytest_failures_override_last_running_test(self):
        """A passing loader must not replace the failures listed by pytest."""
        summary = self.analyzer.parse_test_summary(
            """
sglang/test/test_text_encoder_loader.py::TestLoader::test_quantized [10-27 01:41:06]
server_args: model_path='test-model'
PASSED
sglang/test/test_text_encoder_loader.py::TestLoader::test_restore PASSED
================ short test summary info ================
FAILED sglang/test/test_nomas.py::test_fused_qk_norm
FAILED sglang/test/test_nomas.py::test_fused_swiglu
FAILED sglang/test/test_nomas.py::test_growing_history
=== 3 failed, 1558 passed, 16 skipped, 64 warnings, 107 subtests passed in 154.07s ===
"""
        )
        self.assertEqual(
            summary,
            {
                "passed": 0,
                "total": 0,
                "failed_tests": [
                    {
                        "test_file": "test_nomas.py",
                        "full_path": "sglang/test/test_nomas.py",
                    }
                ],
            },
        )

    def test_timestamped_ansi_errors_and_parametrized_failures(self):
        lines = [
            "================ short test summary info ================",
            "FAILED test/a/test_values.py::TestValues::test_shape[32 - 64] - AssertionError",
            "FAILED test/a/test_values.py::TestValues::test_shape[128] - AssertionError",
            "ERROR test/b/test_values.py::TestValues::test_setup[False] - RuntimeError",
            "ERROR test/test_collection.py - ImportError",
            "================ 2 failed, 2 errors in 1.00s ================",
        ]
        for prefix in (
            "2026-10-27T01:41:23.8215231Z ",
            "test-job\tUNKNOWN STEP\t2026-10-27T01:41:23.8215231Z ",
        ):
            with self.subTest(prefix=prefix):
                logs = "\n".join(f"{prefix}\x1b[31m{line}\x1b[0m" for line in lines)
                summary = self.analyzer.parse_test_summary(logs)
                self.assertEqual(
                    summary["failed_tests"],
                    [
                        {
                            "test_file": path.rsplit("/", 1)[-1],
                            "full_path": path,
                        }
                        for path in (
                            "test/a/test_values.py",
                            "test/b/test_values.py",
                            "test/test_collection.py",
                        )
                    ],
                )

    def test_retry_uses_final_outcome_and_ignores_replayed_failure_tail(self):
        initial_attempt = """
================ test session starts ================
================ short test summary info ================
FAILED test/test_old.py::test_old
================ 1 failed in 1.00s ================
"""
        replay = """
================ Pytest Tail Summary ================
Attempt 1 failure summary:
  FAILED test/test_old.py::test_old
====================================================
"""
        for retry, expected_paths in (
            (
                "================ test session starts ================\n"
                "================ 1 passed in 1.00s ================\n",
                [],
            ),
            ("1 passed in 1.00s\n", []),
            (
                replay + "================ test session starts ================\n"
                "================ 1 passed in 1.00s ================\n",
                ["test/test_old.py"],
            ),
            (
                "================ test session starts ================\n"
                "================ 1 passed in 1.00s ================\n"
                + replay
                + "================ test session starts ================\n"
                "================ short test summary info ================\n"
                "ERROR test/test_collection.py - ImportError\n"
                "================ 1 error in 1.00s ================\n",
                ["test/test_collection.py"],
            ),
            (
                "================ test session starts ================\n"
                "================ short test summary info ================\n"
                "FAILED test/test_final.py::test_final\n"
                "================ 1 failed in 1.00s ================\n",
                ["test/test_final.py"],
            ),
        ):
            with self.subTest(retry=retry):
                summary = self.analyzer.parse_test_summary(
                    initial_attempt + retry + replay
                )
                self.assertEqual(
                    [test["full_path"] for test in summary["failed_tests"]],
                    expected_paths,
                )
                self.assertNotIn("incomplete", summary)

    def test_incomplete_retry_does_not_reuse_previous_attempt(self):
        summary = self.analyzer.parse_test_summary(
            """
================ test session starts ================
================ short test summary info ================
FAILED test/test_old.py::test_old
================ 1 failed in 1.00s ================
================ test session starts ================
test/test_current.py::test_current
server_args: model_path='test-model'
"""
        )
        self.assertTrue(summary["incomplete"])
        self.assertEqual(
            summary["failed_tests"][0]["full_path"], "test/test_current.py"
        )

    def test_incomplete_invocation_retains_earlier_failures(self):
        """An unfinished invocation must remain visible after a completed failure."""
        for current_path in ("test/test_current.py", "test/test_old.py"):
            with self.subTest(current_path=current_path):
                summary = self.analyzer.parse_test_summary(
                    f"""
================ test session starts ================
================ short test summary info ================
FAILED test/test_old.py::test_old
================ 1 failed in 1.00s ================
================ Pytest Tail Summary ================
Attempt 1 failure summary:
  FAILED test/test_old.py::test_old
====================================================
================ test session starts ================
{current_path}::test_current
server_args: model_path='test-model'
"""
                )
                expected_tests = [
                    {"test_file": "test_old.py", "full_path": "test/test_old.py"}
                ]
                if current_path != "test/test_old.py":
                    expected_tests.append(
                        {
                            "test_file": "test_current.py",
                            "full_path": current_path,
                            "context": "last_running",
                        }
                    )
                self.assertEqual(
                    summary,
                    {
                        "passed": 0,
                        "total": 0,
                        "failed_tests": expected_tests,
                        "incomplete": True,
                    },
                )

    def test_custom_summary_retains_file_counts_and_failure_list(self):
        summary = self.analyzer.parse_test_summary(
            """
================ short test summary info ================
FAILED test/test_earlier_attempt.py::test_retry
================ 1 failed in 1.00s ================
Test Summary: 7/8 passed
✗ FAILED:
  test/registered/test_final.py
====================================================
"""
        )
        self.assertEqual(
            summary,
            {
                "passed": 7,
                "total": 8,
                "failed_tests": [
                    {
                        "test_file": "test_final.py",
                        "full_path": "test/registered/test_final.py",
                    }
                ],
            },
        )
        self.assertIsNone(self.analyzer.parse_test_summary("runner setup failed"))


if __name__ == "__main__":
    unittest.main()
