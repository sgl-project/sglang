"""Regressions for attributing CI failures from completed pytest attempts."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts/ci_monitor/ci_failures_analysis.py"
)
_SPEC = importlib.util.spec_from_file_location("ci_failures_analysis", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_SESSION = "================ test session starts ================\n"
_SERVER_ARGS = "server_args: model_path='test-model'\n"


def _failed_session(path):
    return (
        _SESSION + "================ short test summary info ================\n"
        f"FAILED {path}::test_case\n"
        "================ 1 failed in 1.00s ================\n"
    )


_PASSED_SESSION = _SESSION + "================ 1 passed in 1.00s ================\n"
_RETRY_MARKER = (
    "Starting pytest attempt 2/7: retry failed items for 1 assigned item(s)\n"
)
_TAIL = """================ Pytest Tail Summary ================
Attempt 1 failure summary:
  FAILED test/test_old.py::test_old
====================================================
"""


class TestCIFailuresAnalysis(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.analyzer = _MODULE.SGLangFailuresAnalyzer("")
        self.addCleanup(self.analyzer.session.close)

    def _failed_paths(self, logs):
        summary = self.analyzer.parse_test_summary(logs)
        return (
            [test["full_path"] for test in summary["failed_tests"]] if summary else []
        )

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
            "SUBFAILED(dtype=torch.float32) test/test_sub.py::TestSub::test_sub - Error",
            "================ 3 failed, 2 errors in 1.00s ================",
        ]
        for prefix in (
            "2026-10-27T01:41:23.8215231Z ",
            "test-job\tUNKNOWN STEP\t2026-10-27T01:41:23.8215231Z ",
        ):
            with self.subTest(prefix=prefix):
                logs = "\n".join(f"{prefix}\x1b[31m{line}\x1b[0m" for line in lines)
                self.assertEqual(
                    self._failed_paths(logs),
                    [
                        "test/a/test_values.py",
                        "test/b/test_values.py",
                        "test/test_collection.py",
                        "test/test_sub.py",
                    ],
                )

    def test_retry_replaces_previous_attempt(self):
        initial = _failed_session("test/test_old.py")
        for logs, expected in (
            (initial + _RETRY_MARKER + _PASSED_SESSION + _TAIL, []),
            (
                initial + _RETRY_MARKER + _failed_session("test/test_new.py") + _TAIL,
                ["test/test_new.py"],
            ),
            (initial + _TAIL + _RETRY_MARKER + _PASSED_SESSION, ["test/test_old.py"]),
        ):
            with self.subTest(logs=logs):
                self.assertEqual(self._failed_paths(logs), expected)

    def test_real_runner_retry_and_plain_sequential_sessions(self):
        """Parse real output, so a changed retry marker or pytest format fails here."""
        program = (
            "import subprocess, sys\n"
            "from sglang.multimodal_gen.test.runner.pytest_runner import run_pytest\n"
            "run_pytest(['test_recovers.py'])\n"
            "sys.stdout.flush()\n"
            "for name in ('test_fails.py', 'test_passes.py'):\n"
            "    subprocess.run([sys.executable, '-m', 'pytest', name])\n"
        )
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "test_recovers.py").write_text(
                "from pathlib import Path\n"
                "def test_slow():\n"
                "    marker = Path(__file__).with_suffix('.attempt')\n"
                "    if not marker.exists():\n"
                "        marker.touch()\n"
                "        assert False, '[performance] Validation failed'\n"
            )
            (Path(root) / "test_fails.py").write_text(
                "def test_x():\n    assert False\n"
            )
            (Path(root) / "test_passes.py").write_text("def test_x():\n    pass\n")
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=root,
                env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                capture_output=True,
                text=True,
                timeout=120,
            )
        self.assertEqual(self._failed_paths(result.stdout), ["test_fails.py"])

    def test_unfinished_session_keeps_earlier_failures(self):
        summary = self.analyzer.parse_test_summary(
            _failed_session("test/test_old.py")
            + _SESSION
            + "test/test_current.py::test_current\n"
            + _SERVER_ARGS
        )
        self.assertTrue(summary["incomplete"])
        self.assertEqual(
            [test["full_path"] for test in summary["failed_tests"]],
            ["test/test_old.py", "test/test_current.py"],
        )

    def test_passing_session_has_no_summary(self):
        """An empty summary would hide the monitor's missing-summary diagnostic."""
        self.assertIsNone(self.analyzer.parse_test_summary(_PASSED_SESSION))

    def test_hang_after_finished_session_is_inferred(self):
        """A unittest file prints no session header, so a pytest result can precede it."""
        summary = self.analyzer.parse_test_summary(
            _PASSED_SESSION
            + "Begin (1/2):\npython3 /repo/test_hang.py\n"
            + _SERVER_ARGS
        )
        self.assertTrue(summary["incomplete"])
        self.assertEqual(summary["failed_tests"][0]["test_file"], "test_hang.py")

    def test_nested_quiet_run_is_not_a_failure(self):
        """test_sequential_server_requests.py runs `pytest -q` inside its session."""
        summary = self.analyzer.parse_test_summary(
            _SESSION
            + "test/test_outer.py::test_nested\n"
            + "================ short test summary info ================\n"
            + "FAILED test_generated.py::test_generated[failed]\n"
            + "2 failed, 2 passed, 4 warnings in 22.69s\n"
            + "PASSED\n"
            + "test/test_outer.py::test_after_nested\n"
            + _SERVER_ARGS
        )
        self.assertTrue(summary["incomplete"])
        self.assertEqual(
            [test["full_path"] for test in summary["failed_tests"]],
            ["test/test_outer.py"],
        )

    def test_quiet_run_outside_a_session_is_a_failure(self):
        """Simulator jobs run `pytest -q`, which prints no session header."""
        self.assertEqual(
            self._failed_paths(
                "================ short test summary info ================\n"
                "FAILED test/test_quiet.py::test_case - AssertionError\n"
                "1 failed, 1 passed in 1.00s\n"
            ),
            ["test/test_quiet.py"],
        )


if __name__ == "__main__":
    unittest.main()
