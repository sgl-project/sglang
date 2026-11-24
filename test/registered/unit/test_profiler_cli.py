"""Regression test for the `python -m sglang.profiler` entry point."""

import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestProfilerEntryPoint(CustomTestCase):
    def test_parser_takes_no_deprecated_argparse_parameters(self):
        """The parser must build without the BooleanOptionalAction parameters that
        argparse deprecated in Python 3.12 and rejects in 3.14."""
        result = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::DeprecationWarning:argparse",
                "-m",
                "sglang.profiler",
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile-by-stage", result.stdout)


if __name__ == "__main__":
    unittest.main()
