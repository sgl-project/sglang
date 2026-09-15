"""Run the MoE workload regressions on CPU, without model weights or NPUs."""

import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestMoeTPEPBenchmark(unittest.TestCase):
    def test_distributed_workload(self):
        benchmark = (
            Path(__file__).resolve().parents[4] / "benchmark" / "kernels" / "moe_tp_ep"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(benchmark),
                "-p",
                "test_workload.py",
                "-v",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
