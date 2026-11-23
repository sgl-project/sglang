import subprocess
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_SCRIPT = Path(__file__).resolve().parents[4] / "benchmark/prefill_only/bench_rerank.py"


class TestPrefillRerankBenchmark(CustomTestCase):
    def test_cli_imports_tokenizer(self):
        """The benchmark CLI must start when the tokenizer module is reorganized."""
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--num-requests", result.stdout)


if __name__ == "__main__":
    unittest.main()
