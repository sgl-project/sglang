"""The serving client must not import server model code for a constant."""

import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestServingClientImport(unittest.TestCase):
    def test_import_does_not_load_disaggregation_server(self):
        script = """
import importlib.abc
import sys

class NoServerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in (
            "sglang.srt.disaggregation.utils",
            "sglang.srt.configs.model_config",
        ):
            raise AssertionError(f"Benchmark imported server module: {fullname}")

sys.meta_path.insert(0, NoServerImports())
from sglang.benchmark import serving
from sglang.srt.disaggregation.constants import FAKE_BOOTSTRAP_HOST
assert serving.FAKE_BOOTSTRAP_HOST == FAKE_BOOTSTRAP_HOST == "2.2.2.2"
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
