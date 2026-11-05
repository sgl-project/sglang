"""Each KV storage module imports on its own, in a fresh interpreter.

A test process that has already imported the package hides an import cycle
that only some first imports close, so each module here is the first import of
its own interpreter.
"""

import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

MODULES = (
    "sglang.srt.mem_cache.layout.fused_draft",
    "sglang.srt.mem_cache.kv",
    "sglang.srt.mem_cache.kv.decl",
    "sglang.srt.mem_cache.kv.layout.base",
    "sglang.srt.mem_cache.kv.plan",
    "sglang.srt.mem_cache.kv.build",
)


class TestFreshProcessImports(CustomTestCase):
    def test_each_module_imports_first(self):
        for module in MODULES:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-c", f"import {module}"],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_package_exports_host_pool_options(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from sglang.srt.mem_cache.kv import HostPoolOptions; "
                "from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions as H; "
                "assert HostPoolOptions is H",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
