"""Compatibility tests for the macOS Triton import stub."""

import platform
import subprocess
import sys
import unittest

import torch
from packaging.version import Version

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


@unittest.skipUnless(
    sys.platform == "darwin"
    and platform.machine() == "arm64"
    and torch.backends.mps.is_available()
    and Version(torch.__version__) >= Version("2.13.0"),
    "requires Torch >= 2.13 on Apple silicon",
)
class TestMpsTritonStub(unittest.TestCase):
    def test_torch_inductor_imports_after_sglang_installs_stub(self):
        script = """
import sglang
from torch._inductor.runtime.triton_heuristics import _KernelType
assert _KernelType is not None
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
