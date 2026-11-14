"""CPU-only unit tests for JIT CUDA home resolution (wheel-nvcc installs)."""

import os
import tempfile
import unittest
from unittest import mock

from sglang.kernels.jit.utils.compile import toolchain
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestCudaHomeResolution(CustomTestCase):
    def _env(self, **overrides):
        env = {
            k: v for k, v in os.environ.items() if k not in ("CUDA_HOME", "CUDA_PATH")
        }
        env.update(overrides)
        return mock.patch.dict(os.environ, env, clear=True)

    def test_cuda_home_wins(self):
        with self._env(CUDA_HOME="/opt/cuda-a", CUDA_PATH="/opt/cuda-b"):
            self.assertEqual(toolchain.cuda_home(), "/opt/cuda-a")

    def test_cuda_path_used_when_cuda_home_unset(self):
        with self._env(CUDA_PATH="/opt/cuda-b"):
            self.assertEqual(toolchain.cuda_home(), "/opt/cuda-b")

    def test_nvcc_on_path_gives_its_root(self):
        with tempfile.TemporaryDirectory() as root:
            nvcc = os.path.join(root, "bin", "nvcc")
            os.makedirs(os.path.dirname(nvcc))
            with open(nvcc, "w") as f:
                f.write("")
            with (
                self._env(),
                mock.patch.object(toolchain.shutil, "which", return_value=nvcc),
            ):
                self.assertEqual(toolchain.cuda_home(), root)

    def test_falls_back_to_default(self):
        with (
            self._env(),
            mock.patch.object(toolchain.shutil, "which", return_value=None),
        ):
            self.assertEqual(toolchain.cuda_home(), "/usr/local/cuda")


if __name__ == "__main__":
    unittest.main()
