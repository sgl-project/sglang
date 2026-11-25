"""CPU tests for the vector-width gate that routes act-and-mul to the JIT kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.arch._cuda import CudaArch
from sglang.srt.layers import activation
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _cuda_arch(major: int, minor: int) -> CudaArch:
    arch = CudaArch()
    # Pre-fill the cached device properties so no GPU is queried.
    arch.__dict__["props"] = SimpleNamespace(major=major, minor=minor)
    return arch


class TestJitActVectorGate(CustomTestCase):
    def _supported(self, capability, hidden_size, dtype):
        out = torch.empty(2, hidden_size, dtype=dtype)
        with (
            patch.object(
                activation, "current_arch", return_value=_cuda_arch(*capability)
            ),
            patch.object(activation, "_jit_act_max_vec_bytes", None),
        ):
            return activation._jit_act_supported(out)

    def test_hidden_size_must_fill_whole_vectors_of_the_arch(self):
        """The JIT kernel loads 16-byte vectors before Blackwell and 32-byte
        vectors from SM100 on; a hidden size that is not a whole number of
        vectors must go to the sgl_kernel implementation."""
        # 1368 is DeepSeek-V2-Lite's dense intermediate size at TP8.
        cases = [
            ((8, 0), 1368, torch.bfloat16, True),
            ((9, 0), 1368, torch.bfloat16, True),
            ((9, 0), 1364, torch.bfloat16, False),
            ((9, 0), 1364, torch.float32, True),
            ((10, 0), 1368, torch.bfloat16, False),
            ((10, 3), 1368, torch.bfloat16, False),
            ((12, 0), 1368, torch.bfloat16, False),
            ((10, 0), 1376, torch.bfloat16, True),
            ((10, 0), 1364, torch.float32, False),
            ((12, 1), 1368, torch.float32, True),
        ]
        for capability, hidden_size, dtype, expected in cases:
            with self.subTest(capability=capability, hidden=hidden_size, dtype=dtype):
                self.assertEqual(
                    self._supported(capability, hidden_size, dtype), expected
                )


if __name__ == "__main__":
    unittest.main()
