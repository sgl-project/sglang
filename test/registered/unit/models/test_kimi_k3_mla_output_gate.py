import unittest

import torch

from sglang.kernels.ops.kimi_k3.mla_output_gate import covered
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestKimiK3MlaOutputGate(CustomTestCase):
    def test_cpu_tensors_use_fallback(self):
        x = torch.zeros(2, 8, dtype=torch.bfloat16)
        gate = torch.zeros_like(x)

        self.assertFalse(covered(x, gate))


if __name__ == "__main__":
    unittest.main()
