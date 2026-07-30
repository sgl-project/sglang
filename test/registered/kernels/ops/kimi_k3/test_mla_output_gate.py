import unittest

import torch

from sglang.kernels.ops.kimi_k3.mla_output_gate import (
    covered,
    kimi_k3_mla_output_gate,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestKimiK3MlaOutputGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_matches_two_step_bfloat16_reference_bit_exactly(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        for shape in ((1, 7168), (5, 12, 128)):
            with self.subTest(shape=shape):
                x = torch.randn(
                    shape, generator=generator, device="cuda", dtype=torch.bfloat16
                )
                gate = torch.randn(
                    shape, generator=generator, device="cuda", dtype=torch.bfloat16
                )

                self.assertTrue(covered(x, gate))
                expected = x * torch.sigmoid(gate).to(torch.bfloat16)
                actual = kimi_k3_mla_output_gate(x, gate)

                self.assertTrue(torch.equal(actual, expected))

    def test_contract_rejects_unsupported_layouts(self):
        x = torch.empty((2, 16), device="cuda", dtype=torch.bfloat16)
        self.assertFalse(covered(x, x[:, ::2]))
        self.assertFalse(covered(x.float(), x.float()))
        self.assertFalse(covered(x[:, :7], x[:, :7]))


if __name__ == "__main__":
    unittest.main()
