import unittest

import torch

from sglang.kernels.ops.attention.mla_output_gate import (
    covered,
    kimi_k3_mla_output_gate,
)
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestMlaOutputGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("Kimi K3 compute kernels require SM100a+")

    def test_mla_output_gate(self):
        generator = torch.Generator(device="cuda").manual_seed(1)
        shape = (5, 12, 128)
        x = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )

        self.assertTrue(covered(x, gate))
        expected = x * torch.sigmoid(gate).to(torch.bfloat16)
        self.assertTrue(torch.equal(kimi_k3_mla_output_gate(x, gate), expected))


if __name__ == "__main__":
    unittest.main()
