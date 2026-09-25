"""Representative parity coverage for the lightweight Kimi-K3 prerequisites."""

import unittest

import torch

from sglang.kernels.ops.elementwise import add3
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896

TOPK = 16

NOPE_DIM = 512

ROPE_DIM = 64

MLA_DIM = NOPE_DIM + ROPE_DIM

MLA_PAGES = 256


class TestKimiK3PrerequisiteOps(CustomTestCase):
    def test_add3_bit_exact(self):
        torch.manual_seed(0)
        tensors = [
            torch.randn(9, 112, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        actual = add3.add3(*tensors, prefetch_bc=True)
        expected = (tensors[0] + tensors[1]) + tensors[2]
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
