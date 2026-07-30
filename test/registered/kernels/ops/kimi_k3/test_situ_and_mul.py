import unittest

import torch

from sglang.kernels.ops.kimi_k3 import situ_and_mul
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _reference(
    gate_up: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    gate_out = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate_out * up).to(torch.bfloat16)


class TestKimiK3SituAndMul(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_matches_reference_for_both_up_branches(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        hidden_size = 1024
        storage = torch.randn(
            (7, 2 * hidden_size + 16),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate_up = storage[:, : 2 * hidden_size]
        self.assertFalse(gate_up.is_contiguous())

        for linear_beta in (None, 25.0):
            with self.subTest(linear_beta=linear_beta):
                output = torch.empty(
                    (gate_up.shape[0], hidden_size),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                returned = situ_and_mul(gate_up, output, 4.0, linear_beta)
                expected = _reference(gate_up, 4.0, linear_beta)

                self.assertIs(returned, output)
                torch.testing.assert_close(
                    returned.float(), expected.float(), rtol=2e-2, atol=4e-2
                )


if __name__ == "__main__":
    unittest.main()
