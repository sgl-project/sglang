import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    _clamp_swiglu_inputs_,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFusedMoeSwigluClamp(CustomTestCase):
    def test_clamps_gate_and_up_with_the_deepseek_contract(self):
        inputs = torch.tensor([[-20.0, 20.0, -20.0, 20.0], [3.0, 11.0, -4.0, 5.0]])

        _clamp_swiglu_inputs_(inputs, limit=10.0)

        torch.testing.assert_close(
            inputs,
            torch.tensor([[-20.0, 10.0, -10.0, 10.0], [3.0, 10.0, -4.0, 5.0]]),
        )

    def test_clamped_inputs_match_reference_swiglu(self):
        inputs = torch.tensor([[-12.0, 14.0, -13.0, 15.0]])

        _clamp_swiglu_inputs_(inputs, limit=10.0)
        output = F.silu(inputs[..., :2]) * inputs[..., 2:]

        expected_gate = torch.tensor([[-12.0, 10.0]])
        expected_up = torch.tensor([[-10.0, 10.0]])
        torch.testing.assert_close(output, F.silu(expected_gate) * expected_up)


if __name__ == "__main__":
    unittest.main()
