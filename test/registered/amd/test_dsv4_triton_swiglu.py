"""HIP unfiltered MoE must implement DSV4's asymmetric pre-SiLU clamp."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(is_hip(), "requires HIP")
class TestDsv4TritonSwiglu(unittest.TestCase):
    def test_clamp_before_silu_without_clamping_negative_gate(self):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts_impl,
        )
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        size = 128
        x = torch.zeros(4, size, device="cuda", dtype=torch.bfloat16)
        x[:, 0] = 1
        w1 = torch.zeros(1, 2 * size, size, device="cuda", dtype=torch.bfloat16)
        # Forcing large signed gate/up values distinguishes all clamp variants.
        gate = torch.tensor([-20.0, -12.0, 2.0, 20.0], device="cuda").repeat(size // 4)
        up = torch.tensor([20.0, -20.0, 20.0, -20.0], device="cuda").repeat(size // 4)
        w1[0, :size, 0] = gate
        w1[0, size:, 0] = up
        w2 = torch.eye(size, device="cuda", dtype=torch.bfloat16).unsqueeze(0)
        ids = torch.zeros(4, 1, device="cuda", dtype=torch.int32)
        weights = torch.ones(4, 1, device="cuda", dtype=torch.float32)
        expected = (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).to(torch.bfloat16)
        with patch(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.get_tp_group",
            return_value=None,
        ):
            actual = fused_experts_impl(
                x, w1, w2, weights, ids, filter_expert=False, swiglu_limit=10
            )
        torch.testing.assert_close(
            actual, expected.expand_as(actual), rtol=0.01, atol=1e-8
        )
        # A symmetric clamp would increase the negative-gate result by >1000x.
        self.assertLess(abs(actual[0, 0].item()), 1e-5)
        self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
