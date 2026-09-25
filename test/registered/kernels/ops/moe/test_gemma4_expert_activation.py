"""CUDA numerical checks for DiffusionGemma's inference optimizations."""

import unittest

import torch

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGemma4SamplingCUDA(unittest.TestCase):
    def test_expert_activation_matches_checkpoint(self):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts_impl,
        )

        # The runner uses FP16 or BF16 intermediates, even for FP32 inputs.
        for dtype in (torch.float16, torch.bfloat16):
            hidden = (torch.arange(-32, 32, device="cuda") / 8).to(dtype)
            hidden = hidden.repeat(3, 1)
            eye = torch.eye(64, device="cuda", dtype=dtype)
            w1 = torch.stack((torch.cat((eye, eye)), torch.cat((-eye, eye))))
            w2 = torch.stack((eye, eye))
            ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32).repeat(3, 1)
            weights = torch.tensor([[0.25, 0.75]], device="cuda").repeat(3, 1)
            for activation, approximate in (("gelu_tanh", "tanh"), ("gelu", "none")):
                with (
                    self.subTest(dtype=dtype, activation=activation),
                    get_context().override_server_args(
                        enable_deterministic_inference=False
                    ),
                    get_parallel().override(tp_group=None),
                ):
                    actual = fused_experts_impl(
                        hidden,
                        w1,
                        w2,
                        weights,
                        ids,
                        activation=activation,
                        filter_expert=False,
                    )
                    x = hidden.float()
                    left = (
                        torch.nn.functional.gelu(x, approximate=approximate) * x
                    ).to(dtype)
                    right = (
                        torch.nn.functional.gelu(-x, approximate=approximate) * x
                    ).to(dtype)
                    expected = (left * 0.25 + right * 0.75).to(dtype)
                    tolerance = 2 * torch.finfo(dtype).eps
                    torch.testing.assert_close(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )


if __name__ == "__main__":
    unittest.main()
