"""Direct correctness coverage for the Kimi-K3 MoE-front fusion."""

import unittest

import torch

from sglang.kernels.ops.moe.moe_front import (
    NUM_EXPERTS,
    TOPK,
    fused_front,
    fused_front_covered,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestMoeFront(CustomTestCase):
    def test_coverage_contract(self):
        hidden = torch.empty((1, 7168), device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(
            (NUM_EXPERTS + 128, 7168), device="cuda", dtype=torch.bfloat16
        )
        bias = torch.empty(NUM_EXPERTS, device="cuda", dtype=torch.float32)

        self.assertTrue(fused_front_covered(hidden, weight, bias, TOPK, 128))
        self.assertFalse(fused_front_covered(hidden.float(), weight, bias, TOPK, 128))
        self.assertFalse(
            fused_front_covered(hidden, weight, bias.to(torch.bfloat16), TOPK, 128)
        )
        self.assertFalse(fused_front_covered(hidden, weight, bias, TOPK - 1, 128))
        self.assertFalse(fused_front_covered(hidden, weight, bias, TOPK, 126))

    def test_fused_front_matches_unfused_reference(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MoE-front JIT kernel requires SM100 or newer")

        torch.manual_seed(20260730)
        for num_tokens in (1, 4):
            with self.subTest(num_tokens=num_tokens):
                latent = 128
                hidden = (
                    torch.randn(num_tokens, 7168, device="cuda", dtype=torch.bfloat16)
                    / 32
                )
                weight = (
                    torch.randn(
                        NUM_EXPERTS + latent,
                        7168,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    / 32
                )
                bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.float32)

                weights, ids, routed = fused_front(
                    hidden,
                    weight,
                    bias,
                    latent,
                    renormalize=True,
                    routed_scaling_factor=2.5,
                    apply_routed_scaling_factor_on_output=True,
                )

                merged = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
                ref_weights, ref_ids = moe_fused_gate(
                    merged[:, :NUM_EXPERTS],
                    bias,
                    topk=TOPK,
                    scoring_func="sigmoid",
                    renormalize=True,
                    routed_scaling_factor=2.5,
                    apply_routed_scaling_factor_on_output=True,
                )
                actual_order = ids.argsort(dim=-1)
                ref_order = ref_ids.argsort(dim=-1)

                self.assertTrue(
                    torch.equal(
                        ids.gather(1, actual_order),
                        ref_ids.to(torch.int32).gather(1, ref_order),
                    )
                )
                torch.testing.assert_close(
                    weights.gather(1, actual_order),
                    ref_weights.gather(1, ref_order),
                    rtol=1e-6,
                    atol=0,
                )
                torch.testing.assert_close(
                    routed,
                    merged[:, NUM_EXPERTS:].to(torch.bfloat16),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
