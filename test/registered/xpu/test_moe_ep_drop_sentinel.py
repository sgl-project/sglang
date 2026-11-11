"""
python3 -m unittest test_moe_ep_drop_sentinel.py
"""

import unittest

import torch
from sgl_kernel import fused_experts

from sglang.srt.layers.moe.utils import route_dropped_experts_to_local
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"
HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS = 512, 256, 16
# bf16; measured max error is about 0.5% of the peak
RELATIVE_TOLERANCE = 2e-2


def expert_weights(
    with_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    w13 = (
        torch.randn(
            NUM_EXPERTS,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        / 16
    )
    w2 = (
        torch.randn(
            NUM_EXPERTS,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device=DEVICE,
            dtype=torch.bfloat16,
        )
        / 16
    )
    if not with_bias:
        return w13, w2, None, None
    b13 = (
        torch.randn(
            NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, device=DEVICE, dtype=torch.bfloat16
        )
        / 4
    )
    b2 = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, device=DEVICE, dtype=torch.bfloat16) / 4
    return w13, w2, b13, b2


def routing(num_tokens: int, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (num_tokens, top_k), device=DEVICE, dtype=torch.int32
    )
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=DEVICE), dim=-1)
    topk_ids = torch.where(
        torch.rand(num_tokens, top_k, device=DEVICE) < 0.5, -1, topk_ids
    )
    return topk_weights, topk_ids


def reference_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    b13: torch.Tensor | None,
    b2: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """fp32 reference that skips the -1 slots."""
    output = torch.zeros(
        hidden_states.shape[0], HIDDEN_SIZE, device=DEVICE, dtype=torch.float32
    )
    for token in range(topk_ids.shape[0]):
        for slot in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, slot])
            if expert < 0:
                continue
            gate_up = hidden_states[token].float() @ w13[expert].float().T
            if b13 is not None:
                gate_up = gate_up + b13[expert].float()
            activation = (
                torch.nn.functional.silu(gate_up[:INTERMEDIATE_SIZE])
                * gate_up[INTERMEDIATE_SIZE:]
            )
            expert_output = activation @ w2[expert].float().T
            if b2 is not None:
                expert_output = expert_output + b2[expert].float()
            output[token] += float(topk_weights[token, slot]) * expert_output
    return output


class TestXpuMoeEpDropSentinel(CustomTestCase):
    """Regression: a -1 in fused_experts loses the device."""

    def assert_close_to_reference(
        self, kernel_output: torch.Tensor, reference: torch.Tensor
    ) -> None:
        self.assertLessEqual(
            (kernel_output - reference).abs().max().item(),
            RELATIVE_TOLERANCE * reference.abs().max().item(),
        )

    def test_fused_experts_matches_reference_that_skips_dropped_slots(self) -> None:
        torch.manual_seed(0)
        hidden_states = (
            torch.randn(64, HIDDEN_SIZE, device=DEVICE, dtype=torch.bfloat16) / 4
        )
        w13, w2, _, _ = expert_weights(with_bias=False)
        topk_weights, topk_ids = routing(num_tokens=64, top_k=4)
        kernel_weights, kernel_ids = route_dropped_experts_to_local(
            topk_weights=topk_weights, topk_ids=topk_ids
        )
        kernel_output = fused_experts(
            hidden_states, w13, w2, kernel_weights, kernel_ids, activation="silu"
        )
        torch.xpu.synchronize()
        self.assert_close_to_reference(
            kernel_output.float(),
            reference_moe(hidden_states, w13, w2, None, None, topk_weights, topk_ids),
        )

    def test_weight_zero_slot_adds_nothing_with_biased_experts(self) -> None:
        # router weight on the input, as unquant forward_xpu applies it
        torch.manual_seed(1)
        hidden_states = (
            torch.randn(32, HIDDEN_SIZE, device=DEVICE, dtype=torch.bfloat16) / 4
        )
        w13, w2, b13, b2 = expert_weights(with_bias=True)
        topk_weights, topk_ids = routing(num_tokens=32, top_k=1)
        scaled_hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        kernel_weights, kernel_ids = route_dropped_experts_to_local(
            topk_weights=torch.ones_like(topk_weights), topk_ids=topk_ids
        )
        kernel_output = fused_experts(
            scaled_hidden_states,
            w13,
            w2,
            kernel_weights,
            kernel_ids,
            b1=b13,
            b2=b2,
            activation="silu",
        )
        torch.xpu.synchronize()
        reference = reference_moe(
            scaled_hidden_states,
            w13,
            w2,
            b13,
            b2,
            torch.ones_like(topk_weights),
            topk_ids,
        )
        self.assert_close_to_reference(kernel_output.float(), reference)


if __name__ == "__main__":
    unittest.main()
