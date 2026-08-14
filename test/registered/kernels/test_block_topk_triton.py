"""
Unit tests for block_topk_triton kernel (block-level expert routing).

Verifies that the Triton implementation matches a pure PyTorch reference
across various configurations of num_tokens, num_experts, expert_capacity,
block_size, and top_k.
"""

import unittest

import torch

from sglang.srt.models.llada2 import block_topk_triton
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-large-amd")


def block_topk_reference(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    block_size: int,
    expert_capacity: int,
    top_k: int,
):
    """Independent PyTorch reference for block-level expert routing.

    Selection uses the unmodified corrected score. Stable argsort defines the
    exact-tie contract as lower expert ID first, without copying the Triton
    kernel's threshold/cumsum implementation. Routing weights are unbiased
    sigmoid scores normalized in FP32.
    """
    num_tokens, num_experts = router_logits.shape
    device = router_logits.device
    if num_tokens == 0:
        return (
            torch.empty((0, top_k), dtype=torch.float32, device=device),
            torch.empty((0, top_k), dtype=torch.int32, device=device),
        )

    base_scores = torch.sigmoid(router_logits.float())
    corrected_scores = base_scores + correction_bias.float().unsqueeze(0)
    num_blocks = (num_tokens + block_size - 1) // block_size
    padded_tokens = num_blocks * block_size
    if padded_tokens != num_tokens:
        corrected_scores_padded = torch.nn.functional.pad(
            corrected_scores,
            (0, 0, 0, padded_tokens - num_tokens),
            value=float("-inf"),
        )
    else:
        corrected_scores_padded = corrected_scores

    block_scores = corrected_scores_padded.view(
        num_blocks, block_size, num_experts
    ).amax(dim=1)
    allowed_ids = torch.argsort(block_scores, dim=-1, descending=True, stable=True)[
        :, :expert_capacity
    ]
    allowed = torch.zeros((num_blocks, num_experts), dtype=torch.bool, device=device)
    allowed.scatter_(1, allowed_ids, True)
    token_allowed = allowed.repeat_interleave(block_size, dim=0)[:num_tokens]

    token_scores = corrected_scores.masked_fill(~token_allowed, float("-inf"))
    topk_ids = torch.argsort(token_scores, dim=-1, descending=True, stable=True)[
        :, :top_k
    ]
    selected_base_scores = base_scores.gather(1, topk_ids)
    if top_k > 1:
        weight_sum = selected_base_scores.sum(dim=-1, keepdim=True)
        topk_weights = selected_base_scores / (weight_sum + 1e-20)
    else:
        topk_weights = selected_base_scores

    return topk_weights.float(), topk_ids.to(torch.int32)


class TestBlockTopkTriton(CustomTestCase):
    """Test block_topk_triton against PyTorch reference."""

    def _run_case(self, num_tokens, num_experts, expert_capacity, block_size, top_k):
        torch.manual_seed(42)
        device = "cuda"

        router_logits = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device=device
        )
        correction_bias = (
            torch.randn(num_experts, dtype=torch.float32, device=device) * 0.1
        )

        weights_triton, ids_triton = block_topk_triton(
            router_logits, correction_bias, block_size, expert_capacity, top_k
        )
        weights_ref, ids_ref = block_topk_reference(
            router_logits, correction_bias, block_size, expert_capacity, top_k
        )

        self.assertTrue(
            torch.equal(ids_triton, ids_ref),
            f"IDs mismatch for config ({num_tokens}, {num_experts}, "
            f"cap={expert_capacity}, bs={block_size}, k={top_k}).\n"
            f"  Triton: {ids_triton[:4]}\n"
            f"  Ref:    {ids_ref[:4]}",
        )
        self.assertEqual(weights_triton.dtype, torch.float32)
        torch.testing.assert_close(weights_triton, weights_ref, rtol=0.0, atol=2e-6)

    def test_basic_small(self):
        """32 tokens, 16 experts, capacity=8, block_size=8, top_k=2."""
        self._run_case(
            num_tokens=32, num_experts=16, expert_capacity=8, block_size=8, top_k=2
        )

    def test_single_expert_selection(self):
        """top_k=1 path."""
        self._run_case(
            num_tokens=32, num_experts=16, expert_capacity=8, block_size=8, top_k=1
        )

    def test_large_expert_count(self):
        """256 experts (production-like config)."""
        self._run_case(
            num_tokens=64, num_experts=256, expert_capacity=48, block_size=32, top_k=8
        )

    def test_capacity_equals_experts(self):
        """expert_capacity == num_experts means no routing restriction."""
        self._run_case(
            num_tokens=32, num_experts=16, expert_capacity=16, block_size=8, top_k=4
        )

    def test_tight_capacity(self):
        """Very tight capacity (only top_k experts allowed per block)."""
        self._run_case(
            num_tokens=64, num_experts=64, expert_capacity=6, block_size=16, top_k=4
        )

    def test_block_size_equals_one(self):
        """block_size=1 degenerates to per-token routing."""
        self._run_case(
            num_tokens=16, num_experts=32, expert_capacity=8, block_size=1, top_k=2
        )

    def test_many_tokens(self):
        """Stress test with more tokens."""
        self._run_case(
            num_tokens=256, num_experts=128, expert_capacity=32, block_size=32, top_k=4
        )

    def test_partial_block_and_non_power_of_two_experts(self):
        """Tail tokens and non-power-of-two expert counts are supported."""
        self._run_case(
            num_tokens=65,
            num_experts=100,
            expert_capacity=48,
            block_size=32,
            top_k=8,
        )

    def test_exact_ties_choose_lower_expert_id(self):
        logits = torch.zeros((33, 100), dtype=torch.float32, device="cuda")
        bias = torch.zeros((100,), dtype=torch.float32, device="cuda")
        weights, ids = block_topk_triton(logits, bias, 32, 48, 8)
        expected_ids = torch.arange(8, dtype=torch.int32, device="cuda").expand(33, -1)
        self.assertTrue(torch.equal(ids, expected_ids))
        torch.testing.assert_close(
            weights, torch.full_like(weights, 1.0 / 8), rtol=0.0, atol=0.0
        )

    def test_very_negative_logits_use_epsilon_normalization(self):
        """Very small sigmoid scores must not fall back to uniform weights."""
        logits = torch.full((32, 16), -90.0, dtype=torch.float32, device="cuda")
        bias = torch.zeros((16,), dtype=torch.float32, device="cuda")

        weights, ids = block_topk_triton(
            logits, bias, block_size=32, expert_capacity=16, top_k=8
        )

        selected_scores = torch.sigmoid(logits.float()).gather(1, ids.long())
        expected = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        torch.testing.assert_close(weights, expected, rtol=1e-5, atol=1e-25)

        # The old fallback returned 1 / top_k == 0.125 for this input.
        self.assertTrue(torch.all(weights < 1e-10))

    def test_close_unequal_scores_are_not_reordered(self):
        logits = torch.full((32, 16), -20.0, dtype=torch.float32, device="cuda")
        bias = torch.zeros((16,), dtype=torch.float32, device="cuda")
        logits[:, 0] = 0.0
        logits[:, 1] = 0.0
        bias[1] = 2e-7
        _, ids = block_topk_triton(logits, bias, 32, 16, 1)
        self.assertTrue(torch.equal(ids, torch.ones_like(ids)))

    def test_capacity_cutoff_near_tie(self):
        """Phase 1 and phase 2 must use the same sigmoid at the cutoff."""
        logits = torch.full((8, 4), -20.0, dtype=torch.float32, device="cuda")
        bias = torch.zeros((4,), dtype=torch.float32, device="cuda")

        # These adjacent FP32 logits exercise a capacity-1 boundary where
        # approximate sigmoid implementations can round differently. Both
        # routing phases must agree that expert 1 has the higher score.
        logits[:, 0] = 0.49701982736587524
        logits[:, 1] = 0.49701985716819763

        weights, ids = block_topk_triton(
            logits, bias, block_size=8, expert_capacity=1, top_k=1
        )

        self.assertTrue(torch.equal(ids, torch.ones_like(ids)))
        expected_weight = torch.sigmoid(logits[0, 1].cpu()).item()
        torch.testing.assert_close(
            weights,
            torch.full_like(weights, expected_weight),
            rtol=0.0,
            atol=1e-7,
        )

    def test_rejects_non_power_of_two_block_size(self):
        logits = torch.zeros((6, 4), dtype=torch.float32, device="cuda")
        bias = torch.zeros((4,), dtype=torch.float32, device="cuda")
        for block_size in (3, True):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(ValueError, "positive power of two"):
                    block_topk_triton(logits, bias, block_size, 2, 1)

    def test_empty_input_returns_fp32_weights(self):
        logits = torch.empty((0, 16), dtype=torch.bfloat16, device="cuda")
        bias = torch.zeros((16,), dtype=torch.float32, device="cuda")
        weights, ids = block_topk_triton(logits, bias, 32, 8, 2)
        self.assertEqual(weights.shape, (0, 2))
        self.assertEqual(ids.shape, (0, 2))
        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(ids.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
