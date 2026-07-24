"""
Unit tests for block_topk_triton kernel (block-level expert routing).

Verifies that the Triton implementation matches a pure PyTorch reference
across various configurations of num_tokens, num_experts, expert_capacity,
block_size, and top_k.
"""

import unittest

import torch

from sglang.srt.models.llada2 import block_topk_triton


def block_topk_reference(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    block_size: int,
    expert_capacity: int,
    top_k: int,
):
    """Pure PyTorch reference implementation of block-level expert routing."""
    num_tokens, num_experts = router_logits.shape
    assert num_tokens % block_size == 0
    device = router_logits.device

    base_scores = torch.sigmoid(router_logits.float())
    routing_scores = base_scores + correction_bias.float()

    num_blocks = num_tokens // block_size
    topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.zeros((num_tokens, top_k), dtype=router_logits.dtype, device=device)

    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        block_scores = routing_scores[start:end]

        # Phase 1: block-level allowed mask
        block_expert_scores = block_scores.max(dim=0).values
        # Tie-break by expert index
        expert_indices = torch.arange(num_experts, device=device, dtype=torch.float32)
        combined_block = block_expert_scores - expert_indices * 3e-7
        _, sorted_idx = combined_block.sort(descending=True)
        allowed_experts = sorted_idx[:expert_capacity]
        allowed_mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
        allowed_mask[allowed_experts] = True

        # Phase 2: per-token top-k within allowed set
        for t in range(start, end):
            token_scores = routing_scores[t].clone()
            combined_token = token_scores - expert_indices * 3e-7
            combined_token[~allowed_mask] = -10000.0
            _, token_sorted_idx = combined_token.sort(descending=True)
            selected = token_sorted_idx[:top_k]
            topk_ids[t] = selected.int()

            # Weights: normalized base_scores (without bias)
            selected_base_scores = base_scores[t, selected]
            if top_k > 1:
                m = selected_base_scores.max()
                if m > 1e-30:
                    scaled = selected_base_scores / m
                    topk_weights[t] = (scaled / scaled.sum()).to(router_logits.dtype)
                else:
                    topk_weights[t] = 1.0 / top_k
            else:
                topk_weights[t] = selected_base_scores.to(router_logits.dtype)

    return topk_weights, topk_ids


class TestBlockTopkTriton(unittest.TestCase):
    """Test block_topk_triton against PyTorch reference."""

    def _run_case(self, num_tokens, num_experts, expert_capacity, block_size, top_k):
        torch.manual_seed(42)
        device = "cuda"

        router_logits = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device=device
        )
        correction_bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.1

        weights_triton, ids_triton = block_topk_triton(
            router_logits, correction_bias, block_size, expert_capacity, top_k
        )
        weights_ref, ids_ref = block_topk_reference(
            router_logits, correction_bias, block_size, expert_capacity, top_k
        )

        # The Triton kernel returns IDs sorted by expert index (ascending),
        # while the reference sorts by score (descending). We compare the
        # selected expert SET per token, ignoring order.
        ids_triton_sorted, _ = ids_triton.sort(dim=1)
        ids_ref_sorted, _ = ids_ref.sort(dim=1)
        self.assertTrue(
            torch.equal(ids_triton_sorted, ids_ref_sorted),
            f"IDs mismatch (as sets) for config ({num_tokens}, {num_experts}, "
            f"cap={expert_capacity}, bs={block_size}, k={top_k}).\n"
            f"  Triton sorted: {ids_triton_sorted[:4]}\n"
            f"  Ref sorted:    {ids_ref_sorted[:4]}",
        )

        # Verify weights: for each token, gather triton weights by expert id
        # and compare against reference weights gathered the same way.
        # This is order-independent.
        base_scores = torch.sigmoid(router_logits.float())
        for t in range(num_tokens):
            triton_experts = ids_triton[t]
            ref_experts = ids_ref[t]
            # Both should select same experts (verified above), so weights
            # for the same expert should match.
            triton_weight_map = {
                int(triton_experts[k]): float(weights_triton[t, k])
                for k in range(top_k)
            }
            ref_weight_map = {
                int(ref_experts[k]): float(weights_ref[t, k])
                for k in range(top_k)
            }
            for expert_id in triton_weight_map:
                self.assertAlmostEqual(
                    triton_weight_map[expert_id],
                    ref_weight_map[expert_id],
                    places=3,
                    msg=f"Weight mismatch at token={t}, expert={expert_id} "
                    f"for config ({num_tokens}, {num_experts}, "
                    f"cap={expert_capacity}, bs={block_size}, k={top_k})",
                )

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
            num_tokens=64, num_experts=256, expert_capacity=48, block_size=32, top_k=6
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


if __name__ == "__main__":
    unittest.main()
