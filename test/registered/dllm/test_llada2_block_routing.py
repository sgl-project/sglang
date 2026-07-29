from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

import sglang.srt.models.llada2 as llada2


def block_topk_reference(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    block_size: int,
    expert_capacity: int,
    top_k: int,
):
    num_tokens, num_experts = router_logits.shape
    base_scores = torch.sigmoid(router_logits.float())
    routing_scores = base_scores + correction_bias.float()
    tie_break = -torch.arange(num_experts, dtype=torch.float32) * 3e-7
    result_ids = []
    result_weights = []

    for start in range(0, num_tokens, block_size):
        block_scores = routing_scores[start : start + block_size]
        capacity_ids = (
            (block_scores.max(dim=0).values + tie_break).topk(expert_capacity).indices
        )
        token_scores = block_scores[:, capacity_ids] + tie_break[capacity_ids]
        local_ids = token_scores.topk(top_k, dim=-1).indices
        selected_ids = capacity_ids[local_ids]
        selected_scores = base_scores[start : start + block_size].gather(
            1, selected_ids
        )
        selected_weights = selected_scores / selected_scores.sum(dim=-1, keepdim=True)
        result_ids.append(selected_ids)
        result_weights.append(selected_weights)

    return torch.cat(result_weights), torch.cat(result_ids).to(torch.int32)


class TestLLaDA2BlockRouting(unittest.TestCase):
    def test_cann_hybrid_matches_reference(self):
        for num_tokens, num_experts, block_size, capacity, top_k in (
            (32, 16, 8, 8, 2),
            (63, 256, 32, 48, 8),
        ):
            torch.manual_seed(num_tokens)
            logits = torch.randn(num_tokens, num_experts)
            bias = torch.randn(num_experts) * 0.1

            weights, ids = llada2.block_topk_cann_hybrid(
                logits,
                bias,
                block_size,
                capacity,
                top_k,
            )
            expected_weights, expected_ids = block_topk_reference(
                logits,
                bias,
                block_size,
                capacity,
                top_k,
            )

            torch.testing.assert_close(ids, expected_ids)
            torch.testing.assert_close(weights, expected_weights)

    def test_npu_folds_routed_scaling_factor_into_block_topk_weights(self):
        block = llada2.LLaDA2MoeSparseMoeBlock.__new__(llada2.LLaDA2MoeSparseMoeBlock)
        torch.nn.Module.__init__(block)
        block.correction_bias = torch.zeros(4)
        block.block_size = 2
        block.expert_capacity = 4
        block.top_k = 2
        block._block_expert_tie_break = None
        block.routed_scaling_factor = 2.5

        backend_weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
        backend_ids = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)

        with (
            patch.object(llada2, "_is_npu", True),
            patch.object(
                llada2,
                "block_topk_cann_hybrid",
                return_value=(backend_weights, backend_ids),
            ),
        ):
            weights, ids = block._block_topk(torch.zeros((2, 4)))

        torch.testing.assert_close(weights, backend_weights * 2.5)
        torch.testing.assert_close(weights.sum(dim=-1), torch.full((2,), 2.5))
        torch.testing.assert_close(ids, backend_ids)

    def test_cuda_leaves_scaling_to_moe_runner(self):
        block = llada2.LLaDA2MoeSparseMoeBlock.__new__(llada2.LLaDA2MoeSparseMoeBlock)
        torch.nn.Module.__init__(block)
        block.correction_bias = torch.zeros(4)
        block.block_size = 2
        block.expert_capacity = 4
        block.top_k = 2
        block._block_expert_tie_break = None
        block.routed_scaling_factor = 2.5

        backend_weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
        backend_ids = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)

        with (
            patch.object(llada2, "_is_npu", False),
            patch.object(
                llada2,
                "block_topk_triton",
                return_value=(backend_weights, backend_ids),
            ),
        ):
            weights, ids = block._block_topk(torch.zeros((2, 4)))

        torch.testing.assert_close(weights, backend_weights)
        torch.testing.assert_close(ids, backend_ids)


if __name__ == "__main__":
    unittest.main()
