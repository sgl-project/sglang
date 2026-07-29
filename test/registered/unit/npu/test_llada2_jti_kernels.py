from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="stage-a-unit-test-npu")

import unittest

import torch

from sglang.srt.hardware_backend.npu.norm.argmax_softmax_prob import (
    argmax_softmax_prob_fused,
    scrub_argmax_fused,
)
from sglang.srt.models.llada2 import block_topk_cann_hybrid


def _block_topk_reference(
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
        result_ids.append(selected_ids)
        result_weights.append(
            selected_scores / selected_scores.sum(dim=-1, keepdim=True)
        )

    return torch.cat(result_weights), torch.cat(result_ids).to(torch.int32)


class TestLLaDA2JTIKernels(unittest.TestCase):
    def setUp(self):
        torch.npu.set_device(0)

    def test_argmax_softmax_probability(self):
        torch.manual_seed(0)
        logits = torch.randn(32, 4097, dtype=torch.bfloat16, device="npu")

        token_ids, probabilities = argmax_softmax_prob_fused(logits)
        expected_ids = logits.argmax(dim=-1)
        expected_probabilities = (
            logits.float().softmax(dim=-1).gather(1, expected_ids[:, None]).squeeze(1)
        )

        torch.testing.assert_close(token_ids.cpu(), expected_ids.cpu())
        torch.testing.assert_close(
            probabilities.cpu(),
            expected_probabilities.cpu(),
            rtol=2e-3,
            atol=2e-4,
        )

    def test_scrub_argmax(self):
        torch.manual_seed(1)
        delete_token_id = 17
        split_token_id = 29
        logits = torch.randn(32, 4097, dtype=torch.bfloat16, device="npu")
        logits[:, delete_token_id] = 100
        logits[::2, split_token_id] = 101

        token_ids = scrub_argmax_fused(
            logits,
            delete_token_id,
            split_token_id,
        )
        reference = logits.clone()
        reference[:, delete_token_id] = float("-inf")
        reference[:, split_token_id] = float("-inf")

        torch.testing.assert_close(token_ids.cpu(), reference.argmax(dim=-1).cpu())

    def test_block_topk_cann_matches_reference(self):
        torch.manual_seed(2)
        num_tokens = 63
        num_experts = 256
        block_size = 32
        expert_capacity = 48
        top_k = 8
        logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16)
        bias = torch.randn(num_experts, dtype=torch.float32) * 0.1

        expected_weights, expected_ids = _block_topk_reference(
            logits,
            bias,
            block_size,
            expert_capacity,
            top_k,
        )
        weights, ids = block_topk_cann_hybrid(
            logits.to("npu"),
            bias.to("npu"),
            block_size,
            expert_capacity,
            top_k,
        )

        torch.testing.assert_close(ids.cpu(), expected_ids)
        torch.testing.assert_close(
            weights.float().cpu(),
            expected_weights,
            rtol=5e-3,
            atol=5e-4,
        )


if __name__ == "__main__":
    unittest.main()
