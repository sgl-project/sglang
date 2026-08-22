import unittest

import torch

from sglang.kernels.ops.llada2.block_topk_npu import block_topk_npu
from sglang.kernels.ops.llada2.indel_npu import scrub_argmax_fused
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="stage-b-test-1-npu-a3", nightly=False)
register_npu_ci(est_time=30, suite="nightly-1-npu-a3", nightly=True)


def _block_topk_reference(
    router_logits,
    correction_bias,
    block_size,
    expert_capacity,
    top_k,
):
    num_tokens, num_experts = router_logits.shape
    base_scores = torch.sigmoid(router_logits.float())
    routing_scores = base_scores + correction_bias.float()
    result_ids = []
    result_weights = []

    for start in range(0, num_tokens, block_size):
        block_scores = routing_scores[start : start + block_size]
        capacity_ids = torch.argsort(
            block_scores.max(dim=0).values,
            descending=True,
            stable=True,
        )[:expert_capacity]
        capacity_ids = capacity_ids.sort().values
        token_scores = block_scores[:, capacity_ids]
        local_ids = torch.argsort(
            token_scores,
            dim=-1,
            descending=True,
            stable=True,
        )[:, :top_k]
        selected_ids = capacity_ids[local_ids]
        selected_scores = base_scores[start : start + block_size].gather(
            1, selected_ids
        )
        result_ids.append(selected_ids)
        if top_k > 1:
            score_sum = selected_scores.sum(dim=-1, keepdim=True)
            selected_scores = torch.where(
                score_sum > 1e-30,
                selected_scores / score_sum.clamp_min(1e-30),
                torch.full_like(selected_scores, 1.0 / top_k),
            )
        result_weights.append(selected_scores.float())

    return torch.cat(result_weights), torch.cat(result_ids).to(torch.int32)


class TestLLaDA2NPUKernels(unittest.TestCase):
    def test_block_topk_matches_reference(self):
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
        weights, ids = block_topk_npu(
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

    def test_block_topk_zero_score_fallback(self):
        logits = torch.full(
            (32, 16),
            float("-inf"),
            dtype=torch.bfloat16,
            device="npu",
        )
        bias = torch.zeros(16, dtype=torch.float32, device="npu")

        weights, ids = block_topk_npu(
            logits,
            bias,
            block_size=32,
            expert_capacity=8,
            top_k=4,
        )

        torch.testing.assert_close(
            weights.float().cpu(),
            torch.full((32, 4), 0.25),
        )
        self.assertEqual(ids.dtype, torch.int32)

    def test_block_topk_ordering(self):
        cases = (
            ([0.0, 2e-7, 0.0, 0.0], 2, 4, 1, [[1], [1]]),
            ([0.0, 0.0, 0.0, 0.0], 2, 4, 2, [[0, 1], [0, 1]]),
            ([0.3, 0.2, 0.4], 1, 3, 2, [[2, 0]]),
        )
        for bias_values, num_tokens, capacity, top_k, expected_ids in cases:
            with self.subTest(bias=bias_values, top_k=top_k):
                logits = torch.zeros(
                    (num_tokens, len(bias_values)),
                    dtype=torch.float32,
                    device="npu",
                )
                bias = torch.tensor(bias_values, device="npu")

                _, ids = block_topk_npu(
                    logits,
                    bias,
                    block_size=num_tokens,
                    expert_capacity=capacity,
                    top_k=top_k,
                )

                expected = torch.tensor(expected_ids, dtype=torch.int32)
                torch.testing.assert_close(ids.cpu(), expected)

    def test_block_topk_single_expert_keeps_raw_sigmoid_weight(self):
        logits = torch.tensor([[0.0, -1.0]], device="npu")
        bias = torch.zeros(2, dtype=torch.float32, device="npu")

        weights, ids = block_topk_npu(
            logits,
            bias,
            block_size=1,
            expert_capacity=2,
            top_k=1,
        )

        torch.testing.assert_close(ids.cpu(), torch.zeros((1, 1), dtype=torch.int32))
        torch.testing.assert_close(weights.cpu(), torch.full((1, 1), 0.5))
        self.assertEqual(weights.dtype, torch.float32)

    def test_scrub_argmax(self):
        torch.manual_seed(1)
        mask_token_id = 7
        delete_token_id = 17
        split_token_id = 29
        logits = torch.randn(32, 4097, dtype=torch.bfloat16, device="npu")
        logits[:, mask_token_id] = 102
        logits[:, delete_token_id] = 100
        logits[::2, split_token_id] = 101

        fallback_ids, scrub_ids = scrub_argmax_fused(
            logits,
            mask_token_id,
            delete_token_id,
            split_token_id,
        )
        fallback_reference = logits.clone()
        fallback_reference[:, mask_token_id] = float("-inf")
        scrub_reference = fallback_reference.clone()
        scrub_reference[:, delete_token_id] = float("-inf")
        scrub_reference[:, split_token_id] = float("-inf")

        torch.testing.assert_close(
            fallback_ids.cpu(), fallback_reference.argmax(dim=-1).cpu()
        )
        torch.testing.assert_close(
            scrub_ids.cpu(), scrub_reference.argmax(dim=-1).cpu()
        )


if __name__ == "__main__":
    unittest.main()
