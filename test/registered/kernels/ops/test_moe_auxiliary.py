"""Correctness tests for the small decode-only MoE helper kernels."""

import unittest

import torch

from sglang.kernels.ops.moe.moe_align_single_token import (
    moe_align_single_token,
)
from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


class TestMoeAuxiliaryKernels(CustomTestCase):
    def test_align_single_token(self):
        for topk, block_size in ((8, 16), (16, 64)):
            with self.subTest(topk=topk, block_size=block_size):
                expert_ids = torch.randperm(896, device="cuda")[:topk].to(torch.int32)
                topk_ids = expert_ids.unsqueeze(0)

                sorted_ids, sorted_experts, num_post = moe_align_single_token(
                    topk_ids, block_size
                )

                order = torch.argsort(expert_ids)
                expected_sorted_ids = torch.full(
                    (topk * block_size,),
                    topk,
                    dtype=torch.int32,
                    device="cuda",
                )
                expected_sorted_ids[::block_size] = order.to(torch.int32)

                self.assertTrue(
                    torch.equal(sorted_experts, expert_ids[order]),
                    "expert ids must be sorted in ascending order",
                )
                self.assertTrue(torch.equal(sorted_ids, expected_sorted_ids))
                self.assertEqual(num_post.item(), topk * block_size)

    def test_topk_sum(self):
        for num_tokens, topk, hidden_size in (
            (1, 8, 128),
            (4, 16, 128),
            (2, 16, 7168),
        ):
            with self.subTest(
                num_tokens=num_tokens, topk=topk, hidden_size=hidden_size
            ):
                torch.manual_seed(num_tokens * 1000 + topk + hidden_size)
                x = torch.randn(
                    num_tokens,
                    topk,
                    hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                out = torch.empty(
                    num_tokens,
                    hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                )

                actual = moe_topk_sum(x, out)
                expected = x.float().sum(dim=1).to(torch.bfloat16)

                self.assertIs(actual, out)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
