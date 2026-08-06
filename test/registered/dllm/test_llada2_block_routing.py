from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

import sglang.srt.models.llada2 as llada2


class TestLLaDA2BlockRouting(unittest.TestCase):
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
