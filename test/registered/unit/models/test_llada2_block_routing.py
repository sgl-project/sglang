from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.llada2 as llada2


def test_router_keeps_fp32_only_for_block_routing():
    config = SimpleNamespace(
        num_experts=4,
        hidden_size=8,
        moe_router_enable_expert_bias=False,
    )
    hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)

    regular_gate = llada2.LLaDA2MoeGate(config, params_dtype=torch.float32)
    block_gate = llada2.LLaDA2MoeGate(
        SimpleNamespace(**vars(config), expert_capacity=3),
        params_dtype=torch.float32,
    )

    assert regular_gate(hidden_states).dtype == torch.bfloat16
    assert block_gate(hidden_states).dtype == torch.float32


def test_npu_folds_routed_scaling_factor_into_block_topk_weights():
    block = llada2.LLaDA2MoeSparseMoeBlock.__new__(llada2.LLaDA2MoeSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.correction_bias = torch.zeros(4)
    block.block_size = 2
    block.expert_capacity = 4
    block.top_k = 2
    block.routed_scaling_factor = 2.5

    backend_weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
    backend_ids = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
    with patch.object(
        llada2,
        "block_topk_npu",
        return_value=(backend_weights, backend_ids),
    ):
        weights, ids = block._block_topk(torch.zeros((2, 4)))

    torch.testing.assert_close(weights, backend_weights * 2.5)
    torch.testing.assert_close(ids, backend_ids)
