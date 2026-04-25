import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, suite="stage-b-test-1-gpu-small")


class GateStub(nn.Module):
    def __init__(self, router_logits: torch.Tensor):
        super().__init__()
        self.router_logits = router_logits

    def forward(self, hidden_states: torch.Tensor):
        return self.router_logits.clone(), None


class ExpertsStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_hidden_states = None
        self.last_topk_output = None

    def forward(self, hidden_states: torch.Tensor, topk_output):
        self.last_hidden_states = hidden_states
        self.last_topk_output = topk_output
        return torch.zeros_like(hidden_states)


class TestQwen3MoeOnPolicy(unittest.TestCase):
    def _make_block(self, router_logits: torch.Tensor):
        block = object.__new__(Qwen3MoeSparseMoeBlock)
        nn.Module.__init__(block)
        block.top_k = 2
        block.tp_size = 1
        block.ep_size = 1
        block.gate = GateStub(router_logits)
        block.experts = ExpertsStub()
        block.topk = MagicMock(name="topk")
        return block

    def test_on_policy_uses_manual_softmax_topk_routing(self):
        hidden_states = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
        )
        router_logits = torch.tensor(
            [[2.0, 1.0, -1.0], [0.5, 0.2, 0.1]], dtype=torch.float32
        )
        block = self._make_block(router_logits)

        with patch(
            "sglang.srt.models.qwen3_moe.get_global_server_args",
            return_value=SimpleNamespace(rl_on_policy_target="fsdp"),
        ):
            out = block.forward_normal(hidden_states)

        self.assertEqual(out.shape, hidden_states.shape)
        block.topk.assert_not_called()
        self.assertIsInstance(block.experts.last_topk_output, StandardTopKOutput)

        expected_weights = torch.softmax(router_logits, dim=1, dtype=torch.float)
        expected_weights, expected_ids = torch.topk(
            expected_weights, block.top_k, dim=-1
        )
        expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
        expected_weights = expected_weights.to(hidden_states.dtype)

        torch.testing.assert_close(
            block.experts.last_topk_output.topk_weights, expected_weights
        )
        torch.testing.assert_close(
            block.experts.last_topk_output.topk_ids, expected_ids
        )
        torch.testing.assert_close(
            block.experts.last_topk_output.router_logits, router_logits
        )

    def test_non_on_policy_delegates_to_topk_module(self):
        hidden_states = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32
        )
        router_logits = torch.tensor(
            [[2.0, 1.0, -1.0], [0.5, 0.2, 0.1]], dtype=torch.float32
        )
        block = self._make_block(router_logits)
        sentinel_topk_output = object()
        block.topk.return_value = sentinel_topk_output

        with patch(
            "sglang.srt.models.qwen3_moe.get_global_server_args",
            return_value=SimpleNamespace(rl_on_policy_target=None),
        ):
            out = block.forward_normal(hidden_states)

        self.assertEqual(out.shape, hidden_states.shape)
        block.topk.assert_called_once()
        called_hidden_states, called_router_logits = block.topk.call_args.args
        torch.testing.assert_close(called_hidden_states, hidden_states)
        torch.testing.assert_close(called_router_logits, router_logits)
        self.assertIs(block.experts.last_topk_output, sentinel_topk_output)


if __name__ == "__main__":
    unittest.main()
