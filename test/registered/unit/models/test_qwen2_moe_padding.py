import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.qwen2_moe as qwen2_moe
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingTopK:
    def __init__(self):
        self.num_token_non_padded = None

    def __call__(
        self,
        hidden_states,
        router_logits,
        *,
        num_token_non_padded=None,
    ):
        self.num_token_non_padded = num_token_non_padded
        return object()


class TestQwen2MoeCudaGraphPadding(unittest.TestCase):
    def test_standard_moe_path_masks_and_zeroes_padded_rows(self):
        num_tokens, hidden_size, num_valid = 8, 4, 5
        hidden_states = torch.ones(num_tokens, hidden_size)
        num_token_non_padded = torch.tensor(num_valid, dtype=torch.int32)
        forward_batch = SimpleNamespace(
            num_token_non_padded=num_token_non_padded,
        )

        topk = _RecordingTopK()
        block = SimpleNamespace(
            alt_stream=None,
            enable_shared_expert_fusion=False,
            experts=lambda hidden_states, topk_output: torch.full_like(
                hidden_states, 2.0
            ),
            gate=lambda hidden_states: (
                torch.zeros(hidden_states.shape[0], 2),
                None,
            ),
            shared_expert_gate=None,
            topk=topk,
            tp_size=1,
        )
        block._forward_router_experts = MethodType(
            Qwen2MoeSparseMoeBlock._forward_router_experts,
            block,
        )
        block._forward_shared_experts = (
            lambda hidden_states, apply_gate: torch.ones_like(hidden_states)
        )

        backend = SimpleNamespace(is_deepep=lambda: False)
        with patch.object(qwen2_moe, "get_moe_a2a_backend", return_value=backend):
            output = Qwen2MoeSparseMoeBlock.forward(
                block,
                hidden_states,
                forward_batch,
            )

        self.assertIs(topk.num_token_non_padded, num_token_non_padded)
        torch.testing.assert_close(
            output[:num_valid],
            torch.full((num_valid, hidden_size), 3.0),
        )
        self.assertTrue(
            torch.equal(output[num_valid:], torch.zeros_like(output[num_valid:]))
        )


if __name__ == "__main__":
    unittest.main()
