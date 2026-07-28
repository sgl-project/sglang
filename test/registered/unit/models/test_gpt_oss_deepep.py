"""Unit tests for GPT-OSS DeepEP routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.models.gpt_oss import GptOssSparseMoeBlock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _DeepEPBackend:
    def is_deepep(self):
        return True


class _Router(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1, None


class _TopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_token_non_padded = None

    def forward(
        self,
        hidden_states,
        router_logits,
        *,
        num_token_non_padded,
        expert_location_dispatch_info,
    ):
        self.num_token_non_padded = num_token_non_padded
        return hidden_states + router_logits


class _Experts(nn.Module):
    def forward(self, hidden_states, topk_output):
        return hidden_states + topk_output


class TestGptOssDeepEPRouting(CustomTestCase):
    def test_forward_routes_non_padding_token_count_to_deepep_topk(self):
        block = GptOssSparseMoeBlock.__new__(GptOssSparseMoeBlock)
        nn.Module.__init__(block)
        block.layer_id = 3
        block.router = _Router()
        block.topk = _TopK()
        block.experts = _Experts()

        hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        forward_batch = SimpleNamespace(num_token_non_padded=1)

        with (
            patch(
                "sglang.srt.models.gpt_oss.get_moe_a2a_backend",
                return_value=_DeepEPBackend(),
            ),
            patch(
                "sglang.srt.models.gpt_oss.get_server_args",
                return_value=SimpleNamespace(dwdp_size=1),
            ),
            patch.object(ExpertLocationDispatchInfo, "init_new", return_value=None),
        ):
            output = block(hidden_states, forward_batch)

        self.assertEqual(block.topk.num_token_non_padded, 1)
        torch.testing.assert_close(
            output,
            torch.tensor([[4.0, 7.0], [10.0, 13.0]]),
        )


if __name__ == "__main__":
    unittest.main()
