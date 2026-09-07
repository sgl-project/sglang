"""Regression tests for Qwen3 MoE expert-location dispatch.

The ordinary (non-DeepEP) Qwen3 MoE path must translate logical router choices
to the configured physical expert locations before executing the experts.  A
non-trivial initial placement otherwise loads each rank's physical slots using
the placement map while routing tokens as if the slots were still trivial.

This test exercises the real forward method with tiny CPU-only fake modules; it
does not require model weights, CUDA, or a distributed process group.
"""

import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Gate(nn.Module):
    def forward(self, hidden_states):
        return torch.zeros(hidden_states.shape[0], 4), None


class _TopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.dispatch_info = None

    def forward(
        self,
        hidden_states,
        router_logits,
        *,
        expert_location_dispatch_info=None,
    ):
        self.dispatch_info = expert_location_dispatch_info
        return object()


class _Experts(nn.Module):
    def forward(self, hidden_states, topk_output):
        return hidden_states


class TestQwen3MoeExpertLocation(CustomTestCase):
    def _make_block(self):
        block = Qwen3MoeSparseMoeBlock.__new__(Qwen3MoeSparseMoeBlock)
        nn.Module.__init__(block)
        block.layer_id = 7
        block.ep_size = 1
        block.tp_size = 1
        block.gate = _Gate()
        block.topk = _TopK()
        block.experts = _Experts()
        return block

    def test_forward_normal_passes_expert_location_dispatch_info(self):
        block = self._make_block()
        dispatch_info = object()
        with (
            mock.patch(
                "sglang.srt.models.qwen3_moe.get_global_expert_location_metadata",
                return_value=object(),
            ),
            mock.patch(
                "sglang.srt.models.qwen3_moe.ExpertLocationDispatchInfo.init_new",
                return_value=dispatch_info,
            ) as init_dispatch,
        ):
            output = block.forward_normal(torch.ones(2, 8))

        init_dispatch.assert_called_once_with(layer_id=7)
        self.assertIs(block.topk.dispatch_info, dispatch_info)
        self.assertEqual(output.shape, (2, 8))

    def test_forward_normal_skips_dispatch_without_location_metadata(self):
        block = self._make_block()
        with (
            mock.patch(
                "sglang.srt.models.qwen3_moe.get_global_expert_location_metadata",
                return_value=None,
            ),
            mock.patch(
                "sglang.srt.models.qwen3_moe.ExpertLocationDispatchInfo.init_new"
            ) as init_dispatch,
        ):
            output = block.forward_normal(torch.ones(2, 8))

        init_dispatch.assert_not_called()
        self.assertIsNone(block.topk.dispatch_info)
        self.assertEqual(output.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
