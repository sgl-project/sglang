import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDeepseekV2MoESkipSharedExperts(CustomTestCase):
    def _make_moe(self):
        moe = DeepseekV2MoE.__new__(DeepseekV2MoE)
        nn.Module.__init__(moe)
        moe._enable_a2a_moe = False
        moe.layer_id = 0
        moe.alt_stream = object()
        moe.num_fused_shared_experts = 0
        moe._can_dual_stream_graph = Mock(return_value=True)
        moe.forward_normal = Mock(return_value="normal")
        moe.forward_normal_dual_stream = Mock(return_value="dual_stream")
        return moe

    def _forward(self, moe, *, skip_shared_experts):
        hidden_states = torch.ones(1, 1)
        with (
            patch(
                "sglang.srt.layers.moe.mega_moe.should_use_mega_moe",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.get_is_capture_mode",
                return_value=True,
            ),
            patch(
                "sglang.srt.models.deepseek_v2.get_forward",
                return_value=SimpleNamespace(
                    fuse_mlp_allreduce=False,
                    mlp_reduce_scatter=False,
                ),
            ),
            patch(
                "sglang.srt.models.deepseek_v2.dsv2_flashinfer_moe_dual_stream_graph",
                return_value="graph",
            ) as graph_forward,
        ):
            result = moe.forward(
                hidden_states,
                skip_shared_experts=skip_shared_experts,
            )
        return result, graph_forward

    def test_skip_shared_experts_bypasses_graph_paths(self):
        moe = self._make_moe()

        result, graph_forward = self._forward(moe, skip_shared_experts=True)

        self.assertEqual(result, "normal")
        graph_forward.assert_not_called()
        moe.forward_normal_dual_stream.assert_not_called()
        moe.forward_normal.assert_called_once()
        self.assertTrue(moe.forward_normal.call_args.kwargs["skip_shared_experts"])

    def test_graph_path_is_unchanged_when_shared_experts_are_enabled(self):
        moe = self._make_moe()

        result, graph_forward = self._forward(moe, skip_shared_experts=False)

        self.assertEqual(result, "graph")
        graph_forward.assert_called_once()
        moe.forward_normal.assert_not_called()
        moe.forward_normal_dual_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
