"""Unit tests for GPT-OSS expert-parallel checkpoint slicing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.models.gpt_oss import (
    GptOssSparseMoeBlock,
    TinyGemmLinear,
    _narrow_fused_moe_ep_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGptOssEpWeightLoading(CustomTestCase):
    def _make_fused_moe_shell(self) -> FusedMoE:
        """Build the minimum loader surface without allocating expert weights."""
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.__dict__["use_padded_loading"] = True
        layer.quant_config = None
        layer.use_presharded_weights = False
        return layer

    @patch("sglang.srt.models.gpt_oss.get_parallel")
    def test_global_expert_weights_are_sliced_for_ep_rank(self, get_parallel):
        """Global checkpoint experts must be narrowed to the current EP rank."""
        get_parallel.return_value = SimpleNamespace(moe_ep_size=4, moe_ep_rank=2)
        weight = torch.arange(128 * 2).view(128, 2)

        actual = _narrow_fused_moe_ep_weight(weight, num_experts=128)

        torch.testing.assert_close(actual, weight[64:96])

    @patch("sglang.srt.models.gpt_oss.get_parallel")
    def test_ep_one_keeps_global_expert_weights(self, get_parallel):
        """EP1 must preserve the checkpoint tensor without an unnecessary copy."""
        get_parallel.return_value = SimpleNamespace(moe_ep_size=1, moe_ep_rank=0)
        weight = torch.arange(128 * 2).view(128, 2)

        actual = _narrow_fused_moe_ep_weight(weight, num_experts=128)

        self.assertIs(actual, weight)

    @patch("sglang.srt.models.gpt_oss.get_parallel")
    def test_local_expert_weights_are_not_sliced_again(self, get_parallel):
        """Already-local checkpoint experts must not be narrowed a second time."""
        get_parallel.return_value = SimpleNamespace(moe_ep_size=4, moe_ep_rank=2)
        weight = torch.arange(32 * 2).view(32, 2)

        actual = _narrow_fused_moe_ep_weight(weight, num_experts=128)

        self.assertIs(actual, weight)

    @patch("sglang.srt.models.gpt_oss.get_parallel")
    def test_invalid_expert_dimension_is_rejected(self, get_parallel):
        """A checkpoint must contain either all experts or this rank's experts."""
        get_parallel.return_value = SimpleNamespace(moe_ep_size=4, moe_ep_rank=2)
        weight = torch.arange(64 * 2).view(64, 2)

        with self.assertRaisesRegex(
            ValueError, "Expected 128 global or 32 local experts, got 64"
        ):
            _narrow_fused_moe_ep_weight(weight, num_experts=128)

    def test_w2_bias_loading_uses_its_last_dimension(self):
        """Fused down-projection biases must ignore the weight shard dimension."""
        layer = self._make_fused_moe_shell()
        expert_data = torch.empty(2, 4)
        loaded_weight = torch.arange(8, dtype=torch.float32).view(2, 4)

        layer._load_w2(
            expert_data=expert_data,
            shard_dim=2,
            shard_id="w2",
            loaded_weight=loaded_weight,
            tp_rank=0,
            is_bias=True,
        )

        torch.testing.assert_close(expert_data, loaded_weight)

    def test_tinygemm_linear_accepts_empty_idle_batch(self):
        """DP ranks without work must not launch tinygemm with zero M."""
        layer = TinyGemmLinear.__new__(TinyGemmLinear)
        torch.nn.Module.__init__(layer)
        layer.output_size = 128
        layer._use_tinygemm = True
        x = torch.empty((0, 64), dtype=torch.bfloat16)

        output, output_bias = layer(x)

        self.assertEqual(output.shape, (0, 128))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertIsNone(output_bias)

    @patch(
        "sglang.srt.models.gpt_oss.should_skip_post_experts_all_reduce",
        create=True,
        return_value=True,
    )
    @patch("sglang.srt.models.gpt_oss.tensor_model_parallel_all_reduce")
    @patch(
        "sglang.srt.models.gpt_oss.get_forward",
        return_value=SimpleNamespace(fuse_mlp_allreduce=False),
    )
    @patch(
        "sglang.srt.models.gpt_oss.is_in_tc_piecewise_cuda_graph",
        return_value=False,
    )
    def test_flashinfer_a2a_output_is_not_all_reduced(
        self, _is_piecewise, _get_forward, all_reduce, _should_skip
    ):
        """FlashInfer combine already returns the complete source-rank output."""
        block = GptOssSparseMoeBlock.__new__(GptOssSparseMoeBlock)
        torch.nn.Module.__init__(block)
        block.tp_size = 4
        block.hidden_size = 3
        block.__dict__["router"] = Mock(
            return_value=(torch.zeros((2, 4), dtype=torch.float32), None)
        )
        topk_output = object()
        block.__dict__["topk"] = Mock(return_value=topk_output)
        block.__dict__["experts"] = Mock(side_effect=lambda x, _: x + 1)
        hidden_states = torch.zeros((2, 3), dtype=torch.bfloat16)

        actual = block.forward_normal(hidden_states)

        torch.testing.assert_close(actual, hidden_states + 1)
        all_reduce.assert_not_called()

    @patch.object(fused_moe_layer.UnquantizedFusedMoEMethod, "create_moe_runner")
    @patch.object(fused_moe_layer.UnquantizedFusedMoEMethod, "create_weights")
    @patch.object(fused_moe_layer, "create_moe_dispatcher")
    @patch.object(
        fused_moe_layer,
        "get_moe_a2a_backend",
        return_value=SimpleNamespace(is_ascend_fuseep=lambda: False),
    )
    @patch.object(
        fused_moe_layer,
        "get_server_args",
        return_value=SimpleNamespace(
            ep_join_mode="none", moe_runner_backend="flashinfer_trtllm_routed"
        ),
    )
    @patch.object(
        fused_moe_layer,
        "get_parallel",
        return_value=SimpleNamespace(
            moe_ep_size=1, moe_ep_rank=0, moe_tp_size=1, moe_tp_rank=0
        ),
    )
    @patch.object(
        fused_moe_layer, "has_per_rank_fused_shared_slots", return_value=False
    )
    @patch.object(
        fused_moe_layer, "create_kt_config_from_server_args", return_value=None
    )
    @patch.object(
        fused_moe_layer,
        "get_moe_runner_backend",
        return_value=MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
    )
    def test_flashinfer_constructor_keeps_logical_intermediate_size(self, *_):
        """Kernel allocation padding must not overwrite the logical MoE size."""
        layer = FusedMoE(
            num_experts=1,
            hidden_size=1,
            intermediate_size=2880,
            layer_id=0,
            params_dtype=torch.float32,
        )

        self.assertEqual(layer.intermediate_size_per_partition_unpadded, 2880)
        self.assertEqual(layer.intermediate_size_per_partition, 2944)


if __name__ == "__main__":
    unittest.main()
