import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

# Match model initialization order for the NPU quantization package.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.layers.moe.fused_moe_triton import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w4a8_int8_moe import (
    ModelSlimW4A8Int8MoE,
)


class TestModelSlimW4A8Int8MoEBiasLayout(unittest.TestCase):
    def test_w2_scale_bias_uses_moe_tp_k_block_shards(self):
        for moe_tp_size, moe_ep_size in (
            (1, 1),
            (2, 1),
            (4, 1),
            (8, 1),
            (16, 1),
            (1, 16),
        ):
            global_tp_size = moe_tp_size * moe_ep_size
            for rank in sorted({0, moe_tp_size // 2, moe_tp_size - 1}):
                with self.subTest(
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                    rank=rank,
                ):
                    layer = torch.nn.Module()
                    layer.moe_tp_size = moe_tp_size
                    layer.moe_ep_size = moe_ep_size
                    w2 = ModelSlimW4A8Int8MoE(
                        quant_config={},
                        weight_prefix="w2",
                        tp_size=global_tp_size,
                    )
                    w2.create_weights(
                        layer,
                        num_experts=2,
                        hidden_size=32,
                        intermediate_size_per_partition=32 // moe_tp_size,
                    )

                    expected_shape = (2, 32, 16 // moe_tp_size)
                    self.assertEqual(tuple(layer.w2_scale_bias.shape), expected_shape)
                    self.assertEqual(
                        layer.w2_scale_bias.quant_method,
                        FusedMoeWeightScaleSupported.GROUP.value,
                    )
                    self.assertEqual(
                        layer.w2_weight_scale.quant_method,
                        FusedMoeWeightScaleSupported.CHANNEL.value,
                    )
                    self.assertEqual(
                        layer.w2_weight_offset.quant_method,
                        FusedMoeWeightScaleSupported.CHANNEL.value,
                    )

                    checkpoint_bias = torch.arange(32 * 16, dtype=torch.float32).view(
                        32, 16
                    )
                    loader = FusedMoE.__new__(FusedMoE)
                    torch.nn.Module.__init__(loader)
                    loader.quant_config = None
                    loader.use_padded_loading = False
                    loader.use_presharded_weights = False
                    loader.use_triton_kernels = False
                    loader.moe_tp_size = moe_tp_size
                    loader.moe_tp_rank = rank
                    loader._load_model_weight_or_group_weight_scale(
                        shard_dim=1,
                        expert_data=layer.w2_scale_bias.data[0],
                        shard_id="w2",
                        loaded_weight=checkpoint_bias,
                    )

                    shard_width = 16 // moe_tp_size
                    start = rank * shard_width
                    self.assertTrue(
                        torch.equal(
                            layer.w2_scale_bias[0],
                            checkpoint_bias[:, start : start + shard_width],
                        )
                    )

    def test_w13_scale_bias_keeps_single_k_block_and_group_loader_metadata(self):
        layer = torch.nn.Module()
        layer.moe_tp_size = 8
        w13 = ModelSlimW4A8Int8MoE(quant_config={}, weight_prefix="w13", tp_size=8)
        w13.create_weights(
            layer,
            num_experts=2,
            hidden_size=32,
            intermediate_size_per_partition=4,
        )

        self.assertEqual(tuple(layer.w13_scale_bias.shape), (2, 8, 1))
        self.assertEqual(
            layer.w13_scale_bias.quant_method,
            FusedMoeWeightScaleSupported.GROUP.value,
        )
        self.assertEqual(
            layer.w13_weight_scale.quant_method,
            FusedMoeWeightScaleSupported.CHANNEL.value,
        )

        loader = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(loader)
        loader.quant_config = None
        loader.use_padded_loading = False
        loader.use_presharded_weights = False
        loader.use_triton_kernels = False
        loader.moe_tp_size = 8
        loader.moe_tp_rank = 3
        loader.moe_runner_config = SimpleNamespace(is_gated=True)
        loader.quant_method = SimpleNamespace()
        gate_bias = torch.arange(32, dtype=torch.float32).view(32, 1)
        up_bias = gate_bias + 100
        expert_bias = layer.w13_scale_bias.data[0]
        loader._load_model_weight_or_group_weight_scale(
            shard_dim=0,
            expert_data=expert_bias,
            shard_id="w1",
            loaded_weight=gate_bias,
        )
        loader._load_model_weight_or_group_weight_scale(
            shard_dim=0,
            expert_data=expert_bias,
            shard_id="w3",
            loaded_weight=up_bias,
        )

        self.assertTrue(torch.equal(expert_bias[:4], gate_bias[12:16]))
        self.assertTrue(torch.equal(expert_bias[4:], up_bias[12:16]))

    def test_w2_rejects_moe_tp_that_cannot_partition_sixteen_k_blocks(self):
        for moe_tp_size in (0, 3, 32):
            with self.subTest(moe_tp_size=moe_tp_size):
                layer = torch.nn.Module()
                layer.moe_tp_size = moe_tp_size
                w2 = ModelSlimW4A8Int8MoE(
                    quant_config={}, weight_prefix="w2", tp_size=1
                )
                with self.assertRaisesRegex(ValueError, "divisor of 16"):
                    w2.create_weights(
                        layer,
                        num_experts=1,
                        hidden_size=32,
                        intermediate_size_per_partition=32,
                    )

    def test_w13_does_not_validate_moe_tp_for_its_single_block_bias(self):
        layer = torch.nn.Module()
        layer.moe_tp_size = 3
        w13 = ModelSlimW4A8Int8MoE(quant_config={}, weight_prefix="w13", tp_size=3)

        w13.create_weights(
            layer,
            num_experts=1,
            hidden_size=32,
            intermediate_size_per_partition=8,
        )

        self.assertEqual(tuple(layer.w13_scale_bias.shape), (1, 16, 1))
