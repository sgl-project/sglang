import math
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-b200")


class TestFlashInferTrtllmBf16HotUpdate(CustomTestCase):
    num_experts = 2
    hidden_size = 128
    intermediate_size = 192
    padded_intermediate_size = 256

    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for FlashInfer TRTLLM BF16 MoE packing.")
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("FlashInfer TRTLLM BF16 MoE is only tested on SM100+.")

        try:
            from flashinfer.fused_moe.core import convert_to_block_layout  # noqa: F401

            import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm  # noqa: F401
        except ImportError as err:
            self.skipTest(f"FlashInfer TRTLLM MoE helpers are unavailable: {err}")

        server_args = ServerArgs(
            model_path="dummy",
            moe_runner_backend="flashinfer_trtllm",
        )
        set_global_server_args_for_scheduler(server_args)
        initialize_moe_config(server_args)

    def _make_weight(self, shape, offset):
        values = torch.arange(
            math.prod(shape), device="cuda", dtype=torch.float32
        ).reshape(shape)
        return (values / 1000 + offset).to(torch.bfloat16)

    def _make_layer(
        self,
        tp_size=1,
        tp_rank=0,
        use_weight_loader_fused=False,
        intermediate_size=None,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
        from sglang.srt.layers.moe.token_dispatcher import standard

        intermediate_size = intermediate_size or self.intermediate_size
        fm = fused_moe_layer
        std = standard
        with (
            patch.object(fm, "get_moe_expert_parallel_world_size", return_value=1),
            patch.object(fm, "get_moe_expert_parallel_rank", return_value=0),
            patch.object(
                fm, "get_moe_tensor_parallel_world_size", return_value=tp_size
            ),
            patch.object(fm, "get_moe_tensor_parallel_rank", return_value=tp_rank),
            patch.object(std, "get_moe_expert_parallel_world_size", return_value=1),
            patch.object(std, "get_moe_expert_parallel_rank", return_value=0),
        ):
            layer = FusedMoE(
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                intermediate_size=intermediate_size,
                layer_id=0,
                top_k=1,
                params_dtype=torch.bfloat16,
                use_weight_loader_fused=use_weight_loader_fused,
            )
            return layer.cuda()

    def _load_expert(self, layer, expert_id, w1, w3, w2):
        prefix = f"model.layers.0.mlp.experts.{expert_id}"
        layer.weight_loader(
            layer.w13_weight,
            w1,
            f"{prefix}.gate_proj.weight",
            "w1",
            expert_id,
        )
        layer.weight_loader(
            layer.w13_weight,
            w3,
            f"{prefix}.up_proj.weight",
            "w3",
            expert_id,
        )
        layer.weight_loader(
            layer.w2_weight,
            w2,
            f"{prefix}.down_proj.weight",
            "w2",
            expert_id,
        )

    def _load_all_experts(self, layer, offset):
        loaded = []
        for expert_id in range(self.num_experts):
            expert_offset = offset + expert_id * 10
            w1 = self._make_weight(
                (self.intermediate_size, self.hidden_size), expert_offset + 1
            )
            w3 = self._make_weight(
                (self.intermediate_size, self.hidden_size), expert_offset + 3
            )
            w2 = self._make_weight(
                (self.hidden_size, self.intermediate_size), expert_offset + 2
            )
            self._load_expert(layer, expert_id, w1, w3, w2)
            loaded.append((w1, w3, w2))
        return loaded

    def _assert_equal(self, actual, expected):
        if not torch.is_tensor(expected):
            expected = torch.full_like(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _assert_canonical_update_visible(self, layer, loaded):
        real = self.intermediate_size
        padded = self.padded_intermediate_size

        for expert_id, (w1, w3, w2) in enumerate(loaded):
            # FlashInfer TRTLLM stores W13 in W3/W1 order.
            self._assert_equal(layer.w13_weight[expert_id, :real, :], w3)
            self._assert_equal(
                layer.w13_weight[expert_id, padded : padded + real, :], w1
            )
            self._assert_equal(layer.w2_weight[expert_id, :, :real], w2)

            self._assert_equal(layer.w13_weight[expert_id, real:padded, :], 0)
            self._assert_equal(
                layer.w13_weight[expert_id, padded + real : 2 * padded, :], 0
            )
            self._assert_equal(layer.w2_weight[expert_id, :, real:padded], 0)

    def test_hot_update_reloads_canonical_weights_after_flashinfer_pack(self):
        layer = self._make_layer()

        self._load_all_experts(layer, offset=0)
        canonical_w13_shape = tuple(layer.w13_weight.shape)
        canonical_w2_shape = tuple(layer.w2_weight.shape)
        layer.quant_method.process_weights_after_loading(layer)

        # Test precondition: hot update below must start from packed params,
        # otherwise it would not exercise restore-before-load.
        self.assertNotEqual(tuple(layer.w13_weight.shape), canonical_w13_shape)
        self.assertNotEqual(tuple(layer.w2_weight.shape), canonical_w2_shape)

        # Mimic hot update: params are packed, but replacement weights are
        # canonical. The weight loader should restore canonical shape first.
        updated_loaded = self._load_all_experts(layer, offset=100)
        self.assertEqual(tuple(layer.w13_weight.shape), canonical_w13_shape)
        self.assertEqual(tuple(layer.w2_weight.shape), canonical_w2_shape)
        self._assert_canonical_update_visible(layer, updated_loaded)

        layer.quant_method.process_weights_after_loading(layer)
        self.assertNotEqual(tuple(layer.w13_weight.shape), canonical_w13_shape)
        self.assertNotEqual(tuple(layer.w2_weight.shape), canonical_w2_shape)

    def test_fused_w13_load_shards_each_half_independently(self):
        tp_size = 2
        per_rank_size = self.intermediate_size // tp_size

        for tp_rank in range(tp_size):
            with self.subTest(tp_rank=tp_rank):
                layer = self._make_layer(
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    use_weight_loader_fused=True,
                )
                padded_per_rank_size = layer.intermediate_size_per_partition
                full_w13 = self._make_weight(
                    (
                        self.num_experts,
                        2 * self.intermediate_size,
                        self.hidden_size,
                    ),
                    offset=200 + tp_rank,
                )

                layer.weight_loader_fused(
                    layer.w13_weight,
                    full_w13,
                    "model.layers.0.mlp.experts.gate_up_proj.weight",
                    "w13",
                )

                first_half_start = per_rank_size * tp_rank
                second_half_start = self.intermediate_size + per_rank_size * tp_rank
                torch.testing.assert_close(
                    layer.w13_weight[:, :per_rank_size, :],
                    full_w13[
                        :, second_half_start : second_half_start + per_rank_size, :
                    ],
                )
                torch.testing.assert_close(
                    layer.w13_weight[
                        :,
                        padded_per_rank_size : padded_per_rank_size + per_rank_size,
                        :,
                    ],
                    full_w13[:, first_half_start : first_half_start + per_rank_size, :],
                )

    def test_tp_padded_load_uses_unpadded_source_shard(self):
        tp_size = 8
        tp_rank = 3
        intermediate_size = 768
        shard_size = intermediate_size // tp_size
        padded_shard_size = 128
        source_shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        layer = self._make_layer(
            tp_size=tp_size,
            tp_rank=tp_rank,
            intermediate_size=intermediate_size,
        )

        loaded = []
        for expert_id in range(self.num_experts):
            expert_offset = expert_id * 10
            w1 = self._make_weight(
                (intermediate_size, self.hidden_size), expert_offset + 1
            )
            w3 = self._make_weight(
                (intermediate_size, self.hidden_size), expert_offset + 3
            )
            w2 = self._make_weight(
                (self.hidden_size, intermediate_size), expert_offset + 2
            )
            self._load_expert(layer, expert_id, w1, w3, w2)
            loaded.append((w1, w3, w2))

        self.assertEqual(
            tuple(layer.w13_weight.shape),
            (self.num_experts, 2 * padded_shard_size, self.hidden_size),
        )
        self.assertEqual(
            tuple(layer.w2_weight.shape),
            (self.num_experts, self.hidden_size, padded_shard_size),
        )

        for expert_id, (w1, w3, w2) in enumerate(loaded):
            self._assert_equal(
                layer.w13_weight[expert_id, :shard_size, :], w3[source_shard, :]
            )
            self._assert_equal(
                layer.w13_weight[
                    expert_id,
                    padded_shard_size : padded_shard_size + shard_size,
                    :,
                ],
                w1[source_shard, :],
            )
            self._assert_equal(
                layer.w2_weight[expert_id, :, :shard_size], w2[:, source_shard]
            )
            self._assert_equal(
                layer.w13_weight[expert_id, shard_size:padded_shard_size, :], 0
            )
            self._assert_equal(
                layer.w13_weight[
                    expert_id,
                    padded_shard_size + shard_size : 2 * padded_shard_size,
                    :,
                ],
                0,
            )
            self._assert_equal(
                layer.w2_weight[expert_id, :, shard_size:padded_shard_size], 0
            )


if __name__ == "__main__":
    unittest.main()
