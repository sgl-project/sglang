import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def recipe(algo):
    block, payload = (32, 34) if algo == "Q8_0" else (256, 66)
    return dict(
        quant_algo=algo, group_size=block, block_payload_bytes=payload, packing="ggml"
    )


def quant_config(layers):
    config = ModelOptMixedPrecisionConfig.from_config(
        dict(quant_algo="MIXED_PRECISION", quantized_layers=layers)
    )
    config.packed_modules_mapping = {}
    return config


class TestModelOptGGML(CustomTestCase):
    def test_packed_linear_loads_without_becoming_a_dense_weight(self):
        """A GGML recipe must not fall back to a dense BF16 parameter."""
        for algo, blocks, payload in (("Q8_0", 8, 34), ("IQ2_XXS", 1, 66)):
            with self.subTest(algo=algo):
                layer = ReplicatedLinear(
                    256,
                    8,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=quant_config({"linear": recipe(algo)}),
                    prefix="linear",
                )
                source = (
                    torch.arange(8 * blocks * payload)
                    .remainder(256)
                    .to(torch.uint8)
                    .view(8, blocks, payload)
                )
                layer.weight.weight_loader(layer.weight, source)
                self.assertEqual(layer.weight.dtype, torch.uint8)
                torch.testing.assert_close(layer.weight, source)

    def test_mixed_nvfp4_fused_scale_initializes_all_logical_slots(self):
        """A fused scalar must not leave uninitialized scales in the NVFP4 max reduction."""
        layer = MergedColumnParallelLinear(
            input_size=16,
            output_sizes=[16, 16],
            bias=False,
            params_dtype=torch.bfloat16,
            parallel_group="replicated",
            quant_config=quant_config(
                {"linear": dict(quant_algo="W4A16_NVFP4", group_size=16)}
            ),
            prefix="linear",
        )
        layer.weight_scale_2.data.fill_(float("nan"))
        layer.weight_scale_2.weight_loader(layer.weight_scale_2, torch.tensor(0.25))
        torch.testing.assert_close(layer.weight_scale_2, torch.tensor([0.25, 0.25]))

    def test_invalid_block_contract_is_rejected_before_layer_creation(self):
        """A payload incompatible with the advertised codec cannot be copied as valid data."""
        for key, value in (
            ("group_size", 32),
            ("block_payload_bytes", 74),
            ("packing", "other"),
        ):
            with self.subTest(key=key):
                invalid = {**recipe("IQ2_XXS"), key: value}
                with self.assertRaisesRegex(ValueError, "GGML"):
                    quant_config({"linear": invalid})

    def test_mixed_expert_codecs_are_rejected(self):
        """One packed expert buffer cannot silently use the first expert's codec for all experts."""
        config = quant_config(
            {
                "moe.experts.0.up_proj": recipe("IQ2_XXS"),
                "moe.experts.1.up_proj": recipe("Q8_0"),
            }
        )
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        with self.assertRaisesRegex(ValueError, "Mixed GGML expert formats"):
            config.get_quant_method(layer, "moe.experts")

    def test_module_level_expert_recipe_accepts_uniform_children_and_rejects_conflicts(
        self,
    ):
        """An exact module recipe is valid; a conflicting child must still fail."""
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        config = quant_config({"moe.experts": recipe("IQ2_XXS")})
        method = config.get_quant_method(layer, "moe.experts")
        expert, _ = self._expert_layer()
        method.create_weights(expert, 2, 256, 256, torch.bfloat16)
        expert.w13_weight.data.zero_()
        source = torch.ones(512, 1, 66, dtype=torch.uint8)
        method.weight_loader(expert.w13_weight, source, "up_proj.weight", "w1", 1)
        torch.testing.assert_close(expert.w13_weight[1], source[256:])
        for algo in ("IQ2_XXS", "Q8_0"):
            config = quant_config(
                {
                    "moe.experts": recipe("IQ2_XXS"),
                    "moe.experts.0.up_proj": recipe(algo),
                }
            )
            if algo == "IQ2_XXS":
                config.get_quant_method(layer, "moe.experts")
            else:
                with self.assertRaisesRegex(ValueError, "Mixed GGML expert formats"):
                    config.get_quant_method(layer, "moe.experts")

    def test_uncombined_expert_output_request_is_rejected(self):
        """The helper combines outputs, so it must reject no_combine."""
        from sglang.srt.layers.quantization.modelopt_ggml import ModelOptGGMLMoEMethod

        layer = torch.nn.Module()
        method = ModelOptGGMLMoEMethod("IQ2_XXS")
        method.create_moe_runner(
            layer, MoeRunnerConfig(is_gated=False, activation="relu2")
        )
        with self.assertRaisesRegex(ValueError, "no_combine=True"):
            method.create_moe_runner(
                layer,
                MoeRunnerConfig(is_gated=False, activation="relu2", no_combine=True),
            )

    def test_packed_linear_tp_shards_rows_and_complete_input_blocks(self):
        """TP must keep each block's scale and payload together."""
        for algo in ("Q8_0", "IQ2_XXS"):
            block, payload = (32, 34) if algo == "Q8_0" else (256, 66)
            source = (
                torch.arange(16 * (512 // block) * payload)
                .remainder(251)
                .to(torch.uint8)
                .view(16, 512 // block, payload)
            )
            for layer_type in (ColumnParallelLinear, RowParallelLinear):
                with (
                    self.subTest(algo=algo, layer=layer_type.__name__),
                    get_parallel().override(
                        tp_size=2,
                        tp_rank=1,
                        tp_group=SimpleNamespace(rank_in_group=1, world_size=2),
                    ),
                ):
                    layer = layer_type(
                        512,
                        16,
                        bias=False,
                        params_dtype=torch.bfloat16,
                        quant_config=quant_config({"linear": recipe(algo)}),
                        prefix="linear",
                    )
                    layer.weight.weight_loader(layer.weight, source)
                    expected = (
                        source[8:]
                        if layer_type is ColumnParallelLinear
                        else source[:, 256 // block :]
                    )
                    torch.testing.assert_close(layer.weight, expected)

    def test_fused_packed_linear_tp_preserves_projection_boundaries_and_kv_replicas(
        self,
    ):
        """TP must split each projection separately and keep replicated KV heads."""
        for algo in ("Q8_0", "IQ2_XXS"):
            block, payload = (32, 34) if algo == "Q8_0" else (256, 66)
            source = (
                torch.arange(48 * (512 // block) * payload)
                .remainder(251)
                .to(torch.uint8)
                .view(48, 512 // block, payload)
            )
            with (
                self.subTest(algo=algo),
                get_parallel().override(
                    tp_size=2,
                    tp_rank=1,
                    tp_group=SimpleNamespace(rank_in_group=1, world_size=2),
                ),
            ):
                kwargs = dict(
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=quant_config({"linear": recipe(algo)}),
                    prefix="linear",
                )
                merged = MergedColumnParallelLinear(512, [16, 32], **kwargs)
                merged.weight.weight_loader(merged.weight, source)
                torch.testing.assert_close(
                    merged.weight, torch.cat((source[8:16], source[32:48]))
                )
                qkv = QKVParallelLinear(512, 8, 4, 1, **kwargs)
                qkv.weight.weight_loader(qkv.weight, source)
                torch.testing.assert_close(
                    qkv.weight, torch.cat((source[16:32], source[32:40], source[40:48]))
                )

    def _expert_layer(self, *, gated=False):
        from sglang.srt.layers.quantization.modelopt_ggml import ModelOptGGMLMoEMethod

        layer = torch.nn.Module()
        layer.moe_ep_size = 1
        layer.moe_tp_size = 2
        layer.moe_tp_rank = 1
        layer.num_fused_shared_experts = 0
        layer.use_presharded_weights = False
        layer.moe_runner_config = SimpleNamespace(is_gated=gated)
        method = ModelOptGGMLMoEMethod("IQ2_XXS")
        method.create_weights(layer, 2, 256, 256, torch.bfloat16)
        return layer, method

    def test_expert_tp_shards_rows_for_up_and_whole_blocks_for_down(self):
        """TP must shard logical K blocks, preserving their scale and payload bytes together."""
        layer, method = self._expert_layer()
        up = torch.arange(512 * 66).remainder(251).to(torch.uint8).view(512, 1, 66)
        down = (
            torch.arange(256 * 2 * 66).remainder(251).to(torch.uint8).view(256, 2, 66)
        )
        method.weight_loader(layer.w13_weight, up, "up_proj.weight", "w1", 1)
        method.weight_loader(layer.w2_weight, down, "down_proj.weight", "w2", 1)
        torch.testing.assert_close(layer.w13_weight[1], up[256:])
        torch.testing.assert_close(layer.w2_weight[1], down[:, 1:])

    def test_gated_expert_tp_load_keeps_gate_and_up_halves_distinct(self):
        """Gate and up weights must not overwrite each other."""
        layer, method = self._expert_layer(gated=True)
        gate = torch.arange(512 * 66).remainder(113).to(torch.uint8).view(512, 1, 66)
        up = gate + 127
        method.weight_loader(layer.w13_weight, gate, "gate_proj.weight", "w1", 1)
        method.weight_loader(layer.w13_weight, up, "up_proj.weight", "w3", 1)
        torch.testing.assert_close(layer.w13_weight[1, :256], gate[256:])
        torch.testing.assert_close(layer.w13_weight[1, 256:], up[256:])

    def test_expert_loader_success_request_still_copies_the_weight(self):
        """Model loaders asking for success must receive True after the expert is loaded."""
        layer, method = self._expert_layer()
        layer.w13_weight.data.zero_()
        source = torch.ones(512, 1, 66, dtype=torch.uint8)
        success = method.weight_loader(
            layer.w13_weight, source, "up_proj.weight", "w1", 1, return_success=True
        )
        self.assertTrue(success)
        torch.testing.assert_close(layer.w13_weight[1], source[256:])

    def test_invalid_packed_expert_does_not_modify_storage(self):
        layer, method = self._expert_layer()
        layer.w13_weight.data.zero_()
        invalid = torch.ones(512, 1, 65, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "Invalid packed tensor"):
            method.weight_loader(layer.w13_weight, invalid, "up_proj.weight", "w1", 0)
        self.assertEqual(torch.count_nonzero(layer.w13_weight).item(), 0)


if __name__ == "__main__":
    unittest.main()
