"""Checkpoint metadata must select schemes for the registered FFN paths."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.modelslim.modelslim import (
    ModelSlimConfig,
    ModelSlimFusedMoEMethod,
    ModelSlimLinearMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimW4A4Int4,
    ModelSlimW8A8Int8,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelSlimCheckpointNames(unittest.TestCase):
    def test_qwen3_mixed_precision_checkpoint_offsets(self):
        quant = ModelSlimConfig(
            {
                "model.layers.0.mlp.gate_proj.weight": "W4A4_DYNAMIC",
                "model.layers.0.mlp.up_proj.weight": "W4A4_DYNAMIC",
                "model.layers.0.mlp.down_proj.weight": "W8A8_DYNAMIC",
                "packed_modules_mapping": {
                    "model": {"gate_up_proj": ["gate_proj", "up_proj"]}
                },
            }
        )
        quant.apply_weight_name_mapper(Qwen3ForCausalLM.hf_to_sglang_mapper)
        ffn = nn.Module()
        ffn.gate_up_proj = MergedColumnParallelLinear(
            2,
            [4, 4],
            bias=False,
            quant_config=quant,
            prefix="model.layers.0.ffn.gate_up_proj",
            params_dtype=torch.bfloat16,
            tp_rank=0,
            tp_size=1,
        )
        ffn.down_proj = RowParallelLinear(
            4,
            2,
            bias=False,
            quant_config=quant,
            prefix="model.layers.0.ffn.down_proj",
            params_dtype=torch.bfloat16,
            tp_rank=0,
            tp_size=1,
        )
        self.assertIsInstance(ffn.gate_up_proj.scheme, ModelSlimW4A4Int4)
        self.assertIsInstance(ffn.down_proj.scheme, ModelSlimW8A8Int8)
        self.assertTrue(ffn.down_proj.scheme.is_dynamic)
        for projection in (ffn.gate_up_proj, ffn.down_proj):
            self.assertEqual(projection.weight.dtype, torch.int8)
            self.assertIn("weight_offset", dict(projection.named_parameters()))

        model = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(tie_word_embeddings=False)
        model.model = nn.Module()
        model.model.start_layer, model.model.end_layer = 0, 1
        layer = nn.Module()
        layer.ffn = ffn
        model.model.layers = nn.ModuleList([layer])
        weights = []
        expected = {}
        for parameter in ("weight_scale", "weight_offset"):
            fused = getattr(ffn.gate_up_proj, parameter)
            down = getattr(ffn.down_proj, parameter)
            with torch.no_grad():
                fused.zero_()
                down.zero_()
            gate_value = torch.full_like(fused[:4], 1)
            up_value = torch.full_like(fused[4:], 2)
            down_value = torch.full_like(down, 3)
            weights.extend(
                [
                    (f"model.layers.0.mlp.gate_proj.{parameter}", gate_value),
                    (f"model.layers.0.mlp.up_proj.{parameter}", up_value),
                    (f"model.layers.0.mlp.down_proj.{parameter}", down_value),
                ]
            )
            expected[f"gate_up_proj.{parameter}"] = torch.cat((gate_value, up_value))
            expected[f"down_proj.{parameter}"] = down_value
        model.load_weights(weights)
        for name, value in expected.items():
            torch.testing.assert_close(dict(ffn.named_parameters())[name], value)

    def test_checkpoint_schemes_use_registered_ffn_names(self):
        for model_cls in (DeepseekV2ForCausalLM, Qwen3MoeForCausalLM):
            for checkpoint_member in ("mlp", "ffn"):
                with self.subTest(model=model_cls.__name__, member=checkpoint_member):
                    source = f"model.layers.1.{checkpoint_member}"
                    target = "model.layers.1.ffn"
                    description = {
                        f"{source}.gate_proj.weight": "W8A8_DYNAMIC",
                        f"{source}.up_proj.weight": "W8A8_DYNAMIC",
                        f"{source}.down_proj.weight": "FLOAT",
                        **{
                            f"{source}.experts.0.{proj}.weight": "W8A8_DYNAMIC"
                            for proj in ("gate_proj", "up_proj", "down_proj")
                        },
                        "ignore": [f"{source}.down_proj"],
                        "model_quant_type": "W8A8_DYNAMIC",
                    }
                    original = copy.deepcopy(description)
                    config = ModelSlimConfig(description)
                    config.update_packed_modules_mapping(
                        model_cls.packed_modules_mapping
                    )
                    mapper = model_cls.hf_to_sglang_mapper
                    config.apply_weight_name_mapper(mapper)
                    # Reusing an already canonical config must be safe.
                    config.apply_weight_name_mapper(mapper)

                    linear = ReplicatedLinear.__new__(ReplicatedLinear)
                    nn.Module.__init__(linear)
                    method = config.get_quant_method(linear, f"{target}.gate_proj")
                    self.assertIsInstance(method, ModelSlimLinearMethod)
                    self.assertIsInstance(linear.scheme, ModelSlimW8A8Int8)
                    self.assertTrue(linear.scheme.is_dynamic)
                    self.assertIsInstance(
                        config.get_quant_method(linear, f"{target}.down_proj"),
                        UnquantizedLinearMethod,
                    )

                    experts = FusedMoE.__new__(FusedMoE)
                    nn.Module.__init__(experts)
                    # Scheme selection is real; constructing an NPU kernel requires
                    # torch_npu, which is unavailable in CPU CI.
                    with patch(
                        "sglang.srt.layers.quantization.modelslim.schemes."
                        "modelslim_w8a8_int8_moe.NPUW8A8Int8MoEMethod"
                    ):
                        self.assertIsInstance(
                            config.get_quant_method(experts, f"{target}.experts"),
                            ModelSlimFusedMoEMethod,
                        )
                    self.assertEqual(experts.w13_scheme.weight_prefix, "w13")
                    self.assertEqual(experts.w2_scheme.weight_prefix, "w2")
                    self.assertEqual(config.ignore, [f"{target}.down_proj"])
                    self.assertEqual(config.quant_description["ignore"], config.ignore)
                    self.assertEqual(
                        config.quant_description["model_quant_type"], "W8A8_DYNAMIC"
                    )
                    self.assertEqual(description, original)


if __name__ == "__main__":
    unittest.main()
