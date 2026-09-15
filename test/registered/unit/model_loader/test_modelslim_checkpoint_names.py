"""Checkpoint metadata must select schemes for the registered FFN paths."""

import copy
import unittest
from unittest.mock import patch

from torch import nn

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.modelslim.modelslim import (
    ModelSlimConfig,
    ModelSlimFusedMoEMethod,
    ModelSlimLinearMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimW8A8Int8
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelSlimCheckpointNames(unittest.TestCase):
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
