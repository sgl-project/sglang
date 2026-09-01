"""CPU contracts for compressed-tensors MXFP4 expert checkpoint names."""

from types import SimpleNamespace
from unittest import TestCase, mock

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _normalize_mxfp4_packed_expert_weight_name,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _quant_config(*, name="compressed_tensors", quant_format="mxfp4-pack-quantized"):
    return SimpleNamespace(
        get_name=lambda: name,
        quant_format=quant_format,
    )


GLM53_DATAFREE_QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "config_groups": {
        "config_group_0": {
            "format": "mxfp4-pack-quantized",
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "token",
                "dynamic": True,
            },
        }
    },
    "ignore": [],
}


class _DummyFusedMoE:
    pass


class TestMxfp4PackedExpertNames(TestCase):
    def test_actual_glm_quant_config_selects_generic_mxfp4_method(self):
        config = CompressedTensorsConfig.from_config(GLM53_DATAFREE_QUANT_CONFIG)
        layer = _DummyFusedMoE()

        with mock.patch(
            "sglang.srt.layers.moe.fused_moe_triton.FusedMoE",
            _DummyFusedMoE,
        ), mock.patch(
            "sglang.srt.layers.quantization.mxfp4."
            "Mxfp4MoEMethod",
            return_value="generic-mxfp4",
        ) as generic_method:
            method = config.get_quant_method(layer, "model.layers.3.mlp.experts")

        self.assertEqual(method, "generic-mxfp4")
        generic_method.assert_called_once_with(prefix="model.layers.3.mlp.experts")

    def test_actual_glm_quant_config_normalizes_packed_expert_weight(self):
        config = CompressedTensorsConfig.from_config(GLM53_DATAFREE_QUANT_CONFIG)
        name = "model.layers.3.mlp.experts.0.gate_proj.weight_packed"

        self.assertEqual(
            _normalize_mxfp4_packed_expert_weight_name(name, config),
            "model.layers.3.mlp.experts.0.gate_proj.weight",
        )

    def test_actual_glm_gate_up_down_keys_hit_generic_mxfp4_params(self):
        mappings = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=1,
        )
        prefix = "model.layers.3.mlp."
        cases = {
            "gate_proj": "w13",
            "up_proj": "w13",
            "down_proj": "w2",
        }

        for projection, fused in cases.items():
            for suffix in ("weight_packed", "weight_scale"):
                checkpoint_name = f"{prefix}experts.0.{projection}.{suffix}"
                matched = []
                for param_name, weight_name, _, _ in mappings:
                    if weight_name not in checkpoint_name:
                        continue
                    normalized = _normalize_mxfp4_packed_expert_weight_name(
                        checkpoint_name, _quant_config()
                    )
                    matched.append(normalized.replace(weight_name, param_name))

                expected_suffix = "weight" if suffix == "weight_packed" else suffix
                self.assertEqual(
                    matched,
                    [f"{prefix}experts.{fused}_{expected_suffix}"],
                )

    def test_non_mxfp4_checkpoint_name_is_unchanged(self):
        name = "model.layers.3.mlp.experts.0.gate_proj.weight_packed"
        self.assertEqual(
            _normalize_mxfp4_packed_expert_weight_name(
                name, _quant_config(quant_format="pack-quantized")
            ),
            name,
        )

    def test_mxfp4_scale_name_is_unchanged(self):
        name = "model.layers.3.mlp.experts.0.gate_proj.weight_scale"
        self.assertEqual(
            _normalize_mxfp4_packed_expert_weight_name(name, _quant_config()),
            name,
        )
