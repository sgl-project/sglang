"""Quark MXFP4 checkpoint wiring for Mistral Large 4
(amd/Mistral-Large-4-1T-A52B-Quark-MXFP4)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.models.mistral_large_4 import (
    MISTRAL_LARGE_4_QUARK_MAPPER,
    MistralLarge4ForCausalLM,
)
from sglang.srt.models.pixtral import PixtralForConditionalGeneration
from sglang.test.test_utils import CustomTestCase

_MXFP4 = {
    "input_tensors": {
        "dtype": "fp4",
        "is_dynamic": True,
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
    },
    "output_tensors": None,
    "weight": {
        "dtype": "fp4",
        "is_dynamic": False,
        "qscheme": "per_group",
        "group_size": 32,
        "scale_format": "e8m0",
    },
    "bias": None,
}
_BLOCK_FP8 = {
    "input_tensors": {
        "dtype": "fp8_e4m3",
        "is_dynamic": True,
        "qscheme": "per_group",
        "group_size": 128,
    },
    "output_tensors": None,
    "weight": {
        "dtype": "fp8_e4m3",
        "is_dynamic": False,
        "qscheme": "per_block",
        "block_size": [128, 128],
    },
    "bias": None,
}

# Names as the checkpoint's quantization_config spells them (no packed_modules_mapping).
_EXCLUDE = [
    "layers.1.attention.wkv_a_with_mqa",
    "layers.1.attention.wkv_b",
    "layers.1.attention.indexer.wk",
    "layers.1.attention.indexer.weights_proj",
    "layers.1.gate",
    "output",
    "mtp_dspark.backbone.layers.0.attention.wq",
    "vision_encoder.transformer.layers.0.attention.wq",
    "patch_merger.merging_layer",
    "vision_language_adapter.w_in",
]
_LAYER_QUANT_CONFIG = {
    "layers.0.feed_forward.w1": _BLOCK_FP8,
    "layers.0.feed_forward.w2": _BLOCK_FP8,
    "layers.0.feed_forward.w3": _BLOCK_FP8,
    "layers.1.attention.wq": _BLOCK_FP8,
    "layers.1.attention.wo": _BLOCK_FP8,
    "layers.1.attention.indexer.wq_b": _BLOCK_FP8,
    "layers.1.shared_experts.w1": _BLOCK_FP8,
    "layers.1.shared_experts.w2": _BLOCK_FP8,
    "layers.1.shared_experts.w3": _BLOCK_FP8,
}


def _quark_checkpoint_config():
    return {
        "quant_method": "quark",
        "export": {"kv_cache_group": [], "pack_method": "reorder"},
        "global_quant_config": _MXFP4,
        "layer_quant_config": dict(_LAYER_QUANT_CONFIG),
        "layer_type_quant_config": {},
        "exclude": list(_EXCLUDE),
    }


def _mapped_quant_config():
    quant = QuarkConfig.from_config(_quark_checkpoint_config())
    quant.apply_weight_name_mapper(MISTRAL_LARGE_4_QUARK_MAPPER)
    quant.update_packed_modules_mapping(
        MistralLarge4ForCausalLM.packed_modules_mapping
    )
    return quant


class TestMistralLarge4QuarkNames(CustomTestCase):
    def test_exclude_names_map_to_model_modules(self):
        mapped = MISTRAL_LARGE_4_QUARK_MAPPER.apply_list(_EXCLUDE)
        self.assertEqual(
            mapped,
            [
                "model.layers.1.self_attn.kv_a_proj_with_mqa",
                "model.layers.1.self_attn.kv_b_proj",
                "model.layers.1.self_attn.indexer.wk",
                "model.layers.1.self_attn.indexer.weights_proj",
                "model.layers.1.mlp.gate",
                "lm_head",
            ],
        )

    def test_layer_quant_config_names_map_to_model_modules(self):
        mapped = MISTRAL_LARGE_4_QUARK_MAPPER.apply_dict(_LAYER_QUANT_CONFIG)
        self.assertEqual(
            sorted(mapped),
            [
                "model.layers.0.mlp.down_proj",
                "model.layers.0.mlp.gate_proj",
                "model.layers.0.mlp.up_proj",
                "model.layers.1.mlp.shared_experts.down_proj",
                "model.layers.1.mlp.shared_experts.gate_proj",
                "model.layers.1.mlp.shared_experts.up_proj",
                "model.layers.1.self_attn.indexer.wq_b",
                "model.layers.1.self_attn.o_proj",
                "model.layers.1.self_attn.q_proj",
            ],
        )

    def test_model_name_is_left_alone(self):
        name = "model.layers.1.self_attn.kv_b_proj"
        self.assertEqual(MISTRAL_LARGE_4_QUARK_MAPPER.apply_list([name]), [name])


class TestMistralLarge4QuarkConfig(CustomTestCase):
    def test_config_without_packed_modules_mapping_is_accepted(self):
        quant = QuarkConfig.from_config(_quark_checkpoint_config())
        self.assertEqual(quant.packed_modules_mapping, {})

    def test_mapper_only_for_quark_checkpoints(self):
        def text_config(quant_config):
            return SimpleNamespace(quantization_config=quant_config)

        for quant_config, expected in (
            ({"quant_method": "quark"}, True),
            ({"quant_method": "fp8"}, False),
            ({"quant_method": "modelopt"}, False),
            (None, False),
        ):
            with self.subTest(quant_config=quant_config):
                mapper = MistralLarge4ForCausalLM.get_hf_to_sglang_mapper(
                    text_config(quant_config)
                )
                self.assertEqual(mapper is MISTRAL_LARGE_4_QUARK_MAPPER, expected)

    def test_pixtral_host_forwards_the_mapper(self):
        hf_config = SimpleNamespace(
            text_config=SimpleNamespace(
                model_type="mistral_large_4",
                quantization_config={"quant_method": "quark"},
            ),
            quantization_config={"quant_method": "quark"},
        )
        self.assertIs(
            PixtralForConditionalGeneration.get_hf_to_sglang_mapper(hf_config),
            MISTRAL_LARGE_4_QUARK_MAPPER,
        )
        hf_config.text_config.model_type = "deepseek_v3"
        self.assertIsNone(
            PixtralForConditionalGeneration.get_hf_to_sglang_mapper(hf_config)
        )

    def test_excluded_modules_stay_unquantized(self):
        quant = _mapped_quant_config()
        for name in (
            "model.layers.1.self_attn.kv_a_proj_with_mqa",
            "model.layers.1.self_attn.kv_b_proj",
            "model.layers.1.self_attn.indexer.wk",
            "model.layers.1.mlp.gate",
            "lm_head",
        ):
            with self.subTest(name=name):
                self.assertFalse(quant.is_block_fp8_layer(name))
                self.assertTrue(
                    quant.is_linear_unquantized(name),
                    f"{name} must not be quantized",
                )

    def test_dense_and_shared_experts_are_block_fp8_through_fused_names(self):
        quant = _mapped_quant_config()
        for name in (
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.1.self_attn.indexer.wq_b",
            "model.layers.1.mlp.shared_experts.gate_up_proj",
            "model.layers.1.mlp.shared_experts.down_proj",
        ):
            with self.subTest(name=name):
                self.assertTrue(quant.is_block_fp8_layer(name))

    def test_routed_experts_keep_the_global_mxfp4_spec(self):
        quant = _mapped_quant_config()
        self.assertFalse(quant.is_block_fp8_layer("model.layers.1.mlp.experts"))
        self.assertIs(
            quant._find_matched_config(
                "model.layers.1.mlp.experts", SimpleNamespace()
            ),
            _MXFP4,
        )


if __name__ == "__main__":
    unittest.main()
