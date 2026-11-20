"""CPU-only tests: vision layers listed in a compressed-tensors ``ignore`` list
must be built unquantized even when sglang renames the checkpoint modules.

Regression for #49061: Gemma 3 stores the SigLIP output projection as
``self_attn.out_proj`` while the sglang module is ``self_attn.proj``, and
Mistral 3 renames the ``model.vision_tower.`` prefix / Pixtral's
``attention.o_proj``. The quantization config's ``ignore`` list was compared
with sglang module names without those renames, so ignored vision layers were
quantized (FP8/NVFP4/W4A16) and produced wrong image answers.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.test.test_utils import CustomTestCase

_FP8_WEIGHTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "channel",
    "symmetric": True,
    "dynamic": False,
}
_FP8_DYNAMIC_ACTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "token",
    "symmetric": True,
    "dynamic": True,
}


def _config(model_cls, ignore):
    """Build the quant config the loader would build for ``model_cls``.

    ``_get_quantization_config()`` injects the model class's
    ``packed_modules_mapping`` into the config and then calls
    ``apply_weight_name_mapper()`` with its ``hf_to_sglang_mapper``.
    """
    config = CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(ignore),
            "packed_modules_mapping": dict(
                getattr(model_cls, "packed_modules_mapping", {})
            ),
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": _FP8_WEIGHTS,
                    "input_activations": _FP8_DYNAMIC_ACTS,
                }
            },
        }
    )
    mapper = getattr(model_cls, "hf_to_sglang_mapper", None)
    if mapper is not None:
        config.apply_weight_name_mapper(mapper)
    return config


class TestGemma3VisionIgnore(CustomTestCase):
    IGNORE = ["vision_tower.vision_model.encoder.layers.0.self_attn.out_proj"]

    def test_ignored_out_proj_is_built_unquantized(self):
        config = _config(Gemma3ForConditionalGeneration, self.IGNORE)

        # The checkpoint name is renamed to the sglang module name ...
        self.assertEqual(
            config.ignore,
            ["vision_tower.vision_model.encoder.layers.0.self_attn.proj"],
        )
        # ... and the renamed entry matches the module sglang actually builds.
        self.assertTrue(
            should_ignore_layer(
                "vision_tower.vision_model.encoder.layers.0.self_attn.proj",
                ignore=config.ignore,
                fused_mapping=config.packed_modules_mapping,
            )
        )

    def test_other_vision_layers_stay_quantized(self):
        config = _config(Gemma3ForConditionalGeneration, self.IGNORE)

        self.assertFalse(
            should_ignore_layer(
                "vision_tower.vision_model.encoder.layers.1.self_attn.proj",
                ignore=config.ignore,
                fused_mapping=config.packed_modules_mapping,
            )
        )

    def test_unmapped_ignore_list_would_quantize_the_layer(self):
        # Documents the pre-fix behaviour: comparing the checkpoint name with the
        # sglang module name never matched.
        self.assertFalse(
            should_ignore_layer(
                "vision_tower.vision_model.encoder.layers.0.self_attn.proj",
                ignore=self.IGNORE,
                fused_mapping=Gemma3ForConditionalGeneration.packed_modules_mapping,
            )
        )


class TestMistral3VisionIgnore(CustomTestCase):
    IGNORE = [
        "model.vision_tower.transformer.layers.0.attention.o_proj",
        "model.vision_tower.transformer.layers.0.attention.q_proj",
        "model.vision_tower.transformer.layers.0.attention.k_proj",
        "model.vision_tower.transformer.layers.0.attention.v_proj",
        "model.vision_tower.transformer.layers.0.feed_forward.gate_proj",
        "model.vision_tower.transformer.layers.0.feed_forward.up_proj",
    ]

    def test_ignored_vision_layers_are_built_unquantized(self):
        config = _config(Mistral3ForConditionalGeneration, self.IGNORE)

        self.assertEqual(
            config.ignore,
            [
                "vision_tower.transformer.layers.0.attention.proj",
                "vision_tower.transformer.layers.0.attention.q_proj",
                "vision_tower.transformer.layers.0.attention.k_proj",
                "vision_tower.transformer.layers.0.attention.v_proj",
                "vision_tower.transformer.layers.0.feed_forward.gate_proj",
                "vision_tower.transformer.layers.0.feed_forward.up_proj",
            ],
        )
        # Unfused projection.
        self.assertTrue(
            should_ignore_layer(
                "vision_tower.transformer.layers.0.attention.proj",
                ignore=config.ignore,
                fused_mapping=config.packed_modules_mapping,
            )
        )
        # Fused projections are expanded through packed_modules_mapping.
        self.assertTrue(
            should_ignore_layer(
                "vision_tower.transformer.layers.0.attention.qkv_proj",
                ignore=config.ignore,
                fused_mapping=config.packed_modules_mapping,
            )
        )
        self.assertTrue(
            should_ignore_layer(
                "vision_tower.transformer.layers.0.feed_forward.gate_up_proj",
                ignore=config.ignore,
                fused_mapping=config.packed_modules_mapping,
            )
        )

    def test_other_vision_layers_stay_quantized(self):
        config = _config(Mistral3ForConditionalGeneration, self.IGNORE)

        self.assertFalse(
            should_ignore_layer(
                "vision_tower.transformer.layers.1.attention.proj",
                ignore=config.ignore,
                fused_mapping=config.packed_modules_mapping,
            )
        )
        self.assertFalse(
            should_ignore_layer(
                "vision_tower.transformer.layers.1.attention.qkv_proj",
                ignore=config.ignore,
                fused_mapping=config.packed_modules_mapping,
            )
        )

    def test_unmapped_ignore_list_would_quantize_the_tower(self):
        # Pre-fix behaviour: no prefix rename, no o_proj rename, no fused mapping.
        for layer_name in (
            "vision_tower.transformer.layers.0.attention.proj",
            "vision_tower.transformer.layers.0.attention.qkv_proj",
            "vision_tower.transformer.layers.0.feed_forward.gate_up_proj",
        ):
            self.assertFalse(
                should_ignore_layer(layer_name, ignore=self.IGNORE, fused_mapping={})
            )


if __name__ == "__main__":
    unittest.main()
