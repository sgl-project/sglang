"""CPU tests: a compressed-tensors ignore list written with checkpoint module names
must match the vision layers that SGLang builds under other names.

Gemma 3 loads the SigLIP attention's ``out_proj`` into VisionAttention's ``proj``.
Mistral 3 strips the ``model.`` prefix of its vision tower, loads Pixtral's
``o_proj`` into ``proj``, and fuses q/k/v and gate/up. When the ignore list is
not renamed the same way, these BF16 layers are built quantized and their
weights land in quantized parameters (or are not loaded at all), with no error.

The tests repeat what the model loader does: it passes the model class's
``packed_modules_mapping`` into the quantization config and then applies the
class's ``hf_to_sglang_mapper`` to the config's module lists.
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


def _config(model_cls, ignore):
    """The quantization config as the model loader builds it for ``model_cls``."""
    config = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                        "strategy": "channel",
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
            "ignore": ignore,
            "packed_modules_mapping": dict(
                getattr(model_cls, "packed_modules_mapping", {})
            ),
        }
    )
    mapper = getattr(model_cls, "hf_to_sglang_mapper", None)
    if mapper is not None:
        config.apply_weight_name_mapper(mapper)
    return config


def _ignored(config, layer_name):
    # The check get_linear_scheme makes before it looks up a scheme.
    return should_ignore_layer(
        layer_name, ignore=config.ignore, fused_mapping=config.packed_modules_mapping
    )


class TestGemma3VisionIgnore(CustomTestCase):
    # As in RedHatAI/gemma-3-27b-it-FP8-dynamic: the vision tower stays BF16.
    LAYERS = 2
    IGNORE = [
        f"vision_tower.vision_model.encoder.layers.{i}.{name}"
        for i in range(LAYERS)
        for name in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "mlp.fc1",
            "mlp.fc2",
        )
    ] + ["language_model.lm_head"]

    def setUp(self):
        self.config = _config(Gemma3ForConditionalGeneration, self.IGNORE)

    def test_vision_layers_ignored(self):
        for i in range(self.LAYERS):
            prefix = f"vision_tower.vision_model.encoder.layers.{i}"
            for name in ("self_attn.qkv_proj", "self_attn.proj", "mlp.fc1", "mlp.fc2"):
                with self.subTest(layer=f"{prefix}.{name}"):
                    self.assertTrue(_ignored(self.config, f"{prefix}.{name}"))

    def test_language_layers_quantized(self):
        prefix = "language_model.model.layers.0"
        for name in (
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        ):
            with self.subTest(layer=f"{prefix}.{name}"):
                self.assertFalse(_ignored(self.config, f"{prefix}.{name}"))


class TestMistral3VisionIgnore(CustomTestCase):
    # As in RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4 (transformers v5
    # names): the Pixtral tower and the projector stay BF16.
    LAYERS = 2
    IGNORE = [
        f"model.vision_tower.transformer.layers.{i}.{name}"
        for i in range(LAYERS)
        for name in (
            "attention.q_proj",
            "attention.k_proj",
            "attention.v_proj",
            "attention.o_proj",
            "feed_forward.gate_proj",
            "feed_forward.up_proj",
            "feed_forward.down_proj",
        )
    ] + [
        "model.multi_modal_projector.patch_merger.merging_layer",
        "model.multi_modal_projector.linear_1",
        "model.multi_modal_projector.linear_2",
        "lm_head",
    ]

    def _check(self, ignore, vision_prefix):
        config = _config(Mistral3ForConditionalGeneration, ignore)
        for i in range(self.LAYERS):
            prefix = f"{vision_prefix}.transformer.layers.{i}"
            for name in (
                "attention.qkv_proj",
                "attention.proj",
                "feed_forward.gate_up_proj",
                "feed_forward.down_proj",
            ):
                with self.subTest(layer=f"{prefix}.{name}"):
                    self.assertTrue(_ignored(config, f"{prefix}.{name}"))
        prefix = "language_model.model.layers.0"
        for name in (
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        ):
            with self.subTest(layer=f"{prefix}.{name}"):
                self.assertFalse(_ignored(config, f"{prefix}.{name}"))

    def test_v5_names(self):
        self._check(self.IGNORE, "vision_tower")

    def test_v4_names(self):
        # Older exports name the tower without the "model." prefix.
        ignore = [
            name[len("model.") :] if name.startswith("model.") else name
            for name in self.IGNORE
        ]
        self._check(ignore, "vision_tower")

    def test_ignored_language_layer(self):
        # A mixed-precision export that keeps one decoder layer in BF16.
        ignore = self.IGNORE + [
            f"model.language_model.layers.0.self_attn.{p}_proj" for p in "qkvo"
        ]
        config = _config(Mistral3ForConditionalGeneration, ignore)
        self.assertTrue(
            _ignored(config, "language_model.model.layers.0.self_attn.qkv_proj")
        )
        self.assertTrue(
            _ignored(config, "language_model.model.layers.0.self_attn.o_proj")
        )
        self.assertFalse(
            _ignored(config, "language_model.model.layers.1.self_attn.qkv_proj")
        )


if __name__ == "__main__":
    unittest.main()
