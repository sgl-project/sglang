"""Nemotron-3.5-Embed NVFP4 quantization targets match the layer names SGLang builds."""

import unittest

from torch import nn

from sglang.srt.layers.quantization.compressed_tensors.utils import find_matched_target
from sglang.srt.models.nemotron35_embedding import EmbeddingNemotron35Model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")

# Layer 0's targets as nvidia/Nemotron-3.5-Embed-1B-NVFP4 writes them.
_TEXT_PATTERN = r"re:^(?:model\.)?language_model(?:\.model)?\.layers\.0\.{}$"
_TARGETS = [
    _TEXT_PATTERN.format(name.replace(".", r"\."))
    for name in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
] + [
    "vision_tower.transformer.layers.0.attention.q_proj",
    "vision_tower.transformer.layers.0.attention.k_proj",
    "vision_tower.transformer.layers.0.attention.v_proj",
    "vision_tower.transformer.layers.0.attention.o_proj",
    "multi_modal_projector.linear_1",
]


def _match(layer_name: str, targets) -> str:
    return find_matched_target(
        layer_name,
        nn.Module(),
        targets,
        EmbeddingNemotron35Model.packed_modules_mapping,
    )


class TestNemotron35EmbeddingQuantTargets(CustomTestCase):
    def test_text_layers_match_after_mapping(self):
        targets = EmbeddingNemotron35Model.hf_to_sglang_mapper.apply_list(_TARGETS)
        for layer_name, component in (
            ("model.layers.0.self_attn.qkv_proj", "self_attn.q_proj"),
            ("model.layers.0.self_attn.o_proj", "self_attn.o_proj"),
            ("model.layers.0.mlp.gate_up_proj", "mlp.gate_proj"),
            ("model.layers.0.mlp.down_proj", "mlp.down_proj"),
        ):
            with self.subTest(layer=layer_name):
                self.assertIn(
                    component.replace(".", r"\."), _match(layer_name, targets)
                )

    def test_vision_and_projector_targets_are_unchanged(self):
        targets = EmbeddingNemotron35Model.hf_to_sglang_mapper.apply_list(_TARGETS)
        self.assertEqual(targets[7:], _TARGETS[7:])
        self.assertEqual(
            _match("vision_tower.transformer.layers.0.attention.qkv_proj", targets),
            "vision_tower.transformer.layers.0.attention.q_proj",
        )
        self.assertEqual(
            _match("multi_modal_projector.linear_1", targets),
            "multi_modal_projector.linear_1",
        )

    def test_text_layers_do_not_match_the_checkpoint_targets_as_written(self):
        with self.assertRaisesRegex(ValueError, "Unable to find matching target"):
            _match("model.layers.0.self_attn.qkv_proj", _TARGETS)


if __name__ == "__main__":
    unittest.main()
