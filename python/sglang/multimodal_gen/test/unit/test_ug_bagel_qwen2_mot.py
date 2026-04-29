# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.srt.models.bagel_qwen2_mot import (
    BAGELQwen2MoTForCausalLM,
    _iter_bagel_language_model_weights,
)
from sglang.srt.models.registry import ModelRegistry


class TestBAGELQwen2MoTModel(unittest.TestCase):
    def test_filters_bagel_checkpoint_to_language_model_weights(self):
        weights = [
            ("language_model.model.embed_tokens.weight", "embed"),
            (
                "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight",
                "mot_q",
            ),
            ("language_model.model.layers.0.mlp_moe_gen.gate_proj.weight", "mot_mlp"),
            ("vit_model.vision_model.embeddings.patch_embedding.weight", "vit"),
            ("decoder.conv_in.weight", "vae"),
            ("connector.fc1.weight", "outer"),
            ("model.layers.0.self_attn.q_proj.weight", "plain_qwen"),
        ]

        filtered = list(_iter_bagel_language_model_weights(weights))

        self.assertEqual(
            filtered,
            [
                ("model.embed_tokens.weight", "embed"),
                ("model.layers.0.self_attn.q_proj_moe_gen.weight", "mot_q"),
                ("model.layers.0.mlp_moe_gen.gate_proj.weight", "mot_mlp"),
                ("model.layers.0.self_attn.q_proj.weight", "plain_qwen"),
            ],
        )

    def test_model_registry_sees_bagel_qwen2_mot_architecture(self):
        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            ["BAGELQwen2MoTForCausalLM"]
        )

        self.assertIs(model_cls, BAGELQwen2MoTForCausalLM)
        self.assertEqual(resolved_arch, "BAGELQwen2MoTForCausalLM")


if __name__ == "__main__":
    unittest.main()
