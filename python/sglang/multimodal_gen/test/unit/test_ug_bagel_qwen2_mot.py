# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
from torch import nn

from sglang.srt.models.bagel_qwen2_mot import (
    BAGELMoTTokenRouting,
    BAGELQwen2MoTDecoderLayer,
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

    def test_mot_token_routing_rejects_invalid_indices(self):
        valid = BAGELMoTTokenRouting(
            text_token_indices=torch.tensor([0, 2]),
            vae_token_indices=torch.tensor([1]),
        )
        valid.validate(total_tokens=3)

        with self.assertRaisesRegex(ValueError, "cover each input token"):
            BAGELMoTTokenRouting(
                text_token_indices=torch.tensor([0]),
                vae_token_indices=torch.tensor([1]),
            ).validate(total_tokens=3)

        with self.assertRaisesRegex(ValueError, "disjoint"):
            BAGELMoTTokenRouting(
                text_token_indices=torch.tensor([0, 1]),
                vae_token_indices=torch.tensor([1]),
            ).validate(total_tokens=3)

        with self.assertRaisesRegex(ValueError, "out of range"):
            BAGELMoTTokenRouting(
                text_token_indices=torch.tensor([0, 3]),
                vae_token_indices=torch.tensor([1]),
            ).validate(total_tokens=3)

    def test_decoder_gen_routes_text_and_vae_branches(self):
        layer = object.__new__(BAGELQwen2MoTDecoderLayer)
        nn.Module.__init__(layer)
        layer.input_layernorm = _AddModule(1.0)
        layer.input_layernorm_moe_gen = _AddModule(2.0)
        layer.post_attention_layernorm = _AddModule(100.0)
        layer.post_attention_layernorm_moe_gen = _AddModule(200.0)
        layer.self_attn = _FakeGenAttention()
        layer.mlp = _AddModule(1000.0)
        layer.mlp_moe_gen = _AddModule(2000.0)

        routing = BAGELMoTTokenRouting(
            text_token_indices=torch.tensor([0, 2]),
            vae_token_indices=torch.tensor([1, 3]),
        )
        routing.validate(total_tokens=4)

        hidden_states, residual = layer.forward_gen(
            positions=torch.arange(4),
            hidden_states=torch.zeros(4, 2),
            forward_batch=object(),
            residual=None,
            routing=routing,
        )

        self.assertEqual(layer.self_attn.mode, "gen")
        self.assertTrue(
            torch.equal(
                layer.self_attn.routing.text_token_indices, torch.tensor([0, 2])
            )
        )
        self.assertTrue(
            torch.equal(layer.self_attn.routing.vae_token_indices, torch.tensor([1, 3]))
        )
        self.assertTrue(
            torch.equal(
                hidden_states,
                torch.tensor(
                    [
                        [1111.0, 1111.0],
                        [2212.0, 2212.0],
                        [1111.0, 1111.0],
                        [2212.0, 2212.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                residual,
                torch.tensor(
                    [
                        [11.0, 11.0],
                        [12.0, 12.0],
                        [11.0, 11.0],
                        [12.0, 12.0],
                    ]
                ),
            )
        )


class _AddModule(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


class _FakeGenAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = None
        self.routing = None

    def forward(self, positions, hidden_states, forward_batch, *, mode, routing):
        self.mode = mode
        self.routing = routing
        return hidden_states + 10.0


if __name__ == "__main__":
    unittest.main()
