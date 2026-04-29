# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.bagel_qwen2_mot import (
    BAGELMoTTokenRouting,
    BAGELQwen2MoTAttention,
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
            ("time_embedder.mlp.0.weight", "time"),
            ("vae2llm.weight", "vae2llm"),
            ("llm2vae.weight", "llm2vae"),
            ("latent_pos_embed.pos_embed", "latent_pos"),
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
                ("time_embedder.mlp.0.weight", "time"),
                ("vae2llm.weight", "vae2llm"),
                ("llm2vae.weight", "llm2vae"),
                ("latent_pos_embed.pos_embed", "latent_pos"),
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

    def test_attention_gen_routes_text_and_vae_qkv_and_output_branches(self):
        attention = object.__new__(BAGELQwen2MoTAttention)
        nn.Module.__init__(attention)
        attention.head_dim = 2
        attention.q_size = 2
        attention.kv_size = 2
        attention.alt_stream = None
        attention.qkv_proj = _FakeQKVProjector(q_value=10.0, k_value=20.0, v_value=30.0)
        attention.qkv_proj_moe_gen = _FakeQKVProjector(
            q_value=100.0,
            k_value=200.0,
            v_value=300.0,
        )
        attention.q_norm = _AddModule(1.0)
        attention.k_norm = _AddModule(2.0)
        attention.q_norm_moe_gen = _AddModule(3.0)
        attention.k_norm_moe_gen = _AddModule(4.0)
        attention.rotary_emb = _FakeRotary()
        attention.attn = _FakeRadixAttention()
        attention.o_proj = _TupleAddModule(10000.0)
        attention.o_proj_moe_gen = _TupleAddModule(20000.0)

        with patch(
            "sglang.srt.models.bagel_qwen2_mot.apply_qk_norm",
            _fake_apply_qk_norm,
        ):
            output = attention.forward_gen(
                positions=torch.arange(4),
                hidden_states=torch.zeros(4, 2),
                forward_batch=object(),
                routing=BAGELMoTTokenRouting(
                    text_token_indices=torch.tensor([0, 2]),
                    vae_token_indices=torch.tensor([1, 3]),
                ),
            )

        self.assertTrue(
            torch.equal(
                attention.attn.q,
                torch.tensor(
                    [
                        [1011.0, 1011.0],
                        [1103.0, 1103.0],
                        [1011.0, 1011.0],
                        [1103.0, 1103.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                attention.attn.k,
                torch.tensor(
                    [
                        [2022.0, 2022.0],
                        [2204.0, 2204.0],
                        [2022.0, 2022.0],
                        [2204.0, 2204.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                attention.attn.v,
                torch.tensor(
                    [
                        [30.0, 30.0],
                        [300.0, 300.0],
                        [30.0, 30.0],
                        [300.0, 300.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor(
                    [
                        [10030.0, 10030.0],
                        [20300.0, 20300.0],
                        [10030.0, 10030.0],
                        [20300.0, 20300.0],
                    ]
                ),
            )
        )

    def test_predict_velocity_from_packed_gen_uses_native_gen_head(self):
        model = object.__new__(BAGELQwen2MoTForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=2)
        model.model = _FakePackedGenModel()
        model.vae2llm = _WeightedAddModule(1.0)
        model.time_embedder = _FakeTimeEmbedder(2.0)
        model.latent_pos_embed = _FakeLatentPositionEmbedding(3.0)
        model.llm2vae = _AddModule(100.0)

        velocity = model.predict_velocity_from_packed_gen(
            latent_tokens=torch.zeros(2, 2),
            timestep=torch.tensor([0.5]),
            packed_vae_token_indexes=torch.tensor([1, 2]),
            packed_vae_position_ids=torch.tensor([4, 5]),
            packed_text_ids=torch.tensor([7, 8]),
            packed_text_indexes=torch.tensor([0, 3]),
            packed_position_ids=torch.tensor([9, 9, 9, 9]),
            packed_seqlens=torch.tensor([4]),
            forward_batch=object(),
        )

        self.assertTrue(torch.equal(velocity, torch.full((2, 2), 116.0)))
        self.assertTrue(
            torch.equal(
                model.model.input_embeds,
                torch.tensor(
                    [
                        [7.0, 7.0],
                        [6.0, 6.0],
                        [6.0, 6.0],
                        [8.0, 8.0],
                    ]
                ),
            )
        )
        self.assertTrue(torch.equal(model.model.text_indices, torch.tensor([0, 3])))
        self.assertTrue(torch.equal(model.model.vae_indices, torch.tensor([1, 2])))


class _AddModule(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return x + self.value


class _TupleAddModule(_AddModule):
    def forward(self, x):
        return x + self.value, None


class _WeightedAddModule(_AddModule):
    def __init__(self, value: float):
        super().__init__(value)
        self.weight = nn.Parameter(torch.zeros(1))


class _FakeTimeEmbedder(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, timestep):
        return torch.full((timestep.shape[0], 2), self.value)


class _FakeLatentPositionEmbedding(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, position_ids):
        return torch.full((position_ids.shape[0], 2), self.value)


class _FakePackedGenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = _FakeTextEmbedding()
        self.input_embeds = None
        self.text_indices = None
        self.vae_indices = None

    def forward_gen_embeds(
        self,
        *,
        input_embeds,
        positions,
        forward_batch,
        text_token_indices,
        vae_token_indices,
    ):
        del positions, forward_batch
        self.input_embeds = input_embeds
        self.text_indices = text_token_indices
        self.vae_indices = vae_token_indices
        return input_embeds + 10.0


class _FakeTextEmbedding(nn.Module):
    def forward(self, input_ids):
        return input_ids.to(torch.float32).unsqueeze(1).expand(-1, 2)


class _FakeQKVProjector(nn.Module):
    def __init__(self, *, q_value: float, k_value: float, v_value: float):
        super().__init__()
        self.values = (q_value, k_value, v_value)

    def forward(self, x):
        parts = [torch.full_like(x, value) for value in self.values]
        return torch.cat(parts, dim=-1), None


class _FakeRotary(nn.Module):
    def forward(self, positions, q, k):
        return q + 1000.0, k + 2000.0


class _FakeRadixAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = None
        self.k = None
        self.v = None

    def forward(self, q, k, v, forward_batch):
        self.q = q
        self.k = k
        self.v = v
        return v


def _fake_apply_qk_norm(*, q, k, q_norm, k_norm, head_dim, alt_stream=None):
    return q_norm(q), k_norm(k)


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
