"""Small-config coverage for the MLA EAGLE3 auxiliary feature projection."""

import contextlib
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.kimi_k25_eagle3 import (
    Eagle3DeepseekV2ForCausalLM,
    Eagle3MLAModel,
)
from sglang.srt.runtime_context import get_context, get_device, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def small_config(**overrides):
    fields = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        vocab_size=128,
        tie_word_embeddings=False,
        hidden_act="silu",
        rope_theta=10000,
        rope_parameters={
            "rope_type": "deepseek_yarn",
            "rope_theta": 10000,
            "factor": 2.0,
            "original_max_position_embeddings": 16,
        },
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        fc_norm=True,
        norm_output=True,
        eagle_config={"eagle_aux_hidden_state_layer_ids": [2, 30, 58]},
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def reference_projection(model, features):
    if model.fc_norm is not None:
        features = torch.cat(
            [
                F.rms_norm(
                    chunk,
                    (norm.hidden_size,),
                    weight=norm.weight,
                    eps=model.config.rms_norm_eps,
                )
                for chunk, norm in zip(
                    features.chunk(model.num_aux_hidden_states, -1), model.fc_norm
                )
            ],
            dim=-1,
        )
    return F.linear(features, model.fc.weight)


class ReferenceDecoder(torch.nn.Module):
    # Isolate the EAGLE3 feature contract from platform-specific attention kernels.
    def forward(self, positions, embeds, hidden_states, forward_batch, allocator):
        self.projected = hidden_states.detach().clone()
        return hidden_states * 0.25, embeds


class TestKimiEagle3MLAProjection(CustomTestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        # These contract tests run on CPU even when TorchNPU is installed.
        self.stack.enter_context(
            patch(
                "sglang.srt.models.kimi_k25_eagle3.RMSNorm",
                partial(RMSNorm, force_native=True),
            )
        )
        self.stack.enter_context(get_context().override_server_args())
        self.stack.enter_context(get_device().override(device="cpu"))
        self.stack.enter_context(
            get_parallel().override(
                tp_rank=0, attn_tp_rank=0, pp_group=SimpleNamespace()
            )
        )

    def test_unsharded_projection_without_distributed_initialization(self):
        # A size-one projection does not need a CUDA symmetric-memory group.
        with patch("sglang.srt.models.kimi_k25_eagle3._is_cuda", True):
            model = Eagle3MLAModel(small_config())
        self.assertIsNone(model._fc_gatherer)
        with torch.no_grad():
            model.fc.weight.copy_(
                torch.arange(model.fc.weight.numel()).reshape_as(model.fc.weight)
                / 10000
            )
        inputs = torch.arange(2 * 192, dtype=torch.float32).reshape(2, 192) / 100
        actual, _ = model.fc(inputs)
        torch.testing.assert_close(actual, F.linear(inputs, model.fc.weight))
        self.assertEqual(actual.shape, (2, 64))

    def test_checkpoint_qkv_fusion_preserves_projection_and_absorption(self):
        model = Eagle3DeepseekV2ForCausalLM(small_config(draft_vocab_size=128))
        attn = model.model.midlayer.self_attn
        q_weight = torch.arange(32 * 128, dtype=torch.float32).reshape(32, 128) / 1000
        kv_weight = torch.arange(48 * 128, dtype=torch.float32).reshape(48, 128) / 2000
        kv_b_weight = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32) / 1000
        weights = [
            ("layers.0.self_attn.kv_a_proj_with_mqa.weight", kv_weight),
            ("layers.0.self_attn.q_a_proj.weight", q_weight),
            ("layers.0.self_attn.kv_b_proj.weight", kv_b_weight),
        ]
        for index in range(3):
            weights.append((f"fc_norm.{index}.weight", torch.full((64,), index + 1.0)))
        model.load_weights(weights)
        inputs = torch.arange(2 * 128, dtype=torch.float32).reshape(2, 128) / 100
        actual = attn.prepare_qkv_latent(inputs, None)
        expected = torch.cat(
            [F.linear(inputs, q_weight), F.linear(inputs, kv_weight)], dim=-1
        )
        torch.testing.assert_close(actual, expected)
        latent = torch.arange(2 * 32, dtype=torch.float32).reshape(2, 32) / 100
        expanded = F.linear(latent, kv_b_weight).reshape(2, 2, 32)
        q_nope = torch.arange(2 * 2 * 16, dtype=torch.float32).reshape(2, 2, 16) / 100
        absorbed = torch.bmm(q_nope.transpose(0, 1), attn.w_kc).transpose(0, 1)
        torch.testing.assert_close(
            (q_nope * expanded[..., :16]).sum(-1),
            (absorbed * latent[:, None, :]).sum(-1),
        )
        values = torch.bmm(
            latent[:, None, :].expand(-1, 2, -1).transpose(0, 1), attn.w_vc
        ).transpose(0, 1)
        torch.testing.assert_close(values, expanded[..., 16:])
        for index, norm in enumerate(model.model.fc_norm):
            torch.testing.assert_close(norm.weight, torch.full((64,), index + 1.0))

    def test_auxiliary_norm_and_output_contract(self):
        for aux_count, fc_norm, norm_output in (
            (2, True, True),
            (3, True, False),
            (4, False, True),
        ):
            with self.subTest(
                aux_count=aux_count, fc_norm=fc_norm, norm_output=norm_output
            ):
                config = small_config(
                    target_hidden_size=48,
                    fc_norm=fc_norm,
                    norm_output=norm_output,
                    eagle_config={
                        "eagle_aux_hidden_state_layer_ids": list(range(aux_count))
                    },
                )
                model = Eagle3MLAModel(config)
                model.midlayer = ReferenceDecoder()
                generator = torch.Generator().manual_seed(42)
                with torch.no_grad():
                    model.fc.weight.copy_(
                        torch.randn(model.fc.weight.shape, generator=generator) * 0.02
                    )
                    model.norm.weight.fill_(1.25)
                    if fc_norm:
                        for index, norm in enumerate(model.fc_norm):
                            norm.weight.fill_(index + 1.0)
                features = torch.randn(5, 48 * aux_count, generator=generator)
                embeds = torch.randn(5, 64, generator=generator)
                expected_projected = reference_projection(model, features)
                output, aux = model(
                    torch.arange(5),
                    torch.arange(5),
                    SimpleNamespace(spec_info=SimpleNamespace(hidden_states=features)),
                    input_embeds=embeds,
                )
                torch.testing.assert_close(model.midlayer.projected, expected_projected)
                residual = expected_projected * 0.25 + embeds
                expected_output = F.rms_norm(
                    residual, (64,), weight=model.norm.weight, eps=config.rms_norm_eps
                )
                torch.testing.assert_close(output, expected_output)
                torch.testing.assert_close(
                    aux[0], expected_output if norm_output else residual
                )

    def test_projected_decode_and_empty_batch_skip_fc(self):
        model = Eagle3MLAModel(small_config())
        model.midlayer = ReferenceDecoder()
        with patch.object(
            model.fc,
            "forward",
            side_effect=AssertionError("fc called on draft features"),
        ):
            for tokens in (0, 3):
                hidden = torch.ones(tokens, 64)
                output, aux = model(
                    torch.arange(tokens),
                    torch.arange(tokens),
                    SimpleNamespace(spec_info=SimpleNamespace(hidden_states=hidden)),
                    input_embeds=torch.zeros_like(hidden),
                )
                self.assertEqual(output.shape, hidden.shape)
                self.assertEqual(aux[0].shape, hidden.shape)
                if tokens:
                    torch.testing.assert_close(model.midlayer.projected, hidden)

    def test_draft_vocabulary_mapping_and_shared_head(self):
        model = Eagle3DeepseekV2ForCausalLM(small_config(draft_vocab_size=8))
        difference = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        model.load_weights([("d2t", difference), ("t2d", torch.zeros(128))])
        torch.testing.assert_close(
            model.get_hot_token_id(), difference + torch.arange(8)
        )
        self.assertFalse(model.load_lm_head_from_target)
        shared = Eagle3DeepseekV2ForCausalLM(small_config(tie_word_embeddings=True))
        self.assertIs(shared.lm_head, shared.model.embed_tokens)
        inherited = Eagle3DeepseekV2ForCausalLM(small_config())
        self.assertTrue(inherited.load_lm_head_from_target)


if __name__ == "__main__":
    unittest.main()
