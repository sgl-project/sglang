"""CPU regressions for Laguna S 2.1 target and DFlash model wiring."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.configs.laguna import LagunaConfig, normalize_gating
from sglang.srt.layers.attention.triton_backend import get_max_attention_heads
from sglang.srt.models.dflash import (
    DFlashAttention,
    DFlashLagunaAttention,
    DFlashLagunaForCausalLM,
)
from sglang.srt.models.laguna import (
    LagunaForCausalLM,
    LagunaModel,
    LagunaRMSNorm,
)
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.utils.hf_transformers import get_config
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer


class _FakePPGroup:
    is_first_rank = True
    is_last_rank = True


class _AddOneLayer(nn.Module):
    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        residual,
        captured_last_layer_outputs=None,
    ):
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.append(
                hidden_states + residual if residual is not None else hidden_states
            )
        return hidden_states + 1, residual


class _Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, value):
        return value * self.factor


class _Projection(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, hidden_states):
        return self.value.expand(hidden_states.shape[0], -1), None


class TestLagunaConfig(unittest.TestCase):
    def test_hf_yarn_attention_factor_is_not_applied_twice(self):
        factor = 32.0
        expected_attention_factor = 1.0 + 0.1 * math.log(factor)
        config = LagunaConfig(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_attention_heads_per_layer=[2],
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["full_attention"],
            mlp_layer_types=["dense"],
            rope_parameters={
                "full_attention": {
                    "rope_type": "yarn",
                    "factor": factor,
                    "attention_factor": expected_attention_factor,
                }
            },
        )

        self.assertAlmostEqual(config.full_rope_scaling["attn_factor"], 1.0)

    def test_decode_scratch_uses_largest_per_layer_head_count(self):
        model_config = SimpleNamespace(
            num_attention_heads=48,
            hf_text_config=SimpleNamespace(
                num_attention_heads_per_layer=[48, 72, 72, 72]
            ),
        )

        self.assertEqual(get_max_attention_heads(model_config), 72)

    def test_sglang_config_preempts_strict_transformers_native_loader(self):
        raw_config = {
            "architectures": ["LagunaForCausalLM"],
            "model_type": "laguna",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_attention_heads_per_layer": [2, 2],
            "num_key_value_heads": 1,
            "head_dim": 8,
            "layer_types": ["full_attention", "sliding_attention"],
            "mlp_layer_types": ["dense", "sparse"],
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "yarn",
                    "factor": 32.0,
                    "rope_theta": 500_000.0,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 1.0,
                },
            },
        }
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(json.dumps(raw_config))
            config = get_config(model_dir, trust_remote_code=False)

        self.assertIs(type(config), LagunaConfig)
        self.assertEqual(config.full_rope_scaling["rope_type"], "yarn")
        self.assertEqual(config.swa_rope_theta, 10_000.0)

    def test_tokenizer_reuses_sglang_config_and_fixes_mistral_regex(self):
        raw_config = {"model_type": "laguna"}
        compatibility_config = object()
        tokenizer = object()
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(json.dumps(raw_config))
            with (
                patch(
                    "sglang.srt.utils.hf_transformers.tokenizer.get_config",
                    return_value=compatibility_config,
                ),
                patch(
                    "sglang.srt.utils.hf_transformers.tokenizer."
                    "_auto_tokenizer_from_pretrained",
                    return_value=tokenizer,
                ) as from_pretrained,
                patch(
                    "sglang.srt.utils.hf_transformers.tokenizer."
                    "_apply_post_load_fixes",
                    side_effect=lambda value, *_: value,
                ),
            ):
                loaded = get_tokenizer(model_dir, trust_remote_code=False)

        self.assertIs(loaded, tokenizer)
        self.assertIs(
            from_pretrained.call_args.kwargs["config"], compatibility_config
        )
        self.assertTrue(from_pretrained.call_args.kwargs["fix_mistral_regex"])

    def test_dflash_pure_swa_config_uses_layer_zero_geometry(self):
        config = LagunaConfig(
            hidden_size=3072,
            num_hidden_layers=6,
            num_attention_heads=72,
            num_attention_heads_per_layer=[72] * 6,
            num_key_value_heads=8,
            head_dim=128,
            layer_types=["sliding_attention"] * 6,
            mlp_layer_types=["dense"] * 6,
            rope_theta=500_000.0,
            partial_rotary_factor=0.5,
            gating="per-head",
        )

        self.assertEqual(config.num_attention_heads, 72)
        self.assertEqual(config.swa_num_key_value_heads, 8)
        self.assertEqual(config.swa_partial_rotary_factor, 0.5)
        self.assertEqual(config.gating, "per-head")

    def test_gating_values_are_normalized_and_validated(self):
        self.assertEqual(normalize_gating(True), "per-head")
        self.assertEqual(normalize_gating("per-element"), "per-element")
        self.assertEqual(normalize_gating(False), "disabled")
        with self.assertRaises(ValueError):
            normalize_gating("unknown")


class TestLagunaDFlash(unittest.TestCase):
    @staticmethod
    def _make_weight_loader_shell():
        model = object.__new__(LagunaForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            mlp_layer_types=[],
            num_experts=0,
            tie_word_embeddings=False,
        )
        model.model = nn.Module()
        model.model.start_layer = 0
        model.model.end_layer = 1
        return model

    def test_target_ignores_unused_checkpoint_kv_cache_scales(self):
        model = self._make_weight_loader_shell()

        with patch("sglang.srt.models.laguna.logger.warning") as warning:
            model.load_weights(
                [
                    (
                        "model.layers.0.self_attn.k_scale",
                        torch.tensor([0.03125]),
                    ),
                    (
                        "model.layers.0.self_attn.v_scale",
                        torch.tensor([0.00125]),
                    ),
                ]
            )

        warning.assert_not_called()

    def test_target_loads_checkpoint_kv_scales_when_attention_registers_them(self):
        model = self._make_weight_loader_shell()
        model.model.layers = nn.ModuleList([nn.Module()])
        model.model.layers[0].self_attn = nn.Module()
        model.model.layers[0].self_attn.attn = nn.Module()
        model.model.layers[0].self_attn.attn.k_scale = nn.Parameter(
            torch.tensor(-1.0), requires_grad=False
        )
        model.model.layers[0].self_attn.attn.v_scale = nn.Parameter(
            torch.tensor(-1.0), requires_grad=False
        )

        model.load_weights(
            [
                (
                    "model.layers.0.self_attn.k_scale",
                    torch.tensor([0.03125]),
                ),
                (
                    "model.layers.0.self_attn.v_scale",
                    torch.tensor([0.00125]),
                ),
            ]
        )

        self.assertEqual(
            model.model.layers[0].self_attn.attn.k_scale.item(), 0.03125
        )
        self.assertAlmostEqual(
            model.model.layers[0].self_attn.attn.v_scale.item(), 0.00125
        )

    def test_v100_residual_emulates_bf16_without_overflow(self):
        with patch(
            "sglang.srt.models.laguna.get_laguna_wide_output_scale",
            return_value=4.0,
        ):
            norm = LagunaRMSNorm(3)
        norm.weight.data = norm.weight.data.half().fill_(1)

        branch = torch.tensor([[10_000.0, -10_000.0, 5_000.0]], dtype=torch.float16)
        residual = torch.tensor(
            [[40_000.0, -40_000.0, 20_000.0]], dtype=torch.float32
        )
        normalized, wide_residual = norm.forward_native(branch, residual)

        expected_residual = (
            branch.float().mul(4).add(residual).to(torch.bfloat16).float()
        )
        variance = expected_residual.square().mean(dim=-1, keepdim=True)
        expected = expected_residual * torch.rsqrt(variance + 1e-6)
        expected = expected.to(torch.bfloat16).float()
        torch.testing.assert_close(
            wide_residual, expected_residual, rtol=0, atol=0
        )
        self.assertEqual(wide_residual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(wide_residual).all())
        torch.testing.assert_close(
            normalized.float(), expected, rtol=2e-3, atol=2e-3
        )

    def test_draft_attention_forwards_partial_rotary_factor(self):
        config = SimpleNamespace(
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_bias=False,
            max_position_embeddings=1024,
            rope_theta=500_000.0,
            rope_scaling=None,
            rope_is_neox_style=True,
            partial_rotary_factor=0.5,
        )
        module_factory = lambda *args, **kwargs: nn.Identity()
        with (
            patch(
                "sglang.srt.models.dflash.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.models.dflash.QKVParallelLinear",
                side_effect=module_factory,
            ),
            patch(
                "sglang.srt.models.dflash.RowParallelLinear",
                side_effect=module_factory,
            ),
            patch(
                "sglang.srt.models.dflash.RMSNorm",
                side_effect=module_factory,
            ),
            patch(
                "sglang.srt.models.dflash.RadixAttention",
                side_effect=module_factory,
            ),
            patch(
                "sglang.srt.models.dflash.get_rope",
                return_value=nn.Identity(),
            ) as get_rope,
        ):
            DFlashAttention(config, layer_id=0)

        self.assertEqual(
            get_rope.call_args.kwargs["partial_rotary_factor"],
            0.5,
        )

    def test_dflash_architecture_is_registered(self):
        model_class, architecture = ModelRegistry.resolve_model_cls(
            ["DFlashLagunaForCausalLM"]
        )

        self.assertIs(model_class, DFlashLagunaForCausalLM)
        self.assertEqual(architecture, "DFlashLagunaForCausalLM")
        self.assertFalse(model_class.supports_fused_context_kv)

    def test_target_captures_requested_post_layer_hidden_states(self):
        model = object.__new__(LagunaModel)
        nn.Module.__init__(model)
        model.pp_group = _FakePPGroup()
        model.embed_tokens = nn.Identity()
        model.layers = nn.ModuleList([_AddOneLayer() for _ in range(3)])
        model.start_layer = 0
        model.end_layer = 3
        model.norm = nn.Identity()
        model.layers_to_capture = [2, 3]

        hidden, captured = model.forward(
            input_ids=torch.empty(0, dtype=torch.int64),
            positions=torch.arange(2),
            forward_batch=SimpleNamespace(),
            input_embeds=torch.ones(2, 4),
        )

        torch.testing.assert_close(hidden, torch.full((2, 4), 4.0))
        self.assertEqual(len(captured), 2)
        torch.testing.assert_close(captured[0], torch.full((2, 4), 3.0))
        torch.testing.assert_close(captured[1], torch.full((2, 4), 4.0))

    def test_target_layer_ids_are_converted_to_capture_boundaries(self):
        model = object.__new__(LagunaForCausalLM)
        model.pp_group = _FakePPGroup()
        model.model = SimpleNamespace(layers_to_capture=[])
        model.capture_aux_hidden_states = False

        model.set_dflash_layers_to_capture([1, 10, 19, 29, 38, 47])

        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.layers_to_capture, [2, 11, 20, 30, 39, 48])

    def test_draft_normalizes_each_target_feature_before_projection(self):
        model = object.__new__(DFlashLagunaForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=2)
        model.num_context_features = 2
        model.fc = nn.Linear(4, 2, bias=False, dtype=torch.float16)
        with torch.no_grad():
            model.fc.weight.copy_(
                torch.tensor(
                    [[1, 0, 0, 0], [0, 0, 1, 0]],
                    dtype=torch.float16,
                )
            )
        model.hidden_norm = nn.Identity()
        model.aux_hidden_norms = nn.ModuleList([_Scale(2), _Scale(3)])

        result = model.project_target_hidden(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16)
        )

        self.assertEqual(result.dtype, torch.float16)
        torch.testing.assert_close(result, torch.tensor([[2, 9]], dtype=torch.float16))

    def test_draft_attention_applies_per_head_softplus_gate(self):
        attention = object.__new__(DFlashLagunaAttention)
        nn.Module.__init__(attention)
        attention.g_proj = _Projection(torch.zeros(2))
        attention.gate_per_head = True
        attention.num_heads = 2
        attention.head_dim = 2

        result = attention.apply_attention_output(torch.ones(1, 4), torch.ones(1, 3))

        torch.testing.assert_close(
            result,
            torch.full((1, 4), torch.log(torch.tensor(2.0))),
        )

    def test_draft_kv_materialization_uses_layer_input_norm(self):
        worker = object.__new__(DFlashWorkerV2)
        normalized = torch.full((2, 4), 7.0)
        attention = SimpleNamespace(
            kv_proj_only=Mock(return_value=(torch.ones(2, 4), torch.ones(2, 4))),
            apply_k_norm=Mock(side_effect=lambda value: value),
            apply_k_rope=Mock(side_effect=lambda positions, value: value),
            num_kv_heads=1,
            head_dim=4,
            attn=SimpleNamespace(k_scale=None, v_scale=None),
        )
        layer = SimpleNamespace(self_attn=attention)
        worker.draft_model = SimpleNamespace(
            layers=[layer],
            prepare_context_hidden_for_kv=Mock(return_value=normalized),
        )
        set_kv_buffer = Mock()
        worker.draft_model_runner = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(set_kv_buffer=set_kv_buffer)
        )

        worker._append_target_hidden_sequential(
            ctx_hidden=torch.ones(2, 4),
            ctx_positions=torch.arange(2),
            ctx_cache_loc=torch.arange(2),
        )

        attention.kv_proj_only.assert_called_once_with(normalized)
        set_kv_buffer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
