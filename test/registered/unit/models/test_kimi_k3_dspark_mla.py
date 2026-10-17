"""Unit coverage for the native Kimi-K3 MLA DSpark draft model."""

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestKimiK3DSparkMLA(CustomTestCase):
    def test_registry_resolves_native_mla_draft(self):
        from sglang.srt.models.registry import ModelRegistry

        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            "KimiK3DSparkMLADraftModel"
        )

        self.assertEqual(resolved_arch, "KimiK3DSparkMLADraftModel")
        self.assertEqual(model_cls.__name__, "KimiK3DSparkMLADraftModel")
        self.assertEqual(model_cls.__module__, "sglang.srt.models.kimi_k3_dspark_mla")

    def test_model_config_uses_mla_head_shapes(self):
        config = object.__new__(ModelConfig)
        config.hf_config = SimpleNamespace(
            architectures=["KimiK3DSparkMLADraftModel"],
            model_type="kimi_k3_dspark",
        )
        config.hf_text_config = SimpleNamespace(
            hidden_size=4096,
            num_attention_heads=32,
            num_hidden_layers=2,
            vocab_size=163840,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
            kv_lora_rank=512,
            rope_scaling=None,
        )

        config._derive_model_shapes()

        self.assertEqual(config.attention_arch, AttentionArch.MLA)
        self.assertEqual(config.head_dim, 96)
        self.assertEqual(config.qk_nope_head_dim, 64)
        self.assertEqual(config.qk_rope_head_dim, 32)
        self.assertEqual(config.v_head_dim, 64)
        self.assertEqual(config.kv_lora_rank, 512)
        self.assertAlmostEqual(config.scaling, 1 / math.sqrt(96))

    def test_loader_fuses_q_a_and_kv_a_in_either_order(self):
        from sglang.srt.models.kimi_k3_dspark_mla import (
            KimiK3DSparkMLADraftModel,
        )

        model = KimiK3DSparkMLADraftModel.__new__(KimiK3DSparkMLADraftModel)
        nn.Module.__init__(model)

        fused = nn.Parameter(torch.empty(5, 3))
        q_b = nn.Parameter(torch.empty(4, 2))
        params_dict = {
            "layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight": fused,
            "layers.0.self_attn.q_b_proj.weight": q_b,
        }
        q_a_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        kv_a_weight = torch.arange(9, dtype=torch.float32).reshape(3, 3) + 20
        q_b_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 40

        model._load_backbone_weights(
            [
                (
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                    kv_a_weight,
                ),
                ("model.layers.0.self_attn.q_b_proj.weight", q_b_weight),
                ("model.layers.0.self_attn.q_a_proj.weight", q_a_weight),
            ],
            params_dict,
        )

        torch.testing.assert_close(fused, torch.cat([q_a_weight, kv_a_weight]))
        torch.testing.assert_close(q_b, q_b_weight)

    def test_loader_rejects_an_incomplete_a_projection_pair(self):
        from sglang.srt.models.kimi_k3_dspark_mla import (
            KimiK3DSparkMLADraftModel,
        )

        model = KimiK3DSparkMLADraftModel.__new__(KimiK3DSparkMLADraftModel)
        nn.Module.__init__(model)
        params_dict = {
            "layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight": nn.Parameter(
                torch.empty(5, 3)
            )
        }

        with self.assertRaisesRegex(ValueError, "incomplete q_a/kv_a"):
            model._load_backbone_weights(
                [
                    (
                        "layers.0.self_attn.q_a_proj.weight",
                        torch.empty(2, 3),
                    )
                ],
                params_dict,
            )


if __name__ == "__main__":
    unittest.main()
