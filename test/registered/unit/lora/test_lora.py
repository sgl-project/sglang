"""Unit tests for srt/lora/lora.py - no server, no model loading."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora import LoRAAdapter  # noqa: E402
from sglang.srt.lora.lora_config import LoRAConfig  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestLoRAAdapter(CustomTestCase):
    def _make_adapter(self, target_modules, num_hidden_layers=2):
        config = LoRAConfig.from_dict(
            {
                "peft_type": "LORA",
                "target_modules": target_modules,
                "r": 2,
                "lora_alpha": 8,
            }
        )
        return LoRAAdapter(
            uid="test-lora",
            config=config,
            base_hf_config=SimpleNamespace(num_hidden_layers=num_hidden_layers),
            load_config=SimpleNamespace(),
            lora_backend=SimpleNamespace(name="triton"),
        )

    def test_initializes_layer_weights_from_tensors_and_normalizes_qkv(self):
        """Complete q/k/v LoRA weights are stacked into one qkv_proj entry."""
        adapter = self._make_adapter(["q_proj", "k_proj", "v_proj"])

        q_a = torch.full((2, 4), 1.0)
        k_a = torch.full((2, 4), 2.0)
        v_a = torch.full((2, 4), 3.0)
        q_b = torch.full((3, 2), 4.0)
        k_b = torch.full((3, 2), 5.0)
        v_b = torch.full((3, 2), 6.0)

        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": q_a,
                "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": k_a,
                "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": v_a,
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": q_b,
                "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": k_b,
                "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": v_b,
            }
        )

        layer_weights = adapter.layers[0].weights
        self.assertNotIn(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            layer_weights,
        )
        self.assertNotIn(
            "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight",
            layer_weights,
        )
        self.assertNotIn(
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight",
            layer_weights,
        )

        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.weight"
            ],
            torch.cat((q_a, k_a, v_a), dim=0),
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.weight"
            ],
            torch.cat((q_b, k_b, v_b), dim=0),
        )

    def test_qkv_normalization_fills_missing_k_proj_with_zeros(self):
        """Missing k_proj weights are zero-filled so qkv_proj keeps its layout."""
        adapter = self._make_adapter(["q_proj", "v_proj"])

        q_a = torch.full((2, 4), 1.0)
        v_a = torch.full((2, 4), 3.0)
        q_b = torch.full((3, 2), 4.0)
        v_b = torch.full((3, 2), 6.0)

        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": q_a,
                "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": v_a,
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": q_b,
                "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": v_b,
            }
        )

        layer_weights = adapter.layers[0].weights
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.weight"
            ],
            torch.cat((q_a, torch.zeros_like(v_a), v_a), dim=0),
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.weight"
            ],
            torch.cat((q_b, torch.zeros_like(v_b), v_b), dim=0),
        )

    def test_qkv_proj_already_stacked_repeats_lora_a_and_keeps_lora_b(self):
        """Already-stacked qkv_proj repeats LoRA A and keeps stacked LoRA B as-is."""
        adapter = self._make_adapter(["qkv_proj"])

        qkv_a = torch.full((2, 4), 1.0)
        qkv_b = torch.arange(18, dtype=torch.float32).reshape(9, 2)

        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.weight": qkv_a,
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.weight": qkv_b,
            }
        )

        layer_weights = adapter.layers[0].weights
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_A.weight"
            ],
            qkv_a.repeat(3, 1),
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.self_attn.qkv_proj.lora_B.weight"
            ],
            qkv_b,
        )

    def test_fuses_deepseek_mla_q_a_and_kv_a_proj_weights(self):
        """DeepSeek MLA q_a and kv_a LoRA weights are fused into one qkv_a entry."""
        adapter = self._make_adapter(["q_a_proj", "kv_a_proj_with_mqa"])

        q_a_lora_a = torch.full((2, 4), 1.0)
        kv_a_lora_a = torch.full((2, 4), 2.0)
        q_a_lora_b = torch.full((3, 2), 3.0)
        kv_a_lora_b = torch.full((3, 2), 4.0)

        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight": q_a_lora_a,
                "base_model.model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_A.weight": kv_a_lora_a,
                "base_model.model.model.layers.0.self_attn.q_a_proj.lora_B.weight": q_a_lora_b,
                "base_model.model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_B.weight": kv_a_lora_b,
            }
        )

        layer_weights = adapter.layers[0].weights
        self.assertNotIn(
            "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight",
            layer_weights,
        )
        self.assertNotIn(
            "base_model.model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_A.weight",
            layer_weights,
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.lora_A.weight"
            ],
            torch.cat((q_a_lora_a, kv_a_lora_a), dim=0),
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.lora_B.weight"
            ],
            torch.cat((q_a_lora_b, kv_a_lora_b), dim=0),
        )

    def test_normalizes_gate_up_proj_and_missing_up_proj(self):
        """gate_proj is stacked with a zero up_proj when the adapter omits up_proj."""
        adapter = self._make_adapter(["gate_proj"])

        gate_a = torch.full((2, 4), 1.0)
        gate_b = torch.full((5, 2), 2.0)

        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.model.layers.1.mlp.gate_proj.lora_A.weight": gate_a,
                "base_model.model.model.layers.1.mlp.gate_proj.lora_B.weight": gate_b,
            }
        )

        layer_weights = adapter.layers[1].weights
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.1.mlp.gate_up_proj.lora_A.weight"
            ],
            torch.cat((gate_a, torch.zeros_like(gate_a)), dim=0),
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.1.mlp.gate_up_proj.lora_B.weight"
            ],
            torch.cat((gate_b, torch.zeros_like(gate_b)), dim=0),
        )

    def test_gate_up_proj_already_stacked_repeats_lora_a_and_keeps_lora_b(self):
        """Already-stacked gate_up_proj repeats LoRA A and keeps stacked LoRA B as-is."""
        adapter = self._make_adapter(["gate_up_proj"])

        gate_up_a = torch.full((2, 4), 1.0)
        gate_up_b = torch.arange(20, dtype=torch.float32).reshape(10, 2)

        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.model.layers.1.mlp.gate_up_proj.lora_A.weight": gate_up_a,
                "base_model.model.model.layers.1.mlp.gate_up_proj.lora_B.weight": gate_up_b,
            }
        )

        layer_weights = adapter.layers[1].weights
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.1.mlp.gate_up_proj.lora_A.weight"
            ],
            gate_up_a.repeat(2, 1),
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.1.mlp.gate_up_proj.lora_B.weight"
            ],
            gate_up_b,
        )

    def test_renames_moe_expert_w_weights_before_gate_up_normalization(self):
        """MoE expert w1/w3/w2 names map to gate/up/down projection names."""
        adapter = self._make_adapter(["gate_proj", "up_proj", "down_proj"])

        w1_a = torch.full((2, 4), 1.0)
        w3_a = torch.full((2, 4), 2.0)
        w2_a = torch.full((2, 5), 3.0)
        w1_b = torch.full((5, 2), 4.0)
        w3_b = torch.full((5, 2), 5.0)
        w2_b = torch.full((4, 2), 6.0)

        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.model.layers.0.mlp.experts.0.w1.lora_A.weight": w1_a,
                "base_model.model.model.layers.0.mlp.experts.0.w3.lora_A.weight": w3_a,
                "base_model.model.model.layers.0.mlp.experts.0.w2.lora_A.weight": w2_a,
                "base_model.model.model.layers.0.mlp.experts.0.w1.lora_B.weight": w1_b,
                "base_model.model.model.layers.0.mlp.experts.0.w3.lora_B.weight": w3_b,
                "base_model.model.model.layers.0.mlp.experts.0.w2.lora_B.weight": w2_b,
            }
        )

        layer_weights = adapter.layers[0].weights
        self.assertNotIn(
            "base_model.model.model.layers.0.mlp.experts.0.w1.lora_A.weight",
            layer_weights,
        )
        self.assertNotIn(
            "base_model.model.model.layers.0.mlp.experts.0.w3.lora_A.weight",
            layer_weights,
        )
        self.assertNotIn(
            "base_model.model.model.layers.0.mlp.experts.0.w2.lora_A.weight",
            layer_weights,
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.mlp.experts.0.gate_up_proj.lora_A.weight"
            ],
            torch.cat((w1_a, w3_a), dim=0),
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.mlp.experts.0.gate_up_proj.lora_B.weight"
            ],
            torch.cat((w1_b, w3_b), dim=0),
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_A.weight"
            ],
            w2_a,
        )
        torch.testing.assert_close(
            layer_weights[
                "base_model.model.model.layers.0.mlp.experts.0.down_proj.lora_B.weight"
            ],
            w2_b,
        )

    def test_embedding_weights_are_filtered_by_target_modules_and_unembed_is_remapped(
        self,
    ):
        """Embedding weights respect target_modules and PEFT unembed names map to lm_head."""
        adapter = self._make_adapter(["lm_head"])

        embed_weight = torch.ones((2, 8))
        lm_head_weight = torch.full((8, 2), 2.0)

        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.model.embed_tokens.lora_A.weight": embed_weight,
                "base_model.model.unembed_tokens.lora_B.weight": lm_head_weight,
            }
        )

        self.assertNotIn(
            "base_model.model.model.embed_tokens.lora_A.weight",
            adapter.embedding_layers,
        )
        self.assertIn(
            "base_model.model.lm_head.lora_B.weight",
            adapter.embedding_layers,
        )
        torch.testing.assert_close(
            adapter.embedding_layers["base_model.model.lm_head.lora_B.weight"],
            lm_head_weight,
        )

    def test_scaling_uses_lora_alpha_over_rank(self):
        """The adapter scaling factor follows the PEFT lora_alpha / rank convention."""
        adapter = self._make_adapter(["q_proj"])
        self.assertEqual(adapter.scaling, 4.0)


if __name__ == "__main__":
    unittest.main()
