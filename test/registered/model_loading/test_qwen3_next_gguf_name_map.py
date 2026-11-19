# SPDX-License-Identifier: Apache-2.0

import unittest

from transformers import AutoConfig

from sglang.srt.model_loader.gguf_name_maps import (
    GGUF_HF_NAME_MAP_BUILDERS,
    build_qwen3_next_name_map,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_config(num_hidden_layers: int):
    return AutoConfig.for_model("qwen3_next", num_hidden_layers=num_hidden_layers)


class TestQwen3NextGGUFNameMap(unittest.TestCase):
    def test_registered_under_hf_model_type(self) -> None:
        self.assertIn("qwen3_next", GGUF_HF_NAME_MAP_BUILDERS)
        self.assertIs(
            GGUF_HF_NAME_MAP_BUILDERS["qwen3_next"], build_qwen3_next_name_map
        )

    def test_global_tensors(self) -> None:
        m = build_qwen3_next_name_map(_make_config(48))
        self.assertEqual(m["token_embd.weight"], "model.embed_tokens.weight")
        self.assertEqual(m["output_norm.weight"], "model.norm.weight")
        self.assertEqual(m["output.weight"], "lm_head.weight")

    def test_gdn_layer_mapping(self) -> None:
        m = build_qwen3_next_name_map(_make_config(48))
        # Layer 0 is a Gated-DeltaNet layer (full attention at N % 4 == 3).
        self.assertEqual(m["blk.0.ssm_a"], "model.layers.0.linear_attn.A_log")
        self.assertEqual(m["blk.0.ssm_dt.bias"], "model.layers.0.linear_attn.dt_bias")
        self.assertEqual(
            m["blk.0.ssm_in.weight"],
            "model.layers.0.linear_attn.in_proj_qkvz.weight",
        )
        self.assertEqual(
            m["blk.0.ssm_ba.weight"], "model.layers.0.linear_attn.in_proj_ba.weight"
        )
        self.assertEqual(
            m["blk.0.ssm_conv1d.weight"],
            "model.layers.0.linear_attn.conv1d.weight",
        )
        self.assertEqual(
            m["blk.0.ssm_norm.weight"], "model.layers.0.linear_attn.norm.weight"
        )
        self.assertEqual(
            m["blk.0.ssm_out.weight"], "model.layers.0.linear_attn.out_proj.weight"
        )
        # No full-attention tensors on a GDN layer.
        self.assertNotIn("blk.0.attn_q.weight", m)
        self.assertNotIn("blk.0.attn_output.weight", m)

    def test_full_attn_layer_mapping(self) -> None:
        m = build_qwen3_next_name_map(_make_config(48))
        # Layer 3 is a full-attention layer.
        self.assertEqual(
            m["blk.3.attn_q.weight"], "model.layers.3.self_attn.q_proj.weight"
        )
        self.assertEqual(
            m["blk.3.attn_k.weight"], "model.layers.3.self_attn.k_proj.weight"
        )
        self.assertEqual(
            m["blk.3.attn_v.weight"], "model.layers.3.self_attn.v_proj.weight"
        )
        self.assertEqual(
            m["blk.3.attn_output.weight"], "model.layers.3.self_attn.o_proj.weight"
        )
        self.assertEqual(
            m["blk.3.attn_q_norm.weight"], "model.layers.3.self_attn.q_norm.weight"
        )
        self.assertEqual(
            m["blk.3.attn_k_norm.weight"], "model.layers.3.self_attn.k_norm.weight"
        )
        # No GDN tensors on a full-attention layer.
        self.assertNotIn("blk.3.ssm_in.weight", m)
        self.assertNotIn("blk.3.ssm_a", m)

    def test_common_layer_tensors(self) -> None:
        m = build_qwen3_next_name_map(_make_config(48))
        for layer in (0, 3):
            self.assertEqual(
                m[f"blk.{layer}.attn_norm.weight"],
                f"model.layers.{layer}.input_layernorm.weight",
            )
            self.assertEqual(
                m[f"blk.{layer}.post_attention_norm.weight"],
                f"model.layers.{layer}.post_attention_layernorm.weight",
            )
            self.assertEqual(
                m[f"blk.{layer}.ffn_gate_inp.weight"],
                f"model.layers.{layer}.mlp.gate.weight",
            )
            self.assertEqual(
                m[f"blk.{layer}.ffn_gate_inp_shexp.weight"],
                f"model.layers.{layer}.mlp.shared_expert_gate.weight",
            )
            self.assertEqual(
                m[f"blk.{layer}.ffn_gate_shexp.weight"],
                f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
            )
            self.assertEqual(
                m[f"blk.{layer}.ffn_up_shexp.weight"],
                f"model.layers.{layer}.mlp.shared_expert.up_proj.weight",
            )
            self.assertEqual(
                m[f"blk.{layer}.ffn_down_shexp.weight"],
                f"model.layers.{layer}.mlp.shared_expert.down_proj.weight",
            )
            # Routed-expert packed tensors are handled by
            # gguf_quant_weights_iterator, never by the name map.
            for role in ("gate", "up", "down"):
                self.assertNotIn(f"blk.{layer}.ffn_{role}_exps.weight", m)

    def test_layer_type_split_and_bijection(self) -> None:
        cfg = _make_config(48)
        m = build_qwen3_next_name_map(cfg)
        gdn_layers = [n for n in range(48) if n % 4 != 3]
        full_layers = [n for n in range(48) if n % 4 == 3]
        self.assertEqual(len(gdn_layers), 36)
        self.assertEqual(len(full_layers), 12)
        # 3 globals + 36 x (7 common + 5 gdn + 2 raw) + 12 x (7 common + 6 attn)
        self.assertEqual(len(m), 3 + 36 * 14 + 12 * 13)
        # The map is injective: no two GGUF tensors target the same HF param.
        self.assertEqual(len(set(m.values())), len(m))

    def test_custom_full_attention_interval(self) -> None:
        # The attribute arrives from config.json on checkpoint load; it is not
        # a declared Qwen3NextConfig field, so set it explicitly here.
        cfg = _make_config(8)
        cfg.full_attention_interval = 2
        m = build_qwen3_next_name_map(cfg)
        # With interval 2, layers 1, 3, 5, 7 are full attention.
        self.assertIn("blk.1.attn_q.weight", m)
        self.assertNotIn("blk.1.ssm_in.weight", m)
        self.assertIn("blk.0.ssm_in.weight", m)
        self.assertNotIn("blk.0.attn_q.weight", m)


if __name__ == "__main__":
    unittest.main()
