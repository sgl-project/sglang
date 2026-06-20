"""Unit tests for the native Qwen3.5 MoE MLX model module.

Coverage:
  1. Module imports cleanly (`Model`, `TextModelArgs`).
  2. `TextModelArgs` dataclass constructs with sane defaults matching
     Qwen3.6-35B-A3B (40 layers, 256 experts / 8 active, hybrid
     attention interval=4, partial_rotary=0.25).
  3. `TextModelArgs.from_hf_config` parses an HF `text_config` block,
     including `rope_parameters` and `layer_types`.
  4. `Model` instantiates with a small config; `model_type` and the
     per-layer attention kind (`is_linear`) match `layer_types`.
  5. `sanitize` drops `mtp.*` / `model.visual.*`, transposes `conv1d`,
     and shifts RMSNorm weights by `+1` only when the file is in the
     un-sanitised HF layout.
  6. `quant_predicate` returns a callable that forces 8-bit on the
     router and shared-expert gate.
  7. `_is_qwen3_5_moe` (in `model_runner.py`) detects the architecture
     from config.json in three flavours: multimodal
     `Qwen3_5MoeForConditionalGeneration`, text-only
     `qwen3_5_moe` model_type, and a negative case.

Tests use a small (2-layer, 4-expert) config for model construction
and forward so the test process does not OOM on the 35B random init.
"""

import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.hardware_backend.mlx.models.qwen3_5_moe import (
    Attention,
    DecoderLayer,
    GatedDeltaNet,
    Model,
    SparseMoeBlock,
    SwitchGLU,
    TextModelArgs,
    load,
)
from sglang.srt.hardware_backend.mlx.model_runner import _is_qwen3_5_moe
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="unit-test-mlx")


# A small config that keeps random-init memory well under test limits
# (2 layers, 4 experts, hidden_size=128, intermediate=64).
SMALL = dict(
    num_hidden_layers=2,
    hidden_size=128,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    vocab_size=64,
    layer_types=["linear_attention", "full_attention"],
)


class TestTextModelArgs(unittest.TestCase):
    def test_import(self):
        self.assertTrue(callable(Model))
        self.assertTrue(callable(TextModelArgs))

    def test_default_args_match_qwen3_6_35b_a3b(self):
        args = TextModelArgs()
        self.assertEqual(args.model_type, "qwen3_5_moe_text")
        self.assertEqual(args.num_hidden_layers, 40)
        self.assertEqual(args.num_experts, 256)
        self.assertEqual(args.num_experts_per_tok, 8)
        self.assertEqual(args.full_attention_interval, 4)
        self.assertEqual(args.partial_rotary_factor, 0.25)
        self.assertTrue(args.attn_output_gate)
        self.assertEqual(args.moe_intermediate_size, 512)
        self.assertEqual(args.shared_expert_intermediate_size, 512)
        self.assertEqual(args.linear_num_value_heads, 32)
        self.assertEqual(args.linear_key_head_dim, 128)
        self.assertEqual(args.linear_conv_kernel_dim, 4)

    def test_from_hf_config_parses_rope_parameters(self):
        text_cfg = {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention"] * 39 + ["full_attention"],
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
        }
        args = TextModelArgs.from_hf_config(text_cfg)
        self.assertEqual(args.rope_theta, 10000000)
        self.assertEqual(args.partial_rotary_factor, 0.25)
        self.assertEqual(args.rope_parameters["mrope_section"], [11, 11, 10])
        self.assertEqual(args.layer_types[-1], "full_attention")


class TestModelConstruction(unittest.TestCase):
    def test_model_construction_small(self):
        args = TextModelArgs(**SMALL)
        m = Model(args)
        self.assertEqual(m.model_type, "qwen3_5_moe_text")
        self.assertEqual(len(m.layers), 2)
        self.assertIsNotNone(m.args)

    def test_layer_attention_kind_matches_layer_types(self):
        args = TextModelArgs(
            num_hidden_layers=4,
            hidden_size=128,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            vocab_size=64,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        )
        m = Model(args)
        self.assertTrue(m.layers[0].is_linear)
        self.assertTrue(m.layers[1].is_linear)
        self.assertTrue(m.layers[2].is_linear)
        self.assertFalse(m.layers[3].is_linear)
        self.assertIsInstance(m.layers[0].linear_attn, GatedDeltaNet)
        self.assertIsInstance(m.layers[3].self_attn, Attention)

    def test_layer_attention_kind_from_interval_fallback(self):
        # No layer_types — derive from full_attention_interval=4.
        args = TextModelArgs(
            num_hidden_layers=4,
            hidden_size=128,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            linear_num_value_heads=4,
            linear_num_key_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            vocab_size=64,
            full_attention_interval=4,
        )
        m = Model(args)
        self.assertTrue(m.layers[0].is_linear)
        self.assertTrue(m.layers[1].is_linear)
        self.assertTrue(m.layers[2].is_linear)
        self.assertFalse(m.layers[3].is_linear)

    def test_decoder_layer_composition(self):
        args = TextModelArgs(**SMALL)
        m = Model(args)
        layer0 = m.layers[0]
        self.assertIsInstance(layer0, DecoderLayer)
        self.assertIsInstance(layer0.linear_attn, GatedDeltaNet)
        self.assertIsInstance(layer0.mlp, SparseMoeBlock)
        self.assertIsInstance(layer0.mlp.switch_mlp, SwitchGLU)

    def test_quant_predicate_routes_mlp_gate_to_8bit(self):
        args = TextModelArgs(**SMALL)
        m = Model(args)
        predicate = m.quant_predicate
        self.assertTrue(callable(predicate))
        self.assertEqual(predicate("model.layers.0.mlp.gate", None), {"group_size": 64, "bits": 8})
        self.assertEqual(predicate("model.layers.0.mlp.shared_expert_gate", None), {"group_size": 64, "bits": 8})
        self.assertTrue(predicate("model.layers.0.mlp.switch_mlp.gate_proj", None))
        self.assertTrue(predicate("model.layers.0.self_attn.q_proj", None))


class TestSanitize(unittest.TestCase):
    def test_sanitize_drops_mtp_and_visual(self):
        args = TextModelArgs(**SMALL)
        m = Model(args)
        weights = {
            "model.layers.0.mlp.gate.weight": "x",
            "model.layers.0.self_attn.q_proj.weight": "y",
            "mtp.layers.0.weight": "drop",
            "model.visual.blocks.0.weight": "drop",
        }
        sanitized = m.sanitize(weights)
        self.assertIn("model.layers.0.mlp.gate.weight", sanitized)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", sanitized)
        self.assertNotIn("mtp.layers.0.weight", sanitized)
        self.assertNotIn("model.visual.blocks.0.weight", sanitized)

    def test_sanitize_transposes_conv1d_when_unsanitised(self):
        import mlx.core as mx

        args = TextModelArgs(**SMALL)
        m = Model(args)
        # (out, k, in) layout with last-dim != 1 means "needs transpose".
        weight = mx.zeros((4, 16, 3))
        weights = {
            "model.layers.0.linear_attn.conv1d.weight": weight,
            "model.norm.weight": mx.zeros((128,)),
        }
        sanitized = m.sanitize(weights)
        self.assertEqual(sanitized["model.layers.0.linear_attn.conv1d.weight"].shape, (4, 3, 16))

    def test_sanitize_does_not_shift_norms_when_already_sanitised(self):
        import mlx.core as mx

        args = TextModelArgs(**SMALL)
        m = Model(args)
        # Already-sanitised conv1d (last-dim == 1) and no mtp -> no norm shift.
        weights = {
            "model.layers.0.linear_attn.conv1d.weight": mx.zeros((4, 3, 1)),
            "model.norm.weight": mx.zeros((128,)),
        }
        sanitized = m.sanitize(weights)
        self.assertTrue(
            mx.allclose(sanitized["model.norm.weight"], mx.zeros((128,))).item()
        )

    def test_sanitize_shifts_norms_when_mtp_present(self):
        import mlx.core as mx

        args = TextModelArgs(**SMALL)
        m = Model(args)
        norm = mx.zeros((128,))
        weights = {
            "model.layers.0.linear_attn.conv1d.weight": mx.zeros((4, 3, 16)),
            "model.norm.weight": norm,
            "mtp.layers.0.weight": "trigger",
        }
        sanitized = m.sanitize(weights)
        self.assertTrue(
            mx.allclose(sanitized["model.norm.weight"], mx.zeros((128,)) + 1.0).item()
        )
        self.assertNotIn("mtp.layers.0.weight", sanitized)

    def test_sanitize_drops_lm_head_when_tied(self):
        args = TextModelArgs(**SMALL, tie_word_embeddings=True)
        m = Model(args)
        weights = {
            "lm_head.weight": "drop",
            "model.embed_tokens.weight": "keep",
        }
        sanitized = m.sanitize(weights)
        self.assertNotIn("lm_head.weight", sanitized)
        self.assertIn("model.embed_tokens.weight", sanitized)


class TestLoadMissingFiles(unittest.TestCase):
    def test_load_raises_on_missing_safetensors(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_hidden_layers": 2,
                    "hidden_size": 128,
                    "intermediate_size": 64,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 32,
                    "linear_num_value_heads": 4,
                    "linear_num_key_heads": 2,
                    "linear_key_head_dim": 16,
                    "linear_value_head_dim": 16,
                    "linear_conv_kernel_dim": 4,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 64,
                    "shared_expert_intermediate_size": 64,
                    "vocab_size": 64,
                },
            }))
            with self.assertRaises(FileNotFoundError):
                load(d)


class TestIsQwen3_5MoEDetection(unittest.TestCase):
    def _write_config(self, dirpath: str, cfg: dict):
        Path(dirpath, "config.json").write_text(json.dumps(cfg))

    def test_detects_multimodal_arch(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, {
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "text_config": {"model_type": "qwen3_5_moe_text"},
            })
            self.assertTrue(_is_qwen3_5_moe(d))

    def test_detects_text_only_model_type(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, {"model_type": "qwen3_5_moe"})
            self.assertTrue(_is_qwen3_5_moe(d))

    def test_detects_text_config_model_type(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, {
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe_text"},
            })
            self.assertTrue(_is_qwen3_5_moe(d))

    def test_rejects_unrelated_arch(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
            })
            self.assertFalse(_is_qwen3_5_moe(d))

    def test_rejects_missing_config(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(_is_qwen3_5_moe(d))

    def test_rejects_malformed_config(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text("not valid json {{")
            self.assertFalse(_is_qwen3_5_moe(d))


if __name__ == "__main__":
    unittest.main()
