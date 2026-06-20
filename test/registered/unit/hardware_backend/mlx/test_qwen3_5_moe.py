"""Unit tests for the native Qwen3.5 MoE MLX model module.

Phase 1 MVP coverage:
  1. Module imports cleanly (`Model`, `TextModelArgs`)
  2. `TextModelArgs` dataclass constructs with sane defaults matching
     Qwen3.6-35B-A3B (40 layers, 256 experts / 8 active, hybrid
     attention interval=4, partial_rotary=0.25)
  3. `Model` instantiates, exposes `model_type`, empty `layers` list,
     and a passthrough `sanitize`
  4. `quant_predicate` returns a callable that always accepts
  5. `_is_qwen3_5_moe` (in `model_runner.py`) detects the architecture
     from config.json in three flavours: multimodal
     `Qwen3_5MoeForConditionalGeneration`, text-only
     `qwen3_5_moe` model_type, and a negative case

The tests do NOT exercise real model weights — that arrives in
Phase 2 (text-only Qwen3.5 MoE full implementation) and Phase 4
(vision tower).  These tests guarantee the wiring is correct so
Phase 2 can focus on the architecture itself.
"""

import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.hardware_backend.mlx.models.qwen3_5_moe import (
    Model,
    TextModelArgs,
)
from sglang.srt.hardware_backend.mlx.model_runner import _is_qwen3_5_moe
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="unit-test-mlx")


class TestQwen3_5MoEStub(unittest.TestCase):
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

    def test_rope_parameters_default(self):
        args = TextModelArgs()
        self.assertEqual(args.rope_parameters["mrope_section"], [11, 11, 10])
        self.assertEqual(args.rope_parameters["partial_rotary_factor"], 0.25)
        self.assertEqual(args.rope_parameters["rope_theta"], 10000000)

    def test_model_construction(self):
        m = Model(TextModelArgs())
        self.assertEqual(m.model_type, "qwen3_5_moe_text")
        self.assertEqual(m.layers, [])
        self.assertIsNotNone(m.args)

    def test_sanitize_passthrough(self):
        m = Model(TextModelArgs())
        weights = {"a.b.c": 1, "d.e": 2}
        self.assertEqual(m.sanitize(weights), weights)

    def test_quant_predicate_accepts_all(self):
        m = Model(TextModelArgs())
        predicate = m.quant_predicate
        self.assertTrue(callable(predicate))
        self.assertTrue(predicate("any.weight", None))
        self.assertTrue(predicate("mlp.experts.0.gate_proj.weight", None))


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
