"""Unit tests for DeepseekV3ForCausalLMNextN fp8 ignore-list remap — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.layers.quantization import Fp8Config
from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.test.test_utils import CustomTestCase


def _bare_fp8_config(ignored_layers):
    """Skip __init__ — _resolve_nextn_quant_config only reads ignored_layers + use_mxfp8."""
    cfg = Fp8Config.__new__(Fp8Config)
    cfg.ignored_layers = list(ignored_layers)
    cfg.use_mxfp8 = False
    return cfg


def _stub_backend():
    """A bare backend carrying the class-level mapper the quark branch reads."""
    stub = MagicMock(spec=DeepseekV3ForCausalLMNextN)
    stub.hf_to_sglang_mapper = DeepseekV3ForCausalLMNextN.hf_to_sglang_mapper
    return stub


class TestNextnFp8IgnoreRemap(CustomTestCase):
    """_resolve_nextn_quant_config must mirror on-disk `model.layers.{N}.*` ignore-list
    entries onto the sglang `model.decoder.*` prefix the nextn layer is built with, so
    Fp8Config.is_layer_skipped matches and the un-quantized nextn attention stays bf16.

    Regression: a non-source-layer-count fp8 nextn checkpoint (nextn attention left
    bf16, keyed in the ignore-list by on-disk name `model.layers.{N}.*`) crashed at
    weight load with `Downcasting not allowed: target=fp8, loaded=bf16`, because the
    nextn layer is built with the sglang prefix `model.decoder.*` and
    is_layer_skipped never matched.
    """

    def test_non_source_layer_count_appends_decoder_variants(self):
        cfg = _bare_fp8_config(
            [
                "model.layers.45.self_attn.q_b_proj",
                "model.layers.45.self_attn.q_a_proj",
            ]
        )
        out = DeepseekV3ForCausalLMNextN._resolve_nextn_quant_config(
            _stub_backend(), SimpleNamespace(num_hidden_layers=45), cfg
        )
        added = [e for e in out.ignored_layers if e.startswith("model.decoder.")]
        self.assertEqual(
            sorted(added),
            ["model.decoder.self_attn.q_a_proj", "model.decoder.self_attn.q_b_proj"],
        )
        self.assertIn("model.layers.45.self_attn.q_b_proj", out.ignored_layers)
        self.assertIsNot(out, cfg)

    def test_source_layer_count_already_remapped_is_noop(self):
        cfg = _bare_fp8_config(["model.decoder.self_attn.q_b_proj"])
        out = DeepseekV3ForCausalLMNextN._resolve_nextn_quant_config(
            _stub_backend(), SimpleNamespace(num_hidden_layers=61), cfg
        )
        self.assertEqual(out.ignored_layers, ["model.decoder.self_attn.q_b_proj"])
        self.assertIs(out, cfg)

    def test_unrelated_layer_index_not_remapped(self):
        cfg = _bare_fp8_config(["model.layers.0.self_attn.q_proj"])
        out = DeepseekV3ForCausalLMNextN._resolve_nextn_quant_config(
            _stub_backend(), SimpleNamespace(num_hidden_layers=45), cfg
        )
        self.assertFalse(
            any(e.startswith("model.decoder.") for e in out.ignored_layers)
        )


if __name__ == "__main__":
    unittest.main()
