"""Unit tests for KimiW4AFp8Config and related functionality.

Tests cover:
  - Config parsing (from_config) with various quant_config.json layouts
  - Layer name normalization (model. prefix handling)
  - Unquantized-layer auto-detection (lm_head)
  - get_quant_method() routing for LinearBase, FusedMoE, and other layers
  - make_expert_input_scale_params_mapping() with custom naming
  - Registration in BASE_QUANTIZATION_METHODS
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.quantization.w4afp8 import KimiW4AFp8Config, W4AFp8Config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kimi_quant_config(**overrides):
    """Return a typical Kimi K2.5 quant_config dict, with optional overrides."""
    cfg = {
        "quant_method": "kimi_w4afp8",
        "linear_activation_scheme": "dynamic",
        "moe_activation_scheme": "dynamic",
        "group_size": 128,
        "weight_block_size": [128, 128],
        "ignore": [
            "model.layers.0.self_attn",
            "model.layers.0.mlp.shared_experts",
        ],
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Tests: KimiW4AFp8Config.from_config()
# ---------------------------------------------------------------------------


class TestKimiW4AFp8ConfigFromConfig(CustomTestCase):
    """Test the from_config() class method with various quant_config layouts."""

    def test_basic_parsing(self):
        """All fields present — should be read correctly."""
        cfg = _make_kimi_quant_config()
        config = KimiW4AFp8Config.from_config(cfg)

        self.assertEqual(config.get_name(), "kimi_w4afp8")
        self.assertTrue(config.is_checkpoint_w4afp8_serialized)
        self.assertEqual(config.linear_activation_scheme, "dynamic")
        self.assertEqual(config.moe_activation_scheme, "dynamic")
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.weight_block_size, [128, 128])

    def test_static_activation_scheme(self):
        """Static activation scheme should be accepted."""
        cfg = _make_kimi_quant_config(moe_activation_scheme="static")
        config = KimiW4AFp8Config.from_config(cfg)
        self.assertEqual(config.moe_activation_scheme, "static")

    def test_defaults_when_fields_missing(self):
        """Minimal config — only quant_method required."""
        cfg = {"quant_method": "kimi_w4afp8"}
        config = KimiW4AFp8Config.from_config(cfg)

        self.assertEqual(config.linear_activation_scheme, "dynamic")
        self.assertEqual(config.moe_activation_scheme, "static")
        self.assertEqual(config.group_size, 128)
        self.assertEqual(config.weight_block_size, [128, 128])
        self.assertEqual(config.ignored_layers, [])
        self.assertEqual(config.unquantized_layers, [])

    def test_ignored_layers_normalization(self):
        """Ignored layers should be normalized to include both model. and non-model. variants."""
        cfg = _make_kimi_quant_config(
            ignore=["model.layers.0.self_attn", "embed_tokens"]
        )
        config = KimiW4AFp8Config.from_config(cfg)

        # Both with and without "model." prefix should be present
        self.assertIn("layers.0.self_attn", config.ignored_layers)
        self.assertIn("model.layers.0.self_attn", config.ignored_layers)
        self.assertIn("embed_tokens", config.ignored_layers)
        self.assertIn("model.embed_tokens", config.ignored_layers)

    def test_alternative_ignore_key_names(self):
        """The config can use 'ignored_layers' or 'modules_to_not_convert'."""
        for key in ["ignored_layers", "modules_to_not_convert"]:
            cfg = {"quant_method": "kimi_w4afp8", key: ["lm_head"]}
            config = KimiW4AFp8Config.from_config(cfg)
            self.assertIn("lm_head", config.ignored_layers, f"key={key}")

    def test_unquantized_layers_explicit(self):
        """Explicitly specified unquantized_layers should be normalized."""
        cfg = _make_kimi_quant_config(modules_to_not_quantize=["lm_head", "model.norm"])
        config = KimiW4AFp8Config.from_config(cfg)

        self.assertIn("lm_head", config.unquantized_layers)
        self.assertIn("model.lm_head", config.unquantized_layers)
        self.assertIn("norm", config.unquantized_layers)
        self.assertIn("model.norm", config.unquantized_layers)

    def test_unquantized_layers_auto_detect_lm_head(self):
        """When unquantized_layers is not specified but lm_head is in ignore list,
        it should be auto-detected as unquantized."""
        cfg = _make_kimi_quant_config(
            ignore=["model.lm_head", "model.layers.0.self_attn"]
        )
        config = KimiW4AFp8Config.from_config(cfg)

        # lm_head should be in unquantized_layers (auto-detected)
        lm_head_in_unquantized = any(
            "lm_head" in layer for layer in config.unquantized_layers
        )
        self.assertTrue(
            lm_head_in_unquantized,
            f"lm_head not auto-detected in unquantized_layers: {config.unquantized_layers}",
        )

        # self_attn should NOT be in unquantized_layers
        self_attn_in_unquantized = any(
            "self_attn" in layer for layer in config.unquantized_layers
        )
        self.assertFalse(
            self_attn_in_unquantized,
            f"self_attn should not be in unquantized_layers: {config.unquantized_layers}",
        )

    def test_custom_group_size(self):
        """Non-default group_size should be respected."""
        cfg = _make_kimi_quant_config(group_size=64)
        config = KimiW4AFp8Config.from_config(cfg)
        self.assertEqual(config.group_size, 64)


# ---------------------------------------------------------------------------
# Tests: KimiW4AFp8Config.get_quant_method()
# ---------------------------------------------------------------------------


class TestKimiW4AFp8ConfigGetQuantMethod(CustomTestCase):
    """Test the get_quant_method() routing logic."""

    def setUp(self):
        self.config = KimiW4AFp8Config(
            ignored_layers=["layers.0.self_attn", "model.layers.0.self_attn"],
            unquantized_layers=["lm_head", "model.lm_head"],
        )

    def test_linear_layer_returns_fp8(self):
        """A normal LinearBase layer should get Fp8LinearMethod."""
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        mock_layer = MagicMock(spec=LinearBase)
        method = self.config.get_quant_method(
            mock_layer, "model.layers.0.mlp.gate_proj"
        )
        self.assertIsInstance(method, Fp8LinearMethod)

    def test_unquantized_layer_returns_unquantized(self):
        """A layer in unquantized_layers (e.g. lm_head) should get UnquantizedLinearMethod."""
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        mock_layer = MagicMock(spec=LinearBase)
        method = self.config.get_quant_method(mock_layer, "lm_head")
        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_ignored_but_not_unquantized_gets_fp8(self):
        """A layer in ignored_layers but NOT in unquantized_layers should still get FP8.
        This is the key difference from base W4AFp8Config: 'ignore' = skip W4, use FP8.
        """
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        mock_layer = MagicMock(spec=LinearBase)
        # self_attn is in ignored_layers but not in unquantized_layers
        method = self.config.get_quant_method(
            mock_layer, "model.layers.0.self_attn.q_proj"
        )
        self.assertIsInstance(method, Fp8LinearMethod)

    def test_fused_moe_returns_w4afp8(self):
        """FusedMoE layers should get W4AFp8MoEMethod."""
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod

        mock_layer = MagicMock(spec=FusedMoE)
        method = self.config.get_quant_method(mock_layer, "model.layers.0.mlp.experts")
        self.assertIsInstance(method, W4AFp8MoEMethod)

    def test_other_layer_returns_none(self):
        """An unknown layer type should return None."""
        mock_layer = MagicMock(spec=torch.nn.LayerNorm)
        method = self.config.get_quant_method(mock_layer, "model.layers.0.norm")
        self.assertIsNone(method)


# ---------------------------------------------------------------------------
# Tests: KimiW4AFp8Config vs W4AFp8Config base class
# ---------------------------------------------------------------------------


class TestKimiW4AFp8ConfigVsBase(CustomTestCase):
    """Verify the Kimi subclass differs from the base in the expected ways."""

    def test_name_differs(self):
        self.assertEqual(W4AFp8Config.get_name(), "w4afp8")
        self.assertEqual(KimiW4AFp8Config.get_name(), "kimi_w4afp8")

    def test_kimi_has_unquantized_layers(self):
        """KimiW4AFp8Config should have unquantized_layers attribute."""
        config = KimiW4AFp8Config(unquantized_layers=["lm_head"])
        self.assertTrue(hasattr(config, "unquantized_layers"))
        self.assertIn("lm_head", config.unquantized_layers)

    def test_base_uses_ignored_for_unquant(self):
        """Base W4AFp8Config routes ignored layers to UnquantizedLinearMethod.
        This is the DeepSeek behavior (empty ignored_layers, so harmless)."""
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        base_config = W4AFp8Config(ignored_layers=["lm_head", "model.lm_head"])
        mock_layer = MagicMock(spec=LinearBase)
        method = base_config.get_quant_method(mock_layer, "lm_head")
        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_kimi_routes_ignored_to_fp8(self):
        """Kimi routes ignored layers (non-lm_head) to FP8, not unquantized."""
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        kimi_config = KimiW4AFp8Config(
            ignored_layers=["self_attn", "model.self_attn"],
            unquantized_layers=["lm_head", "model.lm_head"],
        )
        mock_layer = MagicMock(spec=LinearBase)
        method = kimi_config.get_quant_method(
            mock_layer, "model.layers.0.self_attn.q_proj"
        )
        self.assertIsInstance(method, Fp8LinearMethod)

    def test_shared_capabilities(self):
        """Both configs should share the same capabilities."""
        self.assertEqual(
            W4AFp8Config.get_supported_act_dtypes(),
            KimiW4AFp8Config.get_supported_act_dtypes(),
        )
        self.assertEqual(
            W4AFp8Config.get_min_capability(),
            KimiW4AFp8Config.get_min_capability(),
        )


# ---------------------------------------------------------------------------
# Tests: make_expert_input_scale_params_mapping()
# ---------------------------------------------------------------------------


class TestMakeExpertInputScaleParamsMapping(CustomTestCase):
    """Test the parameterized input_scale naming in FusedMoE."""

    def test_default_deepseek_naming(self):
        """Default (DeepSeek) uses w1/w2/w3."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        mapping = FusedMoE.make_expert_input_scale_params_mapping(num_experts=2)

        # Should have 6 entries (2 experts * 3 projections)
        self.assertEqual(len(mapping), 6)

        # Check that w1/w2/w3 appear in weight_name
        weight_names = [m[1] for m in mapping]
        self.assertTrue(any("w1." in n for n in weight_names))
        self.assertTrue(any("w2." in n for n in weight_names))
        self.assertTrue(any("w3." in n for n in weight_names))

    def test_kimi_hf_naming(self):
        """Kimi K2.5 uses gate_proj/down_proj/up_proj."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        mapping = FusedMoE.make_expert_input_scale_params_mapping(
            num_experts=2,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
        )

        self.assertEqual(len(mapping), 6)

        weight_names = [m[1] for m in mapping]
        self.assertTrue(any("gate_proj." in n for n in weight_names))
        self.assertTrue(any("down_proj." in n for n in weight_names))
        self.assertTrue(any("up_proj." in n for n in weight_names))

        # w1/w2/w3 should NOT appear
        self.assertFalse(any(".w1." in n for n in weight_names))
        self.assertFalse(any(".w2." in n for n in weight_names))
        self.assertFalse(any(".w3." in n for n in weight_names))

    def test_mapping_shard_ids(self):
        """Shard IDs should always be w1/w2/w3 regardless of checkpoint naming."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        mapping = FusedMoE.make_expert_input_scale_params_mapping(
            num_experts=1,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
        )

        shard_ids = [m[3] for m in mapping]
        self.assertEqual(sorted(shard_ids), ["w1", "w2", "w3"])

    def test_param_name_routing(self):
        """w1 and w3 (gate/up) map to w13_, while w2 (down) maps to w2_."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        mapping = FusedMoE.make_expert_input_scale_params_mapping(num_experts=1)

        for param_name, weight_name, expert_id, shard_id in mapping:
            if shard_id in ("w1", "w3"):
                self.assertIn("w13_", param_name, f"shard_id={shard_id}")
            else:
                self.assertIn("w2_", param_name, f"shard_id={shard_id}")


# ---------------------------------------------------------------------------
# Tests: Registration
# ---------------------------------------------------------------------------


class TestKimiW4AFp8Registration(CustomTestCase):
    """Verify KimiW4AFp8Config is registered in the quantization method registry."""

    def test_registered_in_base_methods(self):
        from sglang.srt.layers.quantization import BASE_QUANTIZATION_METHODS

        self.assertIn("kimi_w4afp8", BASE_QUANTIZATION_METHODS)
        self.assertIs(BASE_QUANTIZATION_METHODS["kimi_w4afp8"], KimiW4AFp8Config)

    def test_w4afp8_still_registered(self):
        """Base W4AFp8Config should also still be registered (no breakage)."""
        from sglang.srt.layers.quantization import BASE_QUANTIZATION_METHODS

        self.assertIn("w4afp8", BASE_QUANTIZATION_METHODS)
        self.assertIs(BASE_QUANTIZATION_METHODS["w4afp8"], W4AFp8Config)


if __name__ == "__main__":
    unittest.main()
