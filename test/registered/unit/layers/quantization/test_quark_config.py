"""Unit tests for QuarkConfig — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.quark.quark import (
    QuarkConfig,
    _build_mixed_precision_layer_quant_config,
    _mixed_precision_layer_map,
    _parse_nvfp4_excludes,
)
from sglang.srt.layers.quantization.quark.utils import check_equal_or_regex_match
from sglang.test.test_utils import CustomTestCase

_GET_CAP = "sglang.srt.layers.quantization.quark.quark.get_device_capability"


def _bare_config() -> QuarkConfig:
    """Skip __init__ — _check_scheme_supported reads no instance attributes."""
    return QuarkConfig.__new__(QuarkConfig)


class TestCheckSchemeSupportedError(CustomTestCase):
    """Regression for `RuntimeError("a", "b", "c")` being passed three args.

    Bug: `_check_scheme_supported` raised `RuntimeError` with three positional
    string fragments. `RuntimeError.__str__` formats `self.args` as a tuple
    when `len(args) != 1`, so the user saw
        ('Quantization scheme is not supported for ', 'the current GPU…', 'Current capability: 70.')
    instead of a sentence. Fix: pass one already-joined message.
    """

    def test_error_is_single_argument(self):
        # The structural assertion that catches the bug regardless of wording.
        with patch(_GET_CAP, return_value=(7, 0)):  # capability = 70 < 200
            with self.assertRaises(RuntimeError) as ctx:
                _bare_config()._check_scheme_supported(min_capability=200)
        err = ctx.exception
        self.assertEqual(
            len(err.args),
            1,
            f"RuntimeError must carry a single joined message, got {err.args!r}",
        )

    def test_error_message_renders_as_sentence(self):
        with patch(_GET_CAP, return_value=(7, 0)):
            with self.assertRaises(RuntimeError) as ctx:
                _bare_config()._check_scheme_supported(min_capability=200)
        msg = str(ctx.exception)
        # Tuple-repr leakage shows up as a leading '(' and quote-comma joins.
        self.assertFalse(
            msg.startswith("("),
            f"error message starts with '(' (tuple repr leaked): {msg!r}",
        )
        self.assertNotIn(
            "', '",
            msg,
            f"error message contains tuple-style fragment join: {msg!r}",
        )

    def test_error_message_content(self):
        with patch(_GET_CAP, return_value=(7, 0)):
            with self.assertRaises(RuntimeError) as ctx:
                _bare_config()._check_scheme_supported(min_capability=200)
        msg = str(ctx.exception)
        self.assertIn("Quantization scheme is not supported", msg)
        self.assertIn("Min capability: 200", msg)
        self.assertIn("Current capability: 70", msg)

    # ---- Guardrails: unchanged code paths ---------------------------------

    def test_unsupported_returns_false_when_error_disabled(self):
        with patch(_GET_CAP, return_value=(7, 0)):
            ok = _bare_config()._check_scheme_supported(min_capability=200, error=False)
        self.assertFalse(ok)

    def test_supported_returns_true(self):
        with patch(_GET_CAP, return_value=(8, 0)):  # capability = 80 >= 70
            ok = _bare_config()._check_scheme_supported(min_capability=70)
        self.assertTrue(ok)

    def test_no_device_returns_false(self):
        with patch(_GET_CAP, return_value=None):
            ok = _bare_config()._check_scheme_supported(min_capability=70)
        self.assertFalse(ok)


class TestMixedPrecisionLayerConfig(CustomTestCase):
    """NVFP4-only-experts + FP8-elsewhere online requant (quark_mxfp4).

    A MIXED_PRECISION NVFP4 checkpoint (e.g. nvidia/Qwen3.5-397B-A17B-NVFP4-V2)
    keeps some layers in NVFP4 while others in FP8. Online requant must send
    only the NVFP4 layers through the dequant->MXFP4 path and load the FP8 layers
    as FP8.
    """

    _LAYER_MAP_SRC = {
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.language_model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
            "model.language_model.layers.0.self_attn.k_proj": {"quant_algo": "FP8"},
            "model.language_model.layers.0.self_attn.v_proj": {"quant_algo": "FP8"},
            "model.language_model.layers.0.self_attn.o_proj": {"quant_algo": "FP8"},
            "model.language_model.layers.0.mlp.shared_expert.gate_proj": {
                "quant_algo": "FP8"
            },
            "model.language_model.layers.0.mlp.shared_expert.down_proj": {
                "quant_algo": "FP8"
            },
            "model.language_model.layers.0.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
            "model.language_model.layers.1.mlp.experts": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
            "model.language_model.layers.1.self_attn.q_proj": {"quant_algo": "FP8"},
        },
    }

    def _build_bare_config(self) -> QuarkConfig:
        layer_map = _mixed_precision_layer_map(self._LAYER_MAP_SRC)
        layer_quant_config, has_nvfp4 = _build_mixed_precision_layer_quant_config(
            layer_map
        )
        self.assertTrue(has_nvfp4)
        synth_config = QuarkConfig._create_online_mxfp4_config(
            model_type="qwen3_5_moe",
            layer_quant_config=layer_quant_config,
        )
        synth_config["packed_modules_mapping"] = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        }
        quark_config = _bare_config()
        quark_config.quant_config = synth_config
        quark_config.packed_modules_mapping = synth_config["packed_modules_mapping"]
        quark_config.exclude_layers = synth_config["exclude"]
        return quark_config

    def test_experts_route_to_mxfp4_requant(self):
        # fnmatch keys (not `re:`) must match the sglang module path so experts
        # hit the fp4 target, not fall through to the global config
        quark_config = self._build_bare_config()
        matched = quark_config._find_matched_config(
            "model.layers.0.mlp.experts", torch.nn.Module()
        )
        self.assertEqual(matched["weight"]["dtype"], "fp4")
        self.assertEqual(matched["weight"]["group_size"], 32)

    def test_fp8_layers_not_requantized(self):
        quark_config = self._build_bare_config()
        for name in (
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.shared_expert.gate_proj",
            "model.layers.0.mlp.shared_expert.down_proj",
        ):
            matched = quark_config._find_matched_config(name, torch.nn.Module())
            self.assertEqual(matched["weight"]["dtype"], "fp8_e4m3", msg=name)
            self.assertEqual(matched["weight"]["qscheme"], "per_tensor", msg=name)

    def test_fused_qkv_shards_share_fp8_scheme(self):
        # _find_matched_config expands qkv_proj -> q/k/v shards and requires a
        # consistent scheme; all three are FP8 so this must resolve
        quark_config = self._build_bare_config()
        matched = quark_config._find_matched_config(
            "model.layers.0.self_attn.qkv_proj", torch.nn.Module()
        )
        self.assertEqual(matched["weight"]["dtype"], "fp8_e4m3")

    def test_shared_expert_fusion_disabled_on_precision_mismatch(self):
        quark_config = self._build_bare_config()
        self.assertFalse(quark_config.can_fuse_shared_expert())

    def test_mixed_precision_skips_model_type_default_excludes(self):
        quark_config = self._build_bare_config()
        self.assertNotIn("re:.*shared_expert", quark_config.exclude_layers)
        self.assertNotIn("re:.*o_proj", quark_config.exclude_layers)

    def test_non_mixed_config_returns_none(self):
        self.assertIsNone(_mixed_precision_layer_map({"quant_algo": "NVFP4"}))


class TestParseNvfp4Excludes(CustomTestCase):
    """ModelOpt `ignore` lists mix `re:`-prefixed regexes with fnmatch globs."""

    def test_already_regex_entries_pass_through_and_match(self):
        # wrapping an already-`re:` entry with another `re:` +
        # fnmatch.translate produced `re:(?s:re:\\..*...)` which never matches,
        excludes = _parse_nvfp4_excludes(
            {"ignore": [r"re:.*linear_attn\.in_proj_a$", "mtp*"]}
        )
        self.assertTrue(
            check_equal_or_regex_match("model.layers.0.linear_attn.in_proj_a", excludes)
        )
        # fnmatch glob still translated and matches.
        self.assertTrue(check_equal_or_regex_match("mtp.layers.0.foo", excludes))
        # A quantized layer stays un-excluded.
        self.assertFalse(
            check_equal_or_regex_match("model.layers.0.mlp.experts", excludes)
        )


if __name__ == "__main__":
    unittest.main()
