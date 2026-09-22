"""Unit tests for QuarkConfig and its MoE scheme — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import sys
import types
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.moe_runner.aiter import AiterQuantType
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.quark.quark import (
    QuarkConfig,
    _build_mixed_precision_layer_quant_config,
    _mixed_precision_layer_map,
    _parse_nvfp4_excludes,
)
from sglang.srt.layers.quantization.quark.schemes import (
    quark_w4a4_mxfp4_moe as quark_moe,
)
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    QuarkW4A4MXFp4MoE,
)
from sglang.srt.layers.quantization.quark.utils import check_equal_or_regex_match
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
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


class TestQuarkPerLayerBlockFp8(CustomTestCase):
    _BLOCK_FP8_CONFIG = {
        "weight": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_block",
            "block_size": [128, 128],
            "is_dynamic": False,
        },
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_group",
            "group_size": 128,
            "is_dynamic": True,
        },
        "output_tensors": None,
        "bias": None,
    }

    def _build_bare_config(self) -> QuarkConfig:
        config = _bare_config()
        config.quant_config = {
            "layer_quant_config": {
                "model.language_model.layers.0.mlp.down_proj": self._BLOCK_FP8_CONFIG
            },
            "layer_type_quant_config": {},
            "global_quant_config": {
                "weight": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "is_dynamic": False,
                    "scale_format": "e8m0",
                },
                "input_tensors": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "is_dynamic": True,
                    "scale_format": "e8m0",
                },
            },
        }
        config.exclude_layers = []
        config.kv_cache_group = []
        config.packed_modules_mapping = {}
        config.excluded_fp8_config = None
        config._online_quantized_layers = set()
        return config

    def test_model_mapper_rewrites_explicit_layer_config(self):
        config = self._build_bare_config()

        config.apply_weight_name_mapper(
            Glm5NextForConditionalGeneration.hf_to_sglang_mapper
        )

        self.assertIn(
            "model.layers.0.mlp.down_proj",
            config.quant_config["layer_quant_config"],
        )

    def test_model_mapper_rewrites_fused_visual_exclusion(self):
        config = self._build_bare_config()
        config.exclude_layers = ["model.visual.blocks.0.attn.qkv"]

        config.apply_weight_name_mapper(
            Glm5NextForConditionalGeneration.hf_to_sglang_mapper
        )

        self.assertEqual(
            config.exclude_layers,
            ["visual.blocks.0.attn.qkv_proj"],
        )
        self.assertNotIn(
            "model.language_model.layers.0.mlp.down_proj",
            config.quant_config["layer_quant_config"],
        )

    def test_explicit_block_fp8_linear_uses_fp8_method(self):
        config = self._build_bare_config()
        config.apply_weight_name_mapper(
            Glm5NextForConditionalGeneration.hf_to_sglang_mapper
        )
        layer = LinearBase.__new__(LinearBase)

        method = config.get_quant_method(layer, "model.layers.0.mlp.down_proj")

        self.assertIsInstance(method, Fp8LinearMethod)
        self.assertTrue(method.quant_config.is_checkpoint_fp8_serialized)
        self.assertEqual(method.quant_config.weight_block_size, [128, 128])

    def test_dynamic_block_fp8_weight_is_not_treated_as_serialized(self):
        layer_config = deepcopy(self._BLOCK_FP8_CONFIG)
        layer_config["weight"]["is_dynamic"] = True

        self.assertIsNone(QuarkConfig._get_block_fp8_config(layer_config, {}))

    def test_unmatched_layer_still_uses_global_quark_config(self):
        config = self._build_bare_config()
        config.apply_weight_name_mapper(
            Glm5NextForConditionalGeneration.hf_to_sglang_mapper
        )

        matched = config._find_matched_config(
            "model.layers.4.mlp.down_proj", torch.nn.Module()
        )

        self.assertEqual(matched["weight"]["dtype"], "fp4")


class _Runner:
    """Records the quant_info apply_weights() hands to the runner."""

    def __init__(self):
        self.quant_info = None

    def run(self, dispatch_output, quant_info):
        self.quant_info = quant_info
        return dispatch_output


class TestQuarkMxfp4MoEAiterQuantInfo(CustomTestCase):
    """apply_weights assembles what the AITER runner consumes.

    The gfx950 e2e builds AiterMoeQuantInfo by hand, so dropping the gate/up
    layout, the clamp or the padding here would leave it passing while served
    experts read the gate and up halves swapped.
    """

    def test_apply_forwards_clamp_separated_layout_and_padding(self):
        scheme = object.__new__(QuarkW4A4MXFp4MoE)
        scheme.moe_runner_config = SimpleNamespace(swiglu_limit=10.0)
        scheme.runner = _Runner()

        layer = SimpleNamespace(
            w13_weight=torch.zeros((1, 4, 2), dtype=torch.uint8),
            w2_weight=torch.zeros((1, 2, 2), dtype=torch.uint8),
            w13_weight_scale=torch.ones((1, 4, 1), dtype=torch.uint8),
            w2_weight_scale=torch.ones((1, 2, 1), dtype=torch.uint8),
            hidden_pad=0,
            intermediate_pad=128,
            dispatcher=SimpleNamespace(expert_mask_gpu=torch.tensor([True, False])),
        )
        layer.w13_weight.is_shuffled = True
        fake_moe_common = types.ModuleType("aiter.ops.flydsl.moe_common")
        fake_moe_common.GateMode = SimpleNamespace(
            SEPARATED=SimpleNamespace(value="separated"),
            INTERLEAVE=SimpleNamespace(value="interleave"),
        )

        with (
            patch.dict(sys.modules, {"aiter.ops.flydsl.moe_common": fake_moe_common}),
            patch.object(quark_moe, "_is_gfx95", True),
            patch.object(quark_moe, "_is_gfx1250", False),
        ):
            marker = object()
            result = scheme.apply_weights(layer, marker)

        self.assertIs(result, marker)
        quant_info = scheme.runner.quant_info
        self.assertEqual(quant_info.quant_type, AiterQuantType.PER_1X32)
        self.assertEqual(quant_info.swiglu_limit, 10.0)
        self.assertEqual(quant_info.hidden_pad, 0)
        self.assertEqual(quant_info.intermediate_pad, 128)
        self.assertEqual(quant_info.fused_moe_kwargs, {"gate_mode": "separated"})
        self.assertIs(quant_info.expert_mask, layer.dispatcher.expert_mask_gpu)
        self.assertTrue(quant_info.w13_weight.is_shuffled)
        self.assertTrue(quant_info.w2_weight.is_shuffled)


if __name__ == "__main__":
    unittest.main()
