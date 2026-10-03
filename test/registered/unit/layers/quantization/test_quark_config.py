"""Unit tests for QuarkConfig and its MoE scheme — CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=18, suite="base-a-test-cpu")

import sys
import types
import unittest
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.moe_runner.aiter import AiterQuantType
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.quark import quark as quark_config_mod
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


class TestFusedExpertConfig(CustomTestCase):
    """Per-expert-only entries must not leave a FusedMoE on the global scheme."""

    _FP8 = {
        "weight": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_block",
            "block_size": [128, 128],
        },
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_group",
            "group_size": 128,
        },
    }
    _MXFP4 = {
        "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32},
        "input_tensors": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32},
    }
    _PREFIX = "model.decoder.mlp.experts"

    def _entries(self, num_experts):
        return {
            f"{index}.{projection}": deepcopy(self._FP8)
            for index in range(num_experts)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }

    def _config_with(self, layer_quant_config):
        quark_config = _bare_config()
        quark_config.quant_config = {
            "layer_quant_config": layer_quant_config,
            "layer_type_quant_config": {},
            "global_quant_config": self._MXFP4,
        }
        quark_config.packed_modules_mapping = {}
        return quark_config

    def _lookup(self, entries):
        quark_config = self._config_with(
            {f"{self._PREFIX}.{s}": cfg for s, cfg in entries.items()}
        )
        return quark_config._find_matched_config(self._PREFIX, torch.nn.Module())

    def test_fused_moe_resolves_from_per_expert_entries(self):
        matched = self._lookup(self._entries(288))
        self.assertEqual(matched["weight"]["dtype"], "fp8_e4m3")

    def test_incomplete_or_mixed_expert_coverage_raises(self):
        missing = {s: c for s, c in self._entries(4).items() if not s.startswith("2.")}
        conflicting = self._entries(4)
        conflicting["2.up_proj"] = deepcopy(self._MXFP4)
        partial_projections = self._entries(4)
        del partial_projections["3.down_proj"]
        cases = {
            "missing_expert": (missing, "skip experts"),
            "conflicting_scheme": (conflicting, "different quantization"),
            "partial_projections": (partial_projections, "different projections"),
            "fused_param_name": ({"w13_weight_scale": self._FP8}, "expert index"),
        }
        for name, (entries, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                self._lookup(entries)

    def test_experts_without_per_expert_entries_fall_through_to_global(self):
        self.assertEqual(self._lookup({})["weight"]["dtype"], "fp4")


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


_MXFP4_MOE_SPEC = {
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
}

_FP8_MOE_SPEC = {
    "weight": {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
    "input_tensors": {
        "dtype": "fp8_e4m3",
        "qscheme": "per_tensor",
        "is_dynamic": False,
    },
}

# Unlike _FP8_MOE_SPEC, this one satisfies _get_block_fp8_config, so a layer
# carrying it is dispatched to Fp8MoEMethod without ever reaching the MoE
# scheme lookup.
_BLOCK_FP8_MOE_SPEC = {
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


class TestSharedExpertOnlineMxfp4Gate(CustomTestCase):
    """Load-time MXFP4 quantization of a BF16 shared expert into the fused slot.

    Quark MXFP4 checkpoints (Qwen3.5, Qwen3.8) quantize the routed experts but
    list the shared-expert body in `exclude`, so it ships in BF16. Fusion appends
    the shared expert as one more routed expert, which only works when every
    slot of the packed FP4 buffer holds the same format. The gate therefore has
    to answer three questions without conflating them: is the target model's
    shared-expert body really excluded, can this build quantize it while
    loading, and does every MoE layer resolve to the one scheme that knows how.

    Getting any of these wrong is silent: the BF16 body gets copied into packed
    FP4 buffers unconverted and the model serves garbage rather than failing.
    """

    _NUM_HIDDEN_LAYERS = 4
    _NUM_NEXTN = 1

    def _config(
        self,
        exclude_layers=(),
        global_spec=None,
        layer_quant_config=None,
        is_prequantized=True,
        model_type="qwen3_5_moe_text",
    ) -> QuarkConfig:
        config = _bare_config()
        config.quant_config = {
            "global_quant_config": deepcopy(global_spec or _MXFP4_MOE_SPEC),
            "layer_quant_config": deepcopy(layer_quant_config or {}),
            "layer_type_quant_config": {},
        }
        config.exclude_layers = list(exclude_layers)
        config.packed_modules_mapping = {}
        config.is_prequantized = is_prequantized
        config.dequantization_config = None
        config.num_hidden_layers = self._NUM_HIDDEN_LAYERS
        config.num_nextn_predict_layers = self._NUM_NEXTN
        # What amd/Qwen3.8-2.4T-A95B-Quark-MXFP4 reports; the conversion is
        # offered on the Qwen3.5 MoE architectures only.
        config.model_type = model_type
        # Normally built in __init__, which _bare_config skips; get_quant_method
        # records every layer it quantizes in it.
        config._online_quantized_layers = set()
        return config

    @staticmethod
    def _moe_layer(has_fused_shared=True):
        """A FusedMoE stand-in that answers the two things the gate reads.

        `_has_fused_shared` is an instance attribute, so a class spec does not
        carry it and it has to be set explicitly.
        """
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        layer = MagicMock(spec=FusedMoE)
        layer._has_fused_shared = has_fused_shared
        return layer

    @staticmethod
    @contextmanager
    def _online_quant_available(use_aiter=True, gfx95=True, flag=True, rocm=True):
        """Pretend this is an aiter + gfx95 build with the opt-in flag set."""
        with (
            envs.SGLANG_USE_AITER.override(use_aiter),
            patch.object(quark_config_mod, "is_hip", return_value=rocm),
            patch.object(quark_config_mod, "is_gfx95_supported", return_value=gfx95),
            envs.SGLANG_FUSE_SHARED_EXPERTS_ONLINE_MXFP4.override(flag),
        ):
            yield

    # ---- which excludes mean "the shared expert body is BF16" -------------

    def test_excluded_shared_expert_body_is_detected(self):
        for exclude in (
            "model.layers.0.mlp.shared_expert.gate_proj",
            "re:.*shared_expert.*",
            "model.language_model.layers.2.mlp.shared_expert.down_proj",
        ):
            with self.subTest(exclude):
                config = self._config(exclude_layers=[exclude])
                self.assertTrue(config.shared_expert_excluded_from_quant())

    def test_shared_expert_gate_alone_does_not_count(self):
        # `"shared_expert" in layer` also matches `shared_expert_gate`, a tiny
        # router that is BF16 in every checkpoint and is not part of the fused
        # GEMM. Counting it would enable online quantization for checkpoints
        # whose shared-expert body is already MXFP4.
        config = self._config(exclude_layers=["model.layers.0.mlp.shared_expert_gate"])
        self.assertFalse(config.shared_expert_excluded_from_quant())

    def test_draft_stack_excludes_do_not_count(self):
        # An MTP/NextN draft layer is excluded in most checkpoints and says
        # nothing about how the target model stores its shared experts.
        appended = self._NUM_HIDDEN_LAYERS  # first draft layer index
        for exclude in (
            "mtp.0.mlp.shared_expert.gate_proj",
            f"model.layers.{appended}.mlp.shared_expert.down_proj",
        ):
            with self.subTest(exclude):
                config = self._config(exclude_layers=[exclude])
                self.assertFalse(config.shared_expert_excluded_from_quant())

    def test_target_body_still_counts_when_draft_is_also_excluded(self):
        config = self._config(
            exclude_layers=[
                "mtp.0.mlp.shared_expert.gate_proj",
                f"model.layers.{self._NUM_HIDDEN_LAYERS}.mlp.shared_expert.down_proj",
                "model.layers.0.mlp.shared_expert.gate_proj",
            ]
        )
        self.assertTrue(config.shared_expert_excluded_from_quant())

    def test_draft_layer_detection_without_hf_layer_count(self):
        # num_hidden_layers is unknown for some configs; the appended-layer
        # spelling cannot be recognised then, but "mtp." still must be.
        config = self._config(exclude_layers=["mtp.0.mlp.shared_expert.gate_proj"])
        config.num_hidden_layers = None
        self.assertFalse(config.shared_expert_excluded_from_quant())

    # ---- when the conversion is actually available -------------------------

    def test_every_precondition_is_required(self):
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"]
        )
        with self._online_quant_available():
            self.assertTrue(config.shared_expert_online_mxfp4_supported())

        cases = {
            "no_aiter": {"use_aiter": False},
            "not_gfx95": {"gfx95": False},
            "flag_unset": {"flag": False},
            "not_rocm": {"rocm": False},
        }
        for name, kwargs in cases.items():
            with self.subTest(name), self._online_quant_available(**kwargs):
                self.assertFalse(config.shared_expert_online_mxfp4_supported())

        with self.subTest("not_prequantized"), self._online_quant_available():
            not_prequantized = self._config(
                exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"],
                is_prequantized=False,
            )
            self.assertFalse(not_prequantized.shared_expert_online_mxfp4_supported())

    def test_a_cuda_build_is_not_blamed_on_aiter(self):
        # is_hip() is checked before SGLANG_USE_AITER: on a CUDA build the aiter
        # flag is meaningless, so naming it would send the user after the wrong
        # fix. Both are unset here, and the ROCm reason has to win.
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"]
        )
        with (
            self._online_quant_available(rocm=False, use_aiter=False),
            patch.object(quark_config_mod.logger, "warning_once") as warn,
        ):
            self.assertFalse(config.shared_expert_online_mxfp4_supported())
        self.assertIn("not a ROCm build", warn.call_args[0][0])

    def test_only_the_qwen_moe_architectures_are_supported(self):
        # can_fuse_shared_expert() is also reached from DeepSeek's
        # quant_blocks_shared_experts_fusion(), and "shared_expert" in layer
        # matches DeepSeek's mlp.shared_experts.* too. Without the model_type
        # gate this flag would silently enable fusion for a model family it was
        # never measured on.
        excluded = ["model.layers.0.mlp.shared_expert.gate_proj"]
        for model_type in ("qwen3_5_moe", "qwen3_5_moe_text"):
            with self.subTest(model_type), self._online_quant_available():
                config = self._config(exclude_layers=excluded, model_type=model_type)
                self.assertTrue(config.shared_expert_online_mxfp4_supported())

        for model_type in ("deepseek_v3", "deepseek_v32", "qwen3_moe", None):
            with self.subTest(model_type), self._online_quant_available():
                config = self._config(exclude_layers=excluded, model_type=model_type)
                self.assertFalse(config.shared_expert_online_mxfp4_supported())
                # And the pre-existing refusal is what such a checkpoint gets.
                self.assertFalse(config.can_fuse_shared_expert())

    def test_non_mxfp4_moe_layer_override_falls_back_instead_of_raising(self):
        # A per-layer entry sending some later MoE layer elsewhere is visible up
        # front, so it is reported here as a warning and fallback rather than as
        # a NotImplementedError part way through loading.
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"],
            layer_quant_config={"model.layers.3.mlp.experts": _FP8_MOE_SPEC},
        )
        with self._online_quant_available():
            self.assertFalse(config.shared_expert_online_mxfp4_supported())
            self.assertFalse(config.shared_expert_needs_online_mxfp4())
            # The layer that is routed elsewhere no longer brings startup down.
            self.assertIsNotNone(
                config.get_quant_method(self._moe_layer(), "model.layers.3.mlp.experts")
            )

    def test_an_mxfp4_moe_layer_override_is_not_mistaken_for_a_refusal(self):
        # The scan must only reject entries that are not MXFP4, and must ignore
        # the shared expert's own entry, which is the tensor being converted.
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"],
            layer_quant_config={
                "model.layers.3.mlp.experts": _MXFP4_MOE_SPEC,
                "model.layers.3.mlp.shared_experts.gate_proj": _FP8_MOE_SPEC,
            },
        )
        with self._online_quant_available():
            self.assertTrue(config.shared_expert_online_mxfp4_supported())

    def test_non_mxfp4_global_spec_is_not_supported(self):
        # Only QuarkW4A4MXFp4MoE can do the conversion, and unmatched names
        # fall back to the global spec, so a non-MXFP4 global spec must not
        # claim support.
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"],
            global_spec=_FP8_MOE_SPEC,
        )
        with self._online_quant_available():
            self.assertFalse(config.shared_expert_online_mxfp4_supported())

    def test_per_layer_config_sending_moe_elsewhere_is_not_supported(self):
        # layer_quant_config can route an individual MoE layer to the W4A8 or
        # FP8 scheme even when the global spec is MXFP4; those would copy the
        # BF16 shared expert into their own buffers unconverted.
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"],
            layer_quant_config={"model.layers.0.mlp.experts": _FP8_MOE_SPEC},
        )
        with self._online_quant_available():
            self.assertFalse(config.shared_expert_online_mxfp4_supported())

    # ---- how the two combine into the fusion decision ----------------------

    def test_needs_online_requires_both_halves(self):
        excluded = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"]
        )
        already_quantized = self._config(exclude_layers=[])
        with self._online_quant_available():
            self.assertTrue(excluded.shared_expert_needs_online_mxfp4())
            self.assertFalse(already_quantized.shared_expert_needs_online_mxfp4())
        with self._online_quant_available(flag=False):
            self.assertFalse(excluded.shared_expert_needs_online_mxfp4())

    def test_fusion_follows_online_support_when_body_is_excluded(self):
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"]
        )
        with self._online_quant_available():
            self.assertTrue(config.can_fuse_shared_expert())
        # Without the opt-in the checkpoint must fall back to a standalone
        # shared expert rather than fuse a BF16 body into FP4 buffers.
        with self._online_quant_available(flag=False):
            self.assertFalse(config.can_fuse_shared_expert())

    def test_fusion_unchanged_for_uniformly_quantized_checkpoint(self):
        # No shared-expert exclude and no per-layer config -> uniform spec, and
        # the pre-existing "nothing to compare" answer must still be reached
        # even with the flag on.
        config = self._config(exclude_layers=[])
        with self._online_quant_available():
            self.assertTrue(config.can_fuse_shared_expert())

    # ---- the scheme must be told, and must refuse when it cannot -----------

    def test_moe_scheme_is_told_to_quantize_the_shared_slot(self):
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"]
        )
        with self._online_quant_available():
            scheme = config.get_moe_scheme(
                self._moe_layer(), "model.layers.0.mlp.experts"
            )
        self.assertIsInstance(scheme, QuarkW4A4MXFp4MoE)
        self.assertTrue(scheme.quantize_shared_expert_online)

    def test_moe_scheme_not_told_when_shared_expert_is_quantized(self):
        config = self._config(exclude_layers=[])
        with self._online_quant_available():
            scheme = config.get_moe_scheme(
                self._moe_layer(), "model.layers.0.mlp.experts"
            )
        self.assertFalse(scheme.quantize_shared_expert_online)

    def test_moe_scheme_not_told_when_this_layer_has_no_fused_slot(self):
        # The model-wide answer says "convert", but fusion can still be off for
        # an individual layer: --disable-shared-experts-fusion, a DeepEP / MoRI
        # a2a backend, a mismatched shared-expert intermediate size, or per-rank
        # shared slots at moe_ep_size > 1. Nothing is copied into a fused slot
        # then, so the scheme must not claim to be converting one -- that log
        # line is what someone greps when chasing an accuracy difference.
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"]
        )
        with self._online_quant_available():
            self.assertTrue(config.shared_expert_needs_online_mxfp4())
            scheme = config.get_moe_scheme(
                self._moe_layer(has_fused_shared=False),
                "model.layers.0.mlp.experts",
            )
        self.assertFalse(scheme.quantize_shared_expert_online)

    def test_moe_layer_outside_the_mxfp4_scheme_raises(self):
        # can_fuse_shared_expert() answers once for the whole model off layer 0,
        # but the fused slot lives in every MoE layer. A later layer routed to a
        # different scheme must fail loudly, not serve corrupted weights.
        # Checked in get_quant_method rather than get_moe_scheme: block-FP8 MoE
        # layers are dispatched to Fp8MoEMethod and never reach the latter.
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"],
            layer_quant_config={"model.layers.3.mlp.experts": _FP8_MOE_SPEC},
        )
        # The up-front scan in shared_expert_online_mxfp4_supported() already
        # rejects this config, so reach past it to pin the backstop itself.
        with (
            self._online_quant_available(),
            patch.object(config, "shared_expert_needs_online_mxfp4", return_value=True),
        ):
            with self.assertRaisesRegex(NotImplementedError, "W4A4 MXFP4 MoE scheme"):
                config.get_quant_method(self._moe_layer(), "model.layers.3.mlp.experts")

    def test_a_layer_without_a_fused_slot_does_not_raise(self):
        # Same mixed config, but this layer never gets a shared slot, so no
        # BF16 tensor can reach its buffers and a hard startup failure would be
        # wrong. It loads through its own scheme instead.
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"],
            layer_quant_config={"model.layers.3.mlp.experts": _FP8_MOE_SPEC},
        )
        with (
            self._online_quant_available(),
            patch.object(config, "shared_expert_needs_online_mxfp4", return_value=True),
        ):
            self.assertIsNotNone(
                config.get_quant_method(
                    self._moe_layer(has_fused_shared=False),
                    "model.layers.3.mlp.experts",
                )
            )

    def test_block_fp8_moe_layer_raises_before_the_fp8_dispatch(self):
        # The per-tensor case above would also be caught by a check inside
        # get_moe_scheme. A block-FP8 layer would not: get_quant_method returns
        # Fp8MoEMethod for it first, so the scheme lookup never happens and the
        # BF16 shared expert would land in an FP8 buffer unconverted. The first
        # assertion pins that this spec really does take that branch, so the
        # test fails if the guard ever moves back into get_moe_scheme.
        self.assertIsNotNone(QuarkConfig._get_block_fp8_config(_BLOCK_FP8_MOE_SPEC, {}))
        config = self._config(
            exclude_layers=["model.layers.0.mlp.shared_expert.gate_proj"],
            layer_quant_config={"model.layers.3.mlp.experts": _BLOCK_FP8_MOE_SPEC},
        )
        with (
            self._online_quant_available(),
            patch.object(config, "shared_expert_needs_online_mxfp4", return_value=True),
        ):
            with self.assertRaisesRegex(NotImplementedError, "W4A4 MXFP4 MoE scheme"):
                config.get_quant_method(self._moe_layer(), "model.layers.3.mlp.experts")


if __name__ == "__main__":
    unittest.main()
