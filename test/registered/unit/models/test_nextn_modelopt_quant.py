"""
Unit tests for NextN ModelOpt FP4 quantization resolution and per-layer exclusions.

Tests cover:
  1. DeepseekV3ForCausalLMNextN._resolve_nextn_quant_config properly resolves
     ModelOpt FP4 configs, remapping checkpoint MTP layer prefixes (e.g. model.layers.61
     or dynamic model.layers.<num_hidden_layers>) to runtime decoder prefixes without junk mappings.
  2. Unquantized/excluded MTP behavior is preserved when exclude_modules targets MTP,
     while quantized MTP is preserved when not excluded.
  3. Shared quant configs are not mutated unexpectedly.
  4. BailingMoeForCausalLMNextN resolves ModelOpt FP4 quant configs and handles MTP exclusions.
  5. DeepseekModelNextN and BailingMoEModelNextN retain ModelOpt FP4 quant_config
     without blanket overriding to None.
  6. Glm4MoeModelNextN, Glm4MoeLiteModelNextN, and GlmOcrModelNextN preserve legacy BF16-MTP
     by overriding modelopt_fp4 to None.
"""

import importlib.abc
import importlib.machinery
import sys
from unittest.mock import MagicMock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

for _pkg in ("deep_ep", "deep_gemm"):
    try:
        __import__(_pkg)
    except (ImportError, OSError, AssertionError):
        sys.modules.pop(_pkg, None)

        class _GpuPkgLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                module.__getattr__ = lambda name: MagicMock()

        class _GpuPkgFinder(importlib.abc.MetaPathFinder):
            def __init__(self, pkg_name):
                self.pkg_name = pkg_name

            def find_spec(self, fullname, path, target=None):
                if fullname == self.pkg_name or fullname.startswith(
                    f"{self.pkg_name}."
                ):
                    return importlib.machinery.ModuleSpec(
                        fullname,
                        _GpuPkgLoader(),
                        is_package=True,
                    )
                return None

        sys.meta_path.insert(0, _GpuPkgFinder(_pkg))
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.bailing_moe_nextn import (
    BailingMoeForCausalLMNextN,
    BailingMoEModelNextN,
)
from sglang.srt.models.deepseek_nextn import (
    DeepseekModelNextN,
    DeepseekV3ForCausalLMNextN,
)
from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN
from sglang.srt.models.glm4_moe_lite_nextn import Glm4MoeLiteModelNextN
from sglang.srt.models.glm4_moe_nextn import Glm4MoeModelNextN
from sglang.srt.models.glm_ocr_nextn import GlmOcrModelNextN
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _mock_linear():
    return object.__new__(ReplicatedLinear)


def _mock_fused_moe():
    return object.__new__(FusedMoE)


class _FakeQuarkConfig(QuantizationConfig):
    def __init__(self, exclude_layers=None, layer_quant_config=None):
        super().__init__()
        self.exclude_layers = exclude_layers or []
        self.quant_config = {}
        if layer_quant_config:
            self.quant_config["layer_quant_config"] = layer_quant_config

    def get_name(self) -> str:
        return "quark"

    def get_quant_method(self, layer, prefix=""):
        return None

    def get_min_capability(self) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_scaled_act_names(self):
        return []

    def get_supported_act_dtypes(self):
        return []


class TestDeepseekNextNModelOptQuantResolution(CustomTestCase):
    def _make_model(self):
        return object.__new__(DeepseekV3ForCausalLMNextN)

    def test_none_quant_config_returns_none(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        res = model._resolve_nextn_quant_config(config, None)
        self.assertIsNone(res)

    def test_modelopt_fp4_not_excluded_returns_config(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertEqual(res.get_name(), "modelopt_fp4")
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertFalse(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("lm_head"))

    def test_modelopt_fp4_layer61_excluded_remaps_to_decoder(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61.*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.kv_b_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("lm_head"))

    def test_modelopt_fp4_mtp_moe_only_excluded(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61.mlp.experts.*"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))

    def test_modelopt_fp4_dynamic_layer_count_excluded(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=28)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.28.*"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))

    def test_modelopt_fp4_no_broad_prefix_matching(self):
        model = self._make_model()
        # Model has 6 hidden layers (MTP layer 6). Exclude module is for layer 60.
        config = SimpleNamespace(num_hidden_layers=6)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.60.*"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        # Layer 6 MTP should not be excluded by layer 60 exclude rule.
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertFalse(res.is_layer_excluded("model.decoder.mlp.experts"))

    def test_modelopt_fp4_nextn_specific_weights_mapping(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[
                "model.layers.61.eh_proj",
                "model.layers.61.enorm",
                "model.layers.61.hnorm",
                "model.layers.61.shared_head.norm",
            ],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertIn("model.eh_proj", res.exclude_modules)
        self.assertIn("model.enorm", res.exclude_modules)
        self.assertIn("model.hnorm", res.exclude_modules)
        self.assertIn("model.shared_head.norm", res.exclude_modules)
        self.assertNotIn("model.decoder.eh_proj", res.exclude_modules)

    def test_modelopt_fp4_short_prefix_mapping(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["layers.61.*"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))

    def test_modelopt_fp4_no_junk_prefix_mapping(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61.self_attn.q_a_proj"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertIn("model.decoder.self_attn.q_a_proj", res.exclude_modules)
        for name in res.exclude_modules:
            self.assertFalse(
                name.startswith("model.model."),
                f"Found junk prefix mapping: {name}",
            )

    def test_shared_quant_config_not_mutated(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        orig_exclude = ["model.layers.61.self_attn.q_a_proj"]
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=list(orig_exclude),
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNot(res, quant_cfg)
        self.assertEqual(quant_cfg.exclude_modules, orig_exclude)

    def test_loader_mapper_before_resolver_whole_layer_excluded(self):
        """Simulate model_loader/loader.py applying hf_to_sglang_mapper before model init.

        When exclude_modules=["model.layers.61.*", "lm_head"], the loader's
        apply_weight_name_mapper transforms "model.layers.61.*" to "model.decoder.*".
        The resolver must recognize "model.decoder.*", add coarse FusedMoE prefix
        "model.decoder.mlp.experts" and NextN spec weights, and exclude all draft layers.
        """
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61.*", "lm_head"],
        )

        # Step 1: loader applies hf_to_sglang_mapper
        quant_cfg.apply_weight_name_mapper(
            DeepseekV3ForCausalLMNextN.hf_to_sglang_mapper
        )
        self.assertIn("model.decoder.*", quant_cfg.exclude_modules)

        # Step 2: model __init__ calls _resolve_nextn_quant_config
        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        # Linear attention excluded
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.kv_b_proj"))
        # FusedMoE coarse prefix excluded
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        # NextN spec weights excluded
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))
        self.assertTrue(res.is_layer_excluded("model.enorm"))
        self.assertTrue(res.is_layer_excluded("model.hnorm"))
        self.assertTrue(res.is_layer_excluded("model.shared_head.norm"))
        # Base model excludes preserved
        self.assertTrue(res.is_layer_excluded("lm_head"))

    def test_loader_mapper_before_resolver_exact_layer_excluded(self):
        """When exclude_modules has exact 'model.layers.61', mapper turns it into 'model.decoder'."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61", "lm_head"],
        )

        quant_cfg.apply_weight_name_mapper(
            DeepseekV3ForCausalLMNextN.hf_to_sglang_mapper
        )
        self.assertIn("model.decoder", quant_cfg.exclude_modules)

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))

    def test_loader_mapper_before_resolver_moe_only_excluded(self):
        """When exclude_modules=["model.layers.61.mlp.experts.*"], mapper turns it into
        "model.decoder.mlp.experts.*". Resolver must add coarse "model.decoder.mlp.experts"
        so FusedMoE is unquantized while attention linear layers remain quantized.
        """
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61.mlp.experts.*"],
        )

        quant_cfg.apply_weight_name_mapper(
            DeepseekV3ForCausalLMNextN.hf_to_sglang_mapper
        )
        self.assertIn("model.decoder.mlp.experts.*", quant_cfg.exclude_modules)

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.kv_b_proj"))
        self.assertFalse(res.is_layer_excluded("model.eh_proj"))

    def test_loader_mapper_before_resolver_single_expert_excluded(self):
        """When exclude_modules=["model.layers.61.mlp.experts.0"], mapper turns it into
        "model.decoder.mlp.experts.0". Resolver must add coarse "model.decoder.mlp.experts".
        """
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61.mlp.experts.0"],
        )

        quant_cfg.apply_weight_name_mapper(
            DeepseekV3ForCausalLMNextN.hf_to_sglang_mapper
        )
        self.assertIn("model.decoder.mlp.experts.0", quant_cfg.exclude_modules)

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))

    def test_loader_mapper_before_resolver_spec_weights_remapped(self):
        """When exclude_modules has NextN spec weights, mapper replaces 'model.layers.61' with 'model.decoder'.
        The resolver must remap 'model.decoder.eh_proj' -> 'model.eh_proj'."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[
                "model.layers.61.eh_proj",
                "model.layers.61.enorm",
                "model.layers.61.hnorm",
                "model.layers.61.shared_head.norm",
            ],
        )

        quant_cfg.apply_weight_name_mapper(
            DeepseekV3ForCausalLMNextN.hf_to_sglang_mapper
        )
        self.assertIn("model.decoder.eh_proj", quant_cfg.exclude_modules)

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))
        self.assertTrue(res.is_layer_excluded("model.enorm"))
        self.assertTrue(res.is_layer_excluded("model.hnorm"))
        self.assertTrue(res.is_layer_excluded("model.shared_head.norm"))
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertFalse(res.is_layer_excluded("model.decoder.mlp.experts"))

    def test_loader_mapper_before_resolver_already_mapped_runtime_names(self):
        """Direct runtime names like 'model.decoder.*' and 'model.decoder.mlp.experts.*'."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.decoder.*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))
        self.assertTrue(res.is_layer_excluded("lm_head"))

    def test_loader_mapper_before_resolver_no_mtp_exclusion_contract(self):
        """Checkpoint contract for the ambiguous case where no MTP exclusion is present:
        MTP draft weights are treated as quantized (retaining ModelOpt FP4).
        Checkpoints with unquantized MTP layers must explicitly declare MTP exclusions.
        """
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["lm_head"],
        )

        quant_cfg.apply_weight_name_mapper(
            DeepseekV3ForCausalLMNextN.hf_to_sglang_mapper
        )
        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertEqual(res.get_name(), "modelopt_fp4")
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertFalse(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.eh_proj"))
        self.assertTrue(res.is_layer_excluded("lm_head"))

    def test_modelopt_fp4_wildcard_no_dot_before_mapper(self):
        """Production wildcard spelling 'model.layers.61*' before mapper."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.decoder.self_attn.q_a_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.decoder.mlp.experts")
        )

    def test_modelopt_fp4_wildcard_no_dot_after_mapper(self):
        """Production wildcard spelling 'model.layers.61*' after mapper transforms it into 'model.decoder*'."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61*", "lm_head"],
        )

        quant_cfg.apply_weight_name_mapper(
            DeepseekV3ForCausalLMNextN.hf_to_sglang_mapper
        )
        self.assertIn("model.decoder*", quant_cfg.exclude_modules)

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.decoder.self_attn.q_a_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.decoder.mlp.experts")
        )

    def test_modelopt_fp4_legacy_layer0_num_hidden_layers_1(self):
        """Legacy checkpoint with num_hidden_layers == 1 uses canonical NextN layer index 0."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=1)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.0.*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))

    def test_quark_legacy_layer0_num_hidden_layers_1(self):
        """Quark exclusion with num_hidden_layers == 1 correctly targets layer 0."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=1)
        quark_cfg = _FakeQuarkConfig(exclude_layers=["model.layers.0"])

        res = model._resolve_nextn_quant_config(config, quark_cfg)
        self.assertIsNone(res)

    def test_modelopt_fp4_consumer_get_quant_method_semantics(self):
        """Verify get_quant_method consumer semantics directly on Linear and FusedMoE."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.61.*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.decoder.self_attn.q_a_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.eh_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.decoder.mlp.experts")
        )

    def test_quark_quant_config_excluded_returns_none(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quark_cfg = _FakeQuarkConfig(exclude_layers=["model.decoder"])

        res = model._resolve_nextn_quant_config(config, quark_cfg)
        self.assertIsNone(res)

    def test_quark_quant_config_not_excluded_returns_config(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        quark_cfg = _FakeQuarkConfig(exclude_layers=["model.layers.0"])

        res = model._resolve_nextn_quant_config(config, quark_cfg)
        self.assertIs(res, quark_cfg)

    def test_unrelated_quant_config_passed_through(self):
        class _DummyQuantConfig(QuantizationConfig):
            def get_name(self):
                return "dummy"

            def get_quant_method(self, layer, prefix=""):
                return None

            def get_min_capability(self):
                return 0

            @classmethod
            def get_config_filenames(cls):
                return []

            @classmethod
            def from_config(cls, config):
                return cls()

            def get_scaled_act_names(self):
                return []

            def get_supported_act_dtypes(self):
                return []

        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=61)
        dummy_cfg = _DummyQuantConfig()

        res = model._resolve_nextn_quant_config(config, dummy_cfg)
        self.assertIs(res, dummy_cfg)

    def test_glm_moe_dsa_quark_shared_config_not_mutated(self):
        model = object.__new__(GlmMoeDsaForCausalLMNextN)
        config = SimpleNamespace(num_hidden_layers=46)
        quark_cfg = _FakeQuarkConfig(
            exclude_layers=["model.layers.46.mlp.experts.0"],
            layer_quant_config={
                "model.layers.46.self_attn.q_a_proj": {"scheme": "fp8"}
            },
        )

        res = model._resolve_nextn_quant_config(config, quark_cfg)

        self.assertIsNot(res, quark_cfg)
        self.assertIn("model.decoder.mlp.experts.0", res.exclude_layers)
        self.assertIn("model.decoder.mlp.experts", res.exclude_layers)
        self.assertEqual(
            quark_cfg.exclude_layers,
            ["model.layers.46.mlp.experts.0"],
        )
        self.assertEqual(
            list(quark_cfg.quant_config["layer_quant_config"].keys()),
            ["model.layers.46.self_attn.q_a_proj"],
        )
        self.assertIn(
            "model.decoder.self_attn.q_a_proj",
            res.quant_config["layer_quant_config"],
        )


class TestBailingMoeNextNModelOptQuantResolution(CustomTestCase):
    def _make_model(self):
        return object.__new__(BailingMoeForCausalLMNextN)

    def test_modelopt_fp4_not_excluded_returns_config(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertEqual(res.get_name(), "modelopt_fp4")
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertFalse(res.is_layer_excluded("model.decoder.mlp.experts"))

    def test_modelopt_fp4_mtp_layer_excluded_remaps(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.30.*"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))

    def test_modelopt_fp4_mtp_moe_only_excluded(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.30.mlp.experts.*"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))

    def test_bailing_single_expert_excluded(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.30.mlp.experts.0"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))

    def test_bailing_exact_layer_excluded(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.30"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.final_layernorm"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))

    def test_bailing_no_broad_prefix_matching(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=3)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.30.*"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertFalse(res.is_layer_excluded("model.decoder.mlp.experts"))

    def test_bailing_nextn_specific_weights_mapping(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[
                "model.layers.30.final_layernorm",
                "model.layers.30.eh_proj",
                "model.layers.30.enorm",
                "model.layers.30.hnorm",
            ],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertIn("model.final_layernorm", res.exclude_modules)
        self.assertIn("model.eh_proj", res.exclude_modules)
        self.assertIn("model.enorm", res.exclude_modules)
        self.assertIn("model.hnorm", res.exclude_modules)
        self.assertNotIn("model.decoder.final_layernorm", res.exclude_modules)

    def test_bailing_shared_quant_config_not_mutated(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        orig_exclude = ["model.layers.30.self_attn.q_a_proj"]
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=list(orig_exclude),
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNot(res, quant_cfg)
        self.assertEqual(quant_cfg.exclude_modules, orig_exclude)

    def test_bailing_no_mtp_exclusion_contract(self):
        """Bailing retains ModelOpt FP4 quant config when no MTP exclusion is present."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertEqual(res.get_name(), "modelopt_fp4")
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertFalse(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.final_layernorm"))
        self.assertTrue(res.is_layer_excluded("lm_head"))

    def test_bailing_wildcard_no_dot(self):
        """Production wildcard spelling 'model.layers.30*' and 'layers.30*'."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.30*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.layers.30.eh_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.layers.30.experts")
        )

    def test_bailing_legacy_layer0_num_hidden_layers_1(self):
        """Legacy Bailing config with num_hidden_layers == 1 uses canonical NextN layer index 0."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=1)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.0.*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.final_layernorm"))
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.layers.0.eh_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.layers.0.experts")
        )

    def test_bailing_quark_legacy_layer0_num_hidden_layers_1(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=1)
        quark_cfg = _FakeQuarkConfig(exclude_layers=["model.layers.0"])

        res = model._resolve_nextn_quant_config(config, quark_cfg)
        self.assertIsNone(res)

    def test_bailing_hybrid_runtime_prefixes_consumer_get_quant_method(self):
        """Bailing hybrid/V3 layout uses model.layers.<N> for decoder/FusedMoE and layers.<N> for eh_proj.
        Verify with ModelOptFp4Config get_quant_method consumers that all runtime forms are unquantized.
        """
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.30.*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        # Bailing hybrid runtime linear layers (eh_proj, attention projections):
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.layers.30.eh_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "layers.30.eh_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.layers.30.attention.q_b_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "layers.30.attention.q_b_proj"),
            UnquantizedLinearMethod,
        )
        # Bailing hybrid runtime FusedMoE layers (both model.layers.<N>.experts and model.layers.<N>.mlp.experts):
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.layers.30.experts")
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.layers.30.mlp.experts")
        )
        self.assertIsNone(res.get_quant_method(_mock_fused_moe(), "layers.30.experts"))
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "layers.30.mlp.experts")
        )
        # Standard decoder layout forms also retained:
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.decoder.mlp.experts")
        )
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.decoder.self_attn.q_a_proj"),
            UnquantizedLinearMethod,
        )

    def test_bailing_hybrid_quantized_mtp_consumer_get_quant_method(self):
        """When MTP is not excluded, Bailing hybrid runtime layers retain ModelOpt FP4 methods."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.layers.30.eh_proj"),
            ModelOptFp4LinearMethod,
        )
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.layers.30.attention.q_b_proj"),
            ModelOptFp4LinearMethod,
        )
        self.assertIsInstance(
            res.get_quant_method(_mock_fused_moe(), "model.layers.30.experts"),
            ModelOptNvFp4FusedMoEMethod,
        )
        self.assertIsInstance(
            res.get_quant_method(_mock_fused_moe(), "model.decoder.mlp.experts"),
            ModelOptNvFp4FusedMoEMethod,
        )

    def test_bailing_hybrid_moe_only_excluded_consumer_get_quant_method(self):
        """When only MTP MoE is excluded, FusedMoE is unquantized while attention remains quantized."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=30)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.30.mlp.experts.*"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.layers.30.attention.q_b_proj"),
            ModelOptFp4LinearMethod,
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.layers.30.experts")
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.layers.30.mlp.experts")
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.decoder.mlp.experts")
        )


class TestGlmMoeDsaNextNModelOptQuantResolution(CustomTestCase):
    def _make_model(self):
        return object.__new__(GlmMoeDsaForCausalLMNextN)

    def test_glm_moe_dsa_modelopt_fp4_not_excluded_retains_quant_config(self):
        """GLM-5.3-Flash contract: quantized GLM NextN MTP weights retain ModelOpt FP4."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=46)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertEqual(res.get_name(), "modelopt_fp4")
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertFalse(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.eh_proj"))
        self.assertTrue(res.is_layer_excluded("lm_head"))

    def test_glm_moe_dsa_modelopt_fp4_mtp_layer_excluded_remaps(self):
        """When GLM MTP layer 46 is excluded, all draft decoder modules are unquantized."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=46)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.46.*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))
        self.assertTrue(res.is_layer_excluded("lm_head"))

    def test_glm_moe_dsa_modelopt_fp4_mtp_moe_only_excluded(self):
        """When only MTP MoE is excluded, FusedMoE is unquantized while attention remains quantized."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=46)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.46.mlp.experts.*"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))

    def test_glm_moe_dsa_modelopt_fp4_single_expert_excluded(self):
        """When single expert is excluded, FusedMoE coarse prefix is added."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=46)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.46.mlp.experts.0"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertFalse(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))

    def test_glm_moe_dsa_modelopt_fp4_exact_layer_excluded(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=46)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.46"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))

    def test_glm_moe_dsa_modelopt_fp4_shared_quant_config_not_mutated(self):
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=46)
        orig_exclude = ["model.layers.46.mlp.experts.0"]
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=list(orig_exclude),
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNot(res, quant_cfg)
        self.assertEqual(quant_cfg.exclude_modules, orig_exclude)

    def test_glm_moe_dsa_wildcard_no_dot(self):
        """Production wildcard spelling 'model.layers.46*'."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=46)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.46*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.decoder.self_attn.q_a_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.decoder.mlp.experts")
        )

    def test_glm_moe_dsa_legacy_layer0_num_hidden_layers_1(self):
        """Legacy GLM config with num_hidden_layers == 1 uses canonical NextN layer index 0."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=1)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.0.*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertTrue(res.is_layer_excluded("model.decoder.self_attn.q_a_proj"))
        self.assertTrue(res.is_layer_excluded("model.decoder.mlp.experts"))
        self.assertTrue(res.is_layer_excluded("model.eh_proj"))

    def test_glm_moe_dsa_quark_legacy_layer0_num_hidden_layers_1(self):
        """Quark resolution with num_hidden_layers == 1 correctly targets layer 0."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=1)
        quark_cfg = _FakeQuarkConfig(
            exclude_layers=["model.layers.0.mlp.experts.0"],
            layer_quant_config={"model.layers.0.self_attn.q_a_proj": {"scheme": "fp8"}},
        )

        res = model._resolve_nextn_quant_config(config, quark_cfg)

        self.assertIsNot(res, quark_cfg)
        self.assertIn("model.decoder.mlp.experts.0", res.exclude_layers)
        self.assertIn("model.decoder.mlp.experts", res.exclude_layers)
        self.assertIn(
            "model.decoder.self_attn.q_a_proj",
            res.quant_config["layer_quant_config"],
        )

    def test_glm_moe_dsa_consumer_get_quant_method_semantics(self):
        """Verify get_quant_method consumer semantics directly on Linear and FusedMoE."""
        model = self._make_model()
        config = SimpleNamespace(num_hidden_layers=46)
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.46.*", "lm_head"],
        )

        res = model._resolve_nextn_quant_config(config, quant_cfg)

        self.assertIsNotNone(res)
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.decoder.self_attn.q_a_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsInstance(
            res.get_quant_method(_mock_linear(), "model.eh_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsNone(
            res.get_quant_method(_mock_fused_moe(), "model.decoder.mlp.experts")
        )


class TestNextNModelQuantConfigRetention(CustomTestCase):
    @patch(
        "sglang.srt.models.deepseek_nextn.is_mla_prefill_cp_enabled", return_value=False
    )
    @patch(
        "sglang.srt.models.deepseek_nextn.is_dsa_enable_prefill_cp", return_value=False
    )
    @patch("sglang.srt.models.deepseek_nextn.VocabParallelEmbedding")
    @patch("sglang.srt.models.deepseek_nextn.DeepseekV2DecoderLayer")
    @patch("sglang.srt.models.deepseek_nextn.RMSNorm")
    @patch("sglang.srt.models.deepseek_nextn.nn.Linear")
    def test_deepseek_model_nextn_retains_modelopt_fp4_config(
        self, mock_linear, mock_norm, mock_decoder, mock_embed, mock_dsa_cp, mock_mla_cp
    ):
        config = SimpleNamespace(
            vocab_size=1000,
            hidden_size=64,
            rms_norm_eps=1e-6,
            num_hidden_layers=61,
        )
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )

        with (
            get_context().override_server_args(
                enable_dp_attention=False,
            ),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.enable_nextn_moe_bf16_cast_to_fp8",
                return_value=False,
            ),
        ):
            model = DeepseekModelNextN(config, quant_config=quant_cfg)

        self.assertIsNotNone(
            model.quant_config, "DeepseekModelNextN should retain quant_config"
        )
        self.assertEqual(model.quant_config.get_name(), "modelopt_fp4")
        mock_decoder.assert_called_once()
        _, kwargs = mock_decoder.call_args
        self.assertEqual(kwargs.get("quant_config"), quant_cfg)

    @patch("sglang.srt.models.bailing_moe_nextn.VocabParallelEmbedding")
    @patch("sglang.srt.models.bailing_moe_nextn.BailingMoEBlock")
    @patch("sglang.srt.models.bailing_moe_nextn.RMSNorm")
    @patch("sglang.srt.models.bailing_moe_nextn.ReplicatedLinear")
    def test_bailing_moe_model_nextn_retains_modelopt_fp4_config(
        self, mock_linear, mock_norm, mock_block, mock_embed
    ):
        config = SimpleNamespace(
            vocab_size=1000,
            hidden_size=64,
            rms_norm_eps=1e-6,
            num_hidden_layers=30,
            model_type="bailing_moe",
            use_kda=False,
        )
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )

        with (
            get_context().override_server_args(
                enable_dp_attention=False,
            ),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
            ),
        ):
            _ = BailingMoEModelNextN(config, quant_config=quant_cfg)

        mock_block.assert_called_once()
        _, kwargs = mock_block.call_args
        self.assertEqual(kwargs.get("quant_config"), quant_cfg)


class TestLegacyGlm4NextNQuantRetention(CustomTestCase):
    @patch("sglang.srt.models.glm4_moe_nextn.VocabParallelEmbedding")
    @patch("sglang.srt.models.glm4_moe_nextn.Glm4MoeDecoderLayer")
    @patch("sglang.srt.models.glm4_moe_nextn.RMSNorm")
    @patch("sglang.srt.models.glm4_moe_nextn.nn.Linear")
    def test_glm4_moe_nextn_overrides_modelopt_fp4_to_none(
        self, mock_linear, mock_norm, mock_decoder, mock_embed
    ):
        config = SimpleNamespace(
            vocab_size=1000,
            hidden_size=64,
            rms_norm_eps=1e-6,
            num_hidden_layers=46,
        )
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        with (
            get_context().override_server_args(
                enable_dp_attention=False,
            ),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
            ),
        ):
            _ = Glm4MoeModelNextN(config, quant_config=quant_cfg)

        mock_decoder.assert_called_once()
        _, kwargs = mock_decoder.call_args
        self.assertIsNone(kwargs.get("quant_config"))

    @patch("sglang.srt.models.glm4_moe_lite_nextn.VocabParallelEmbedding")
    @patch("sglang.srt.models.glm4_moe_lite_nextn.Glm4MoeLiteDecoderLayer")
    @patch("sglang.srt.models.glm4_moe_lite_nextn.RMSNorm")
    @patch("sglang.srt.models.glm4_moe_lite_nextn.nn.Linear")
    def test_glm4_moe_lite_nextn_overrides_modelopt_fp4_to_none(
        self, mock_linear, mock_norm, mock_decoder, mock_embed
    ):
        config = SimpleNamespace(
            vocab_size=1000,
            hidden_size=64,
            rms_norm_eps=1e-6,
            num_hidden_layers=46,
        )
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        with (
            get_context().override_server_args(
                enable_dp_attention=False,
            ),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
            ),
        ):
            _ = Glm4MoeLiteModelNextN(config, quant_config=quant_cfg)

        mock_decoder.assert_called_once()
        _, kwargs = mock_decoder.call_args
        self.assertIsNone(kwargs.get("quant_config"))

    @patch("sglang.srt.models.glm_ocr_nextn.VocabParallelEmbedding")
    @patch("sglang.srt.models.glm_ocr_nextn.Glm4DecoderLayer")
    @patch("sglang.srt.models.glm_ocr_nextn.RMSNorm")
    @patch("sglang.srt.models.glm_ocr_nextn.nn.Linear")
    def test_glm_ocr_nextn_overrides_modelopt_fp4_to_none(
        self, mock_linear, mock_norm, mock_decoder, mock_embed
    ):
        config = SimpleNamespace(
            vocab_size=1000,
            hidden_size=64,
            rms_norm_eps=1e-6,
            num_hidden_layers=46,
        )
        quant_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        with (
            get_context().override_server_args(
                enable_dp_attention=False,
            ),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
            ),
        ):
            _ = GlmOcrModelNextN(config, quant_config=quant_cfg)

        mock_decoder.assert_called_once()
        _, kwargs = mock_decoder.call_args
        self.assertIsNone(kwargs.get("quant_config"))


if __name__ == "__main__":
    unittest.main()
