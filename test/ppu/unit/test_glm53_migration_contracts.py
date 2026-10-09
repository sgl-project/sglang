"""Static contract & dispatcher tests for GLM-5.3-Flash migration (v0.5.18 → v0.5.21).

These tests are designed to run on Mac/CPU without GPU/CUDA/PPU dependencies.
They use AST analysis and monkeypatching instead of real imports where GPU
extensions would be loaded at import time.

Migrated from the v0.5.20 migration branch (commit 8f4fff8345 + later CR
revisions) and adapted to the v0.5.21 APIs.

Covers:
  - INT8 W8A8 MoE dispatcher: monkeypatch-driven branch selection
  - W8A8 INT8 MoE scheme: AST-level structural contract
  - GLM-5.3 MXFP4 weight-name mapper: behavioral test on real WeightsMapper

Part 1 of the v0.5.20 source file (CP hybrid communicator AST contract on
communicator_mhc_hybrid_cp.py) is intentionally omitted: the v0.5.20
migration built a dedicated MHCHybridDSACPLayerCommunicator, but v0.5.21's
layer_boundary system already provides the CP collectives natively (migrated
commit 0f962b3247), so there is no equivalent module to contract-test.
Part numbering is kept aligned with the v0.5.20 source file for
cross-reference.
"""

import ast
import os
import re
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Optional
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helper: locate python/ source root for AST analysis
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PYTHON_ROOT = os.path.join(_REPO_ROOT, "python")


def _has_npu_dynamic_quant() -> bool:
    """Return True iff torch.ops.npu.npu_dynamic_quant is callable.

    NPUCompressedTensorsW8A8Int8DynamicMoE.__init__ constructs
    HiddenStatesDynamicQuant(quant_dtype=torch.int8) which accesses
    torch.ops.npu.npu_dynamic_quant at init time. On platforms without
    torch_npu (PPU / CUDA / Mac) this attribute does not exist and raises
    AttributeError, so the NPU-only test must be skipped.
    """
    try:
        import torch

        return callable(getattr(torch.ops.npu, "npu_dynamic_quant", None))
    except Exception:
        return False


_HAS_NPU_OPS = _has_npu_dynamic_quant()


def _parse_module(rel_path: str) -> ast.Module:
    """Parse a .py file relative to the python/ source root and return its AST."""
    full = os.path.join(_PYTHON_ROOT, *rel_path.split("/"))
    with open(full, "r") as fh:
        return ast.parse(fh.read(), filename=full)


# ===================================================================
# Part 1: CP hybrid communicator — N/A in v0.5.21 (intentionally omitted)
# ===================================================================
# The v0.5.20 migration (commit 21692be5ec) built a dedicated
# MHCHybridDSACPLayerCommunicator in communicator_mhc_hybrid_cp.py and
# contract-tested it with ScatterMode-based AST checks. v0.5.21's
# layer_boundary system already provides the CP collectives natively
# (migrated commit 0f962b3247), so no equivalent module exists and there
# is nothing to contract-test here.


# ===================================================================
# Part 2: INT8 W8A8 MoE dispatcher selection test
# ===================================================================
class TestW8A8Int8MoEDispatcherSelection(unittest.TestCase):
    """Tests the three-way branch in CompressedTensorsConfig.get_moe_scheme()
    for the _is_dynamic_token_w8a8 path:
      - NPU -> NPUCompressedTensorsW8A8Int8DynamicMoE
      - PPU -> CompressedTensorsW8A8Int8MoE
      - Other -> NotImplementedError

    [CR Warning 6 - real dispatch] 旧代码直接实例化 scheme 或手写 if/else 模拟。
    改为 mock get_scheme_dict 旁路层名匹配，真正执行 get_moe_scheme 的生产分支。
    """

    @classmethod
    def setUpClass(cls):
        """Pre-check: the compressed_tensors module can be imported.
        This may fail if compressed_tensors (the pip package) is not installed."""
        try:
            import sglang.srt.layers.quantization.compressed_tensors.compressed_tensors as ct_mod  # noqa: E501

            cls.ct_mod = ct_mod
            cls.skip_reason = None
        except ImportError as e:
            cls.ct_mod = None
            cls.skip_reason = f"Cannot import compressed_tensors dispatcher: {e}"

    def setUp(self):
        if self.skip_reason:
            self.skipTest(self.skip_reason)

    def _make_weight_quant(self):
        """Minimal weight quant config for dynamic token W8A8."""
        from compressed_tensors.quantization import QuantizationStrategy

        return SimpleNamespace(
            num_bits=8,
            strategy=QuantizationStrategy.CHANNEL.value,
            symmetric=True,
            dynamic=False,
            type="int",
            group_size=None,
        )

    def _make_input_quant(self):
        from compressed_tensors.quantization import QuantizationStrategy

        return SimpleNamespace(
            num_bits=8,
            strategy=QuantizationStrategy.TOKEN.value,
            symmetric=True,
            dynamic=True,
        )

    def _make_config_instance(self):
        """Create a minimal CompressedTensorsConfig for invoking get_moe_scheme."""
        ct = self.ct_mod
        cfg = object.__new__(ct.CompressedTensorsConfig)
        cfg.target_scheme_map = {}
        cfg.ignore = []
        cfg.quant_format = None
        cfg.packed_modules_mapping = {}
        return cfg

    def _call_get_moe_scheme(self, is_npu: bool, is_ppu: bool):
        """Invoke the real get_moe_scheme production path under controlled
        platform flags.

        Mocks ``get_scheme_dict`` to return a W8A8 INT8 dynamic-token scheme
        dict, bypassing layer-name matching while exercising **all** downstream
        dispatch logic (predicate checks + three-way NPU/PPU/other branch).
        """
        ct = self.ct_mod
        cfg = self._make_config_instance()
        wq = self._make_weight_quant()
        iq = self._make_input_quant()
        scheme_dict = {"weights": wq, "input_activations": iq}

        # get_moe_scheme only forwards layer to get_scheme_dict; a stub suffices.
        dummy_layer = SimpleNamespace()

        with (
            patch.object(ct, "_is_npu", is_npu),
            patch.object(ct, "_is_ppu", is_ppu),
            patch.object(cfg, "get_scheme_dict", return_value=scheme_dict),
            patch.object(cfg, "_add_fused_moe_to_target_scheme_map"),
        ):
            return cfg.get_moe_scheme(
                layer=dummy_layer,
                layer_name="model.layers.0.block_sparse_moe",
            )

    # --- Three-way dispatch via real production code path ---

    @unittest.skipUnless(
        _HAS_NPU_OPS,
        "NPU scheme __init__ requires torch.ops.npu.npu_dynamic_quant "
        "(only available with torch_npu on Ascend)",
    )
    def test_npu_returns_npu_scheme(self):
        """On NPU, get_moe_scheme returns NPUCompressedTensorsW8A8Int8DynamicMoE."""
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8_moe import (  # noqa: E501
            NPUCompressedTensorsW8A8Int8DynamicMoE,
        )

        result = self._call_get_moe_scheme(is_npu=True, is_ppu=False)
        self.assertIsInstance(result, NPUCompressedTensorsW8A8Int8DynamicMoE)

    def test_ppu_returns_generic_scheme(self):
        """On PPU, get_moe_scheme returns CompressedTensorsW8A8Int8MoE."""
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8_generic_moe import (  # noqa: E501
            CompressedTensorsW8A8Int8MoE,
        )

        result = self._call_get_moe_scheme(is_npu=False, is_ppu=True)
        self.assertIsInstance(result, CompressedTensorsW8A8Int8MoE)
        self.assertIn("PPU", CompressedTensorsW8A8Int8MoE.__doc__)

    def test_other_platform_raises_not_implemented(self):
        """On non-NPU, non-PPU platforms, get_moe_scheme raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as ctx:
            self._call_get_moe_scheme(is_npu=False, is_ppu=False)
        self.assertIn("NPU and PPU", str(ctx.exception))

    def test_predicate_consistency(self):
        """The _is_dynamic_token_w8a8 predicate matches our fixture quant args."""
        cfg = self._make_config_instance()
        wq = self._make_weight_quant()
        iq = self._make_input_quant()
        self.assertTrue(cfg._is_dynamic_token_w8a8(wq, iq))


# ===================================================================
# Part 3: W8A8 Int8 MoE scheme AST contract (static, no pip dep)
# ===================================================================
class TestW8A8Int8MoESchemeASTContract(unittest.TestCase):
    """AST-level contract test for the generic W8A8 INT8 MoE scheme.

    Verifies the class exists, inherits CompressedTensorsMoEScheme,
    and exposes required methods without importing the module
    (which would require the compressed_tensors pip package).
    """

    @classmethod
    def setUpClass(cls):
        cls.tree = _parse_module(
            "sglang/srt/layers/quantization/compressed_tensors/schemes/"
            "compressed_tensors_w8a8_int8_generic_moe.py"
        )
        cls.class_map = {
            node.name: node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.ClassDef)
        }

    def test_class_exists_and_inherits_moe_scheme(self):
        cls_node = self.class_map.get("CompressedTensorsW8A8Int8MoE")
        self.assertIsNotNone(cls_node, "CompressedTensorsW8A8Int8MoE class not found")
        base_names = []
        for base in cls_node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        self.assertIn("CompressedTensorsMoEScheme", base_names)

    def test_required_methods_present(self):
        cls_node = self.class_map["CompressedTensorsW8A8Int8MoE"]
        methods = {
            node.name
            for node in ast.walk(cls_node)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for expected in (
            "__init__",
            "create_weights",
            "process_weights_after_loading",
            "create_moe_runner",
            "apply_weights",
        ):
            self.assertIn(expected, methods, f"Missing method: {expected}")

    def test_module_exports_only_the_scheme_class(self):
        """__all__ should export exactly CompressedTensorsW8A8Int8MoE."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            elements = [
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]
                            self.assertEqual(
                                elements,
                                ["CompressedTensorsW8A8Int8MoE"],
                            )
                            return
        self.fail("__all__ not found in module")

    def test_is_ppu_module_level_cache_present(self):
        """The module-level _is_ppu = is_ppu() pattern must be present
        for the create_moe_runner deep_gemm branch to work at runtime."""
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_is_ppu":
                        found = True
                        break
        self.assertTrue(found, "_is_ppu module-level cache not found")


# ===================================================================
# Part 4: GLM-5.3 MXFP4 weight-name mapper — behavioral test
# ===================================================================


# --- Extract real WeightsMapper from AST (mirrors dspark test style) ---
def _top_level(path):
    return ast.parse(path.read_text(encoding="utf-8")).body


def _compile_nodes(nodes, base_globals):
    """Compile AST nodes extracted from the checkout into a namespace."""

    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    ns = dict(base_globals)
    exec(compile(module, "<sglang source extract>", "exec"), ns)
    return ns


from pathlib import Path

_SGLANG_SRC = Path(_PYTHON_ROOT) / "sglang"

_utils_ns = _compile_nodes(
    [
        n
        for n in _top_level(_SGLANG_SRC / "srt/models/utils.py")
        if (isinstance(n, ast.ClassDef) and n.name == "WeightsMapper")
        or (
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "WeightsMapping" for t in n.targets
            )
        )
    ],
    {
        "Mapping": Mapping,
        "Optional": Optional,
        "dataclass": dataclass,
        "field": field,
    },
)
_WEIGHTS_MAPPER = _utils_ns["WeightsMapper"]

# --- Extract hf_to_sglang_mapper from Glm5NextForConditionalGeneration ---
_glm5_class = next(
    n
    for n in _top_level(_SGLANG_SRC / "srt/models/glm5_next.py")
    if isinstance(n, ast.ClassDef) and n.name == "Glm5NextForConditionalGeneration"
)
_glm5_attrs = {
    t.id: n
    for n in _glm5_class.body
    if isinstance(n, ast.Assign)
    for t in n.targets
    if isinstance(t, ast.Name)
}
assert "hf_to_sglang_mapper" in _glm5_attrs, (
    "hf_to_sglang_mapper missing from Glm5NextForConditionalGeneration"
)
_glm5_mapper_ns = _compile_nodes(
    [_glm5_attrs["hf_to_sglang_mapper"]],
    {"WeightsMapper": _WEIGHTS_MAPPER},
)
_GLM5_MAPPER = _glm5_mapper_ns["hf_to_sglang_mapper"]

# --- Extract should_ignore_layer (best-effort, fallback to suffix matcher) ---
_SHOULD_IGNORE_LAYER = None
try:
    _sil_path = _SGLANG_SRC / "srt/layers/quantization/compressed_tensors/utils.py"
    if _sil_path.is_file():
        _sil_ns = _compile_nodes(
            [
                n
                for n in _top_level(_sil_path)
                if isinstance(n, ast.FunctionDef)
                and n.name
                in {
                    "should_ignore_layer",
                    "check_equal_or_regex_match",
                    "_is_equal_or_regex_match",
                }
            ],
            {"re": re, "MappingProxyType": MappingProxyType},
        )
        _SHOULD_IGNORE_LAYER = _sil_ns.get("should_ignore_layer")
except Exception:
    pass

if _SHOULD_IGNORE_LAYER is None:
    # Fallback: simplified suffix matcher equivalent to the real function
    # for exact-string and suffix matching (the two modes used in MXFP4
    # fp8_channelwise_layers).  Clearly marked as a static stand-in.
    def _SHOULD_IGNORE_LAYER(prefix, ignore):
        return any(prefix == ig or prefix.endswith("." + ig) for ig in ignore)


class TestGlm5NextMXFP4MapperBehavior(unittest.TestCase):
    """Behavioral test: real WeightsMapper.apply_list on combined paths.

    Validates the two-stage substr→prefix semantics required by the MXFP4
    channelwise metadata remap (patch #5 1c8267b9fe).
    """

    mapper = _GLM5_MAPPER

    # --- AST structural assertions (retained from Part 4 v1) ---
    def test_mapper_has_qkv_in_substr(self):
        self.assertIn("attn.qkv", self.mapper.orig_to_new_substr)
        self.assertEqual(self.mapper.orig_to_new_substr["attn.qkv"], "attn.qkv_proj")

    def test_mapper_has_prefixes(self):
        self.assertIn("model.language_model.", self.mapper.orig_to_new_prefix)
        self.assertIn("model.visual", self.mapper.orig_to_new_prefix)

    # --- Combined path behavioral tests ---
    def test_visual_qkv_combined_path(self):
        """model.visual.blocks.0.attn.qkv → visual.blocks.0.attn.qkv_proj"""
        result = self.mapper.apply_list(["model.visual.blocks.0.attn.qkv"])
        self.assertEqual(result, ["visual.blocks.0.attn.qkv_proj"])

    def test_language_model_qkv_combined_path(self):
        """model.language_model.layers.61.self_attn.qkv →
        model.layers.61.self_attn.qkv_proj"""
        result = self.mapper.apply_list(
            ["model.language_model.layers.61.self_attn.qkv"]
        )
        self.assertEqual(result, ["model.layers.61.self_attn.qkv_proj"])

    def test_bare_qkv(self):
        """attn.qkv → attn.qkv_proj (no prefix involved)"""
        result = self.mapper.apply_list(["attn.qkv"])
        self.assertEqual(result, ["attn.qkv_proj"])

    def test_prefix_only_no_qkv(self):
        """model.language_model.layers.0.mlp.gate_proj → model.layers.0.mlp.gate_proj"""
        result = self.mapper.apply_list(["model.language_model.layers.0.mlp.gate_proj"])
        self.assertEqual(result, ["model.layers.0.mlp.gate_proj"])

    def test_visual_prefix_only(self):
        """model.visual.patch_embed.proj.weight → visual.patch_embed.proj.weight"""
        result = self.mapper.apply_list(["model.visual.patch_embed.proj.weight"])
        self.assertEqual(result, ["visual.patch_embed.proj.weight"])

    def test_no_match_passthrough(self):
        """Unrelated key passes through unchanged."""
        result = self.mapper.apply_list(["lm_head.weight"])
        self.assertEqual(result, ["lm_head.weight"])

    # --- Batch remap: MXFP4 fp8_channelwise_layers scenario ---
    def test_fp8_channelwise_batch_remap(self):
        """Simulate the fp8_channelwise_layers remap that Mxfp4Config.apply_weight_name_mapper does."""
        channelwise = [
            "model.language_model.layers.61.self_attn.qkv",
            "model.visual.blocks.0.attn.qkv",
            "model.language_model.layers.0.mlp.gate_proj",
            "model.visual.patch_embed.proj.weight",
        ]
        remapped = self.mapper.apply_list(channelwise)
        self.assertEqual(
            remapped,
            [
                "model.layers.61.self_attn.qkv_proj",
                "visual.blocks.0.attn.qkv_proj",
                "model.layers.0.mlp.gate_proj",
                "visual.patch_embed.proj.weight",
            ],
        )

    # --- should_ignore_layer integration ---
    def test_remapped_name_matches_ignore(self):
        """After remap, should_ignore_layer correctly matches runtime prefixes."""
        remapped = self.mapper.apply_list(
            [
                "model.language_model.layers.61.self_attn.qkv",
                "model.visual.blocks.0.attn.qkv",
            ]
        )
        # The runtime module prefix for these layers:
        self.assertTrue(
            _SHOULD_IGNORE_LAYER("model.layers.61.self_attn.qkv_proj", remapped),
            "should_ignore_layer must match after remap",
        )
        self.assertTrue(
            _SHOULD_IGNORE_LAYER("visual.blocks.0.attn.qkv_proj", remapped),
            "should_ignore_layer must match after remap",
        )


if __name__ == "__main__":
    unittest.main()
