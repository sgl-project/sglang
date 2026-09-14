"""Unit tests for the DSPARK draft quant-config metadata and the fused
CommitKvProj dequant support gate.

Covers the staged changes in:

- python/sglang/srt/models/deepseek_v4_dspark.py
  ``DeepseekV4ForCausalLMDSpark`` now declares ``packed_modules_mapping``
  and ``hf_to_sglang_mapper`` mirroring the target DeepseekV4ForCausalLM,
  so hybrid quantization configs (MoE MXFP4 + dense FP8 per-channel)
  resolve ``fp8_channelwise_layers`` against the draft runtime names
  (``stages.N.self_attn.*`` / ``stages.N.mlp.*``) instead of the checkpoint
  names (``mtp.N.attn.*`` / ``mtp.N.ffn.*``), and the fused shared-experts
  ``gate_up_proj`` expands into its gate/up shards during the ignore check.

- python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py
  ``_dequant_supported`` returns False when a float8_e4m3fn linear carries no
  ``weight_scale_inv`` attribute.  Per-channel FP8 (W8A8Fp8) linears store
  ``weight_scale`` instead of the 128x128 block scale, so the gate used to
  raise ``AttributeError`` on such linears; now it declines the fused
  dequant path and ``CommitKvProj.execute`` falls back to the per-linear
  torch path.

The tested objects are extracted directly from the checked-out sources with
``ast`` instead of ``import sglang``: the sglang runtime stack does not
import on macOS / pure-CPU hosts (torch.compile in the quantization package
pulls in torch._inductor), and importing would also resolve a site-packages
install rather than this checkout.  The only runtime dependency is torch,
so the test runs anywhere on CPU:

    python test/ppu/unit/test_deepseek_v4_dspark_quant_metadata.py
    pytest test/ppu/unit/test_deepseek_v4_dspark_quant_metadata.py -v
"""

import ast
import re
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SGLANG_SRC = _REPO_ROOT / "python" / "sglang"

_BLOCK = 128


def _top_level(path):
    return ast.parse(Path(path).read_text(encoding="utf-8")).body


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


# WeightsMapper + WeightsMapping (python/sglang/srt/models/utils.py).
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
WEIGHTS_MAPPER = _utils_ns["WeightsMapper"]

# packed_modules_mapping + hf_to_sglang_mapper
# (python/sglang/srt/models/deepseek_v4_dspark.py).
_draft_class = next(
    n
    for n in _top_level(_SGLANG_SRC / "srt/models/deepseek_v4_dspark.py")
    if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4ForCausalLMDSpark"
)
_draft_attrs = {
    t.id: n
    for n in _draft_class.body
    if isinstance(n, ast.Assign)
    for t in n.targets
    if isinstance(t, ast.Name)
}
for _name in ("packed_modules_mapping", "hf_to_sglang_mapper"):
    assert _name in _draft_attrs, f"{_name} missing from DeepseekV4ForCausalLMDSpark"
_draft_ns = _compile_nodes(
    [_draft_attrs["packed_modules_mapping"], _draft_attrs["hf_to_sglang_mapper"]],
    {"WeightsMapper": WEIGHTS_MAPPER},
)
HF_TO_SGLANG_MAPPER = _draft_ns["hf_to_sglang_mapper"]
PACKED_MODULES_MAPPING = _draft_ns["packed_modules_mapping"]

# should_ignore_layer (python/sglang/srt/layers/quantization/compressed_tensors/utils.py).
_sil_ns = _compile_nodes(
    [
        n
        for n in _top_level(
            _SGLANG_SRC / "srt/layers/quantization/compressed_tensors/utils.py"
        )
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
SHOULD_IGNORE_LAYER = _sil_ns["should_ignore_layer"]

# Dequant support gate
# (python/sglang/kernels/ops/speculative/dspark/dspark_draft_model.py; checkouts
# predating the kernel sweep keep it under
# srt/speculative/dspark_components/kernels/dspark_draft_model.py).
_DSPARK_DRAFT_MODEL_RELS = (
    "kernels/ops/speculative/dspark/dspark_draft_model.py",
    "srt/speculative/dspark_components/kernels/dspark_draft_model.py",
)


def _dspark_draft_model_source() -> Path:
    for rel in _DSPARK_DRAFT_MODEL_RELS:
        candidate = _SGLANG_SRC / rel
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "dspark_draft_model.py not found under python/sglang "
        f"(checked {', '.join(_DSPARK_DRAFT_MODEL_RELS)})"
    )


_kernel_ns = _compile_nodes(
    [
        n
        for n in _top_level(_dspark_draft_model_source())
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "_dequant_supported",
            "_fused_commit_kv_proj_supported",
            "_block_quant_stack_applies",
        }
    ],
    {"torch": torch},
)
DEQUANT_SUPPORTED = _kernel_ns["_dequant_supported"]
FUSED_COMMIT_KV_PROJ_SUPPORTED = _kernel_ns["_fused_commit_kv_proj_supported"]


def _linear(weight, **attrs):
    return SimpleNamespace(weight=weight, **attrs)


def _per_channel_fp8_linear(out_dim=512, in_dim=224):
    """Mirror W8A8Fp8LinearMethod: per-channel scale, no weight_scale_inv."""
    return _linear(
        torch.zeros(out_dim, in_dim, dtype=torch.float8_e4m3fn),
        weight_scale=torch.ones(out_dim, 1),
    )


def _block_fp8_linear(out_dim=512, in_dim=224):
    """Mirror the 128x128 block-quant scale layout."""
    return _linear(
        torch.zeros(out_dim, in_dim, dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.ones(
            (out_dim + _BLOCK - 1) // _BLOCK, (in_dim + _BLOCK - 1) // _BLOCK
        ),
    )


class TestDsparkV4QuantMetadata(unittest.TestCase):
    def test_mapper_resolves_attention_checkpoint_name_to_runtime_name(self):
        self.assertEqual(
            HF_TO_SGLANG_MAPPER.apply_list(["mtp.0.attn.wkv"]),
            ["stages.0.self_attn.wkv"],
        )

    def test_mapper_resolves_shared_expert_shard_names(self):
        checkpoint_patterns = [
            "mtp.1.ffn.shared_experts.w1",
            "mtp.1.ffn.shared_experts.w2",
            "mtp.1.ffn.shared_experts.w3",
        ]
        self.assertEqual(
            HF_TO_SGLANG_MAPPER.apply_list(checkpoint_patterns),
            [
                "stages.1.mlp.shared_experts.gate_proj",
                "stages.1.mlp.shared_experts.down_proj",
                "stages.1.mlp.shared_experts.up_proj",
            ],
        )

    def test_mapped_fp8_channelwise_layers_match_draft_runtime_modules(self):
        # Mxfp4Config.apply_weight_name_mapper applies the draft mapper to
        # the checkpoint-style fp8_channelwise_layers entries; the mapped
        # names must then match the draft runtime module prefixes.
        checkpoint_patterns = [
            "mtp.0.attn.wkv",
            "mtp.0.ffn.shared_experts.w1",
            "mtp.0.ffn.shared_experts.w3",
        ]
        runtime_patterns = HF_TO_SGLANG_MAPPER.apply_list(checkpoint_patterns)
        self.assertTrue(
            SHOULD_IGNORE_LAYER(
                "stages.0.self_attn.wkv",
                ignore=runtime_patterns,
                fused_mapping=PACKED_MODULES_MAPPING,
            )
        )

    def test_fused_gate_up_proj_expands_to_shard_patterns(self):
        checkpoint_patterns = [
            "mtp.0.ffn.shared_experts.w1",
            "mtp.0.ffn.shared_experts.w3",
        ]
        runtime_patterns = HF_TO_SGLANG_MAPPER.apply_list(checkpoint_patterns)
        # The fused shared-experts module is recognized through its shards.
        self.assertTrue(
            SHOULD_IGNORE_LAYER(
                "stages.0.mlp.shared_experts.gate_up_proj",
                ignore=runtime_patterns,
                fused_mapping=PACKED_MODULES_MAPPING,
            )
        )
        # Without the packed-modules mapping the fused name never matches.
        self.assertFalse(
            SHOULD_IGNORE_LAYER(
                "stages.0.mlp.shared_experts.gate_up_proj",
                ignore=runtime_patterns,
                fused_mapping={},
            )
        )

    def test_routed_experts_stay_out_of_the_fp8_channelwise_set(self):
        checkpoint_patterns = ["mtp.0.attn.wkv", "mtp.0.ffn.shared_experts.w1"]
        runtime_patterns = HF_TO_SGLANG_MAPPER.apply_list(checkpoint_patterns)
        # MoE routed experts stay MXFP4: the fp8 channelwise set must not
        # swallow them.
        self.assertFalse(
            SHOULD_IGNORE_LAYER(
                "stages.0.mlp.experts",
                ignore=runtime_patterns,
                fused_mapping=PACKED_MODULES_MAPPING,
            )
        )


class TestDequantSupported(unittest.TestCase):
    def test_unquantized_dtypes_are_supported(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                linear = _linear(torch.zeros(64, 64, dtype=dtype))
                self.assertTrue(DEQUANT_SUPPORTED(linear))

    def test_non_fp8_quantized_dtype_is_unsupported(self):
        linear = _linear(torch.zeros(64, 64, dtype=torch.int8))
        self.assertFalse(DEQUANT_SUPPORTED(linear))

    def test_per_channel_fp8_without_weight_scale_inv_is_unsupported(self):
        # Regression: per-channel W8A8Fp8 linears only carry weight_scale;
        # the gate used to raise AttributeError on weight_scale_inv.
        linear = _per_channel_fp8_linear()
        self.assertFalse(DEQUANT_SUPPORTED(linear))

    def test_block_fp8_with_matching_scale_shape_is_supported(self):
        linear = _block_fp8_linear(out_dim=512, in_dim=224)
        self.assertTrue(DEQUANT_SUPPORTED(linear))

    def test_fp8_with_mismatched_scale_shape_is_unsupported(self):
        # A per-channel-shaped scale (out_dim, 1) must decline the fused path.
        linear = _linear(
            torch.zeros(512, 224, dtype=torch.float8_e4m3fn),
            weight_scale_inv=torch.ones(512, 1),
        )
        self.assertFalse(DEQUANT_SUPPORTED(linear))


class TestFusedCommitKvProjSupport(unittest.TestCase):
    def test_bf16_linears_take_the_fused_path(self):
        linears = [
            _linear(
                torch.zeros(64, 64, dtype=torch.bfloat16),
                quant_method=SimpleNamespace(),
            )
            for _ in range(3)
        ]
        self.assertTrue(FUSED_COMMIT_KV_PROJ_SUPPORTED(wkv_linears=linears))

    def test_per_channel_fp8_linears_decline_the_fused_path(self):
        # Per-channel FP8 must fall back to the per-linear torch path in
        # CommitKvProj.execute instead of crashing inside the fused kernel.
        linears = [
            _linear(
                torch.zeros(512, 224, dtype=torch.float8_e4m3fn),
                weight_scale=torch.ones(512, 1),
                quant_method=SimpleNamespace(block_quant=False),
            )
            for _ in range(3)
        ]
        self.assertFalse(FUSED_COMMIT_KV_PROJ_SUPPORTED(wkv_linears=linears))

    def test_block_fp8_linears_take_the_fused_path(self):
        linears = [
            _linear(
                torch.zeros(512, 224, dtype=torch.float8_e4m3fn),
                weight_scale_inv=torch.ones(
                    (512 + _BLOCK - 1) // _BLOCK, (224 + _BLOCK - 1) // _BLOCK
                ),
                quant_method=SimpleNamespace(block_quant=False),
            )
            for _ in range(3)
        ]
        self.assertTrue(FUSED_COMMIT_KV_PROJ_SUPPORTED(wkv_linears=linears))


if __name__ == "__main__":
    unittest.main()
