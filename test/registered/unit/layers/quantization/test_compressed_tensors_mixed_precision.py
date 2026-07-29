"""CPU regression tests for multi-group ("mixed-precision") compressed-tensors configs.

Two independent bugs made every layer of such a checkpoint fall back to an
unquantized method, which surfaces as an OOM (weights allocated in bf16) or as
``KeyError: '...experts.w2_input_global_scale'`` while loading weights -- never
as a readable "this layer is not quantized" message.

1. ``ignore`` was matched by raw substring. llm-compressor writes *parent*
   modules into ``ignore`` (e.g. ``model.layers.0.mlp.experts.0``, which owns no
   weights of its own), so such an entry swallowed its quantized children
   (``model.layers.0.mlp.experts.0.gate_proj``). A real Qwen3.5-VL NVFP4
   checkpoint has 10240 such entries and lost 30810 quantized modules this way.

2. The activation-quantization gate read the *top-level* ``format``. When
   config_groups disagree, compressed-tensors sets the top-level format to
   ``mixed-precision`` and puts the real format on each group, so
   ``input_activations`` was dropped for every group and no W4A4 / W8A8 scheme
   could ever be selected.

These are pure config-parsing paths (no weights are created and no kernels run),
so they run on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
    should_ignore_layer,
)
from sglang.test.test_utils import CustomTestCase

EXPERTS_LAYER = "model.language_model.layers.0.mlp.experts"
GATE_PROJ = f"{EXPERTS_LAYER}.0.gate_proj"

# NVFP4 experts + FP8 attention/GDN projections, i.e. two groups with different
# formats -> compressed-tensors writes format="mixed-precision" at the top level.
MIXED_PRECISION_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "config_groups": {
        "group_0": {
            "format": "float-quantized",
            "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$"],
            "weights": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "channel",
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "token",
                "dynamic": True,
            },
        },
        "group_1": {
            "format": "nvfp4-pack-quantized",
            "targets": ["re:.*mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$"],
            "weights": {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "strategy": "tensor_group",
                "group_size": 16,
                "dynamic": False,
            },
            "input_activations": {
                "num_bits": 4,
                "type": "float",
                "symmetric": True,
                "strategy": "tensor_group",
                "group_size": 16,
                "dynamic": "local",
            },
        },
    },
    # The parent-module entries llm-compressor emits. They own no weights, so
    # they are no-ops for quantization -- but they must not swallow their
    # children either.
    "ignore": [
        "model.language_model.layers.0.mlp.gate",
        "model.language_model.layers.0.linear_attn",
        f"{EXPERTS_LAYER}.0",
        f"{EXPERTS_LAYER}.1",
    ],
}


class TestIgnoreListPrefixMatching(CustomTestCase):
    """A parent module in `ignore` must not de-quantize its children."""

    def test_parent_module_does_not_swallow_children(self):
        ignore = [f"{EXPERTS_LAYER}.0", "model.language_model.layers.0.linear_attn"]

        # The parent itself is still ignored: semantics are unchanged.
        self.assertTrue(should_ignore_layer(f"{EXPERTS_LAYER}.0", ignore=ignore))
        self.assertTrue(
            should_ignore_layer(
                "model.language_model.layers.0.linear_attn", ignore=ignore
            )
        )

        # Its quantized children are not.
        self.assertFalse(should_ignore_layer(GATE_PROJ, ignore=ignore))
        self.assertFalse(
            should_ignore_layer(
                "model.language_model.layers.0.linear_attn.out_proj", ignore=ignore
            )
        )

    def test_module_suffix_target_still_matches(self):
        # Targets written as a module suffix keep working (dotted-path boundary).
        self.assertTrue(
            check_equal_or_regex_match(
                "model.layers.0.self_attn.kv_b_proj", ["self_attn.kv_b_proj"]
            )
        )
        self.assertTrue(check_equal_or_regex_match("model.lm_head", ["lm_head"]))
        # ... but only on a boundary, never mid-token.
        self.assertFalse(
            check_equal_or_regex_match("model.layers.0.gate_proj", ["ate"])
        )
        self.assertFalse(
            check_equal_or_regex_match(
                "model.layers.0.mlp.shared_expert_gate", ["gate"]
            )
        )

    def test_exact_and_regex_targets_unchanged(self):
        self.assertTrue(check_equal_or_regex_match(GATE_PROJ, [GATE_PROJ]))
        self.assertTrue(check_equal_or_regex_match(GATE_PROJ, ["re:.*gate_proj$"]))
        self.assertFalse(check_equal_or_regex_match(GATE_PROJ, ["re:.*down_proj$"]))


class TestMixedPrecisionFormat(CustomTestCase):
    """The per-group `format` must win over a top-level "mixed-precision"."""

    def test_input_activations_survive_mixed_precision(self):
        quant_config = CompressedTensorsConfig.from_config(MIXED_PRECISION_CONFIG)

        for target, scheme in quant_config.target_scheme_map.items():
            self.assertIsNotNone(
                scheme["input_activations"],
                f"input_activations dropped for target {target}",
            )

        expert_target = "re:.*mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$"
        expert_scheme = quant_config.target_scheme_map[expert_target]
        self.assertEqual(expert_scheme["format"], "nvfp4-pack-quantized")
        self.assertEqual(expert_scheme["weights"].num_bits, 4)
        self.assertEqual(expert_scheme["input_activations"].num_bits, 4)

    def test_moe_resolves_nvfp4_scheme(self):
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            CompressedTensorsW4A4Nvfp4MoE,
        )

        quant_config = CompressedTensorsConfig.from_config(MIXED_PRECISION_CONFIG)
        # Before the fix this returned None (experts ignored) -> the MoE fell back
        # to UnquantizedFusedMoEMethod and blew up later in load_weights.
        scheme = quant_config.get_moe_scheme(
            torch.nn.Module(), layer_name=EXPERTS_LAYER
        )
        self.assertIsInstance(scheme, CompressedTensorsW4A4Nvfp4MoE)


if __name__ == "__main__":
    unittest.main()
