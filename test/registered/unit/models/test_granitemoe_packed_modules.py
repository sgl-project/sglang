"""GraniteMoe's fused qkv_proj must match per-projection compressed-tensors
targets — CPU-only.

A compressed-tensors checkpoint can quantize attention with its own scheme
(e.g. FP8 attention next to 4-bit experts), targeting q_proj/k_proj/v_proj
separately. SGLang fuses those into qkv_proj, so the scheme lookup can only
match them through the model's packed_modules_mapping, which the loader hands
to the quant config. Without it, loading fails with "Unable to find matching
target for model.layers.0.self_attn.qkv_proj".
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.models.granitemoe import (
    GraniteMoeForCausalLM,
    GraniteMoeSharedForCausalLM,
    GraniteMoeSWAForCausalLM,
)
from sglang.test.test_utils import CustomTestCase

_FP8_WEIGHTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "channel",
    "symmetric": True,
    "dynamic": False,
}
_FP8_DYNAMIC_ACTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "token",
    "symmetric": True,
    "dynamic": True,
}
_FP4_WEIGHTS = {
    "num_bits": 4,
    "type": "float",
    "strategy": "tensor_group",
    "group_size": 16,
    "symmetric": True,
    "dynamic": False,
}

# Shaped like an llm-compressor mixed-precision checkpoint: FP8 attention,
# NVFP4 weight-only experts, routers left unquantized.
_ATTN_TARGET = "re:.*self_attn\\.(q|k|v|o)_proj$"
_EXPERT_TARGET = "re:.*block_sparse_moe\\.experts\\.\\d+\\.(gate|up|down)_proj$"


def _config(mapping):
    config = CompressedTensorsConfig.from_config(
        {
            "format": "mixed-precision",
            "quant_method": "compressed-tensors",
            "ignore": ["re:.*block_sparse_moe\\.router$", "lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": [_ATTN_TARGET],
                    "weights": _FP8_WEIGHTS,
                    "input_activations": _FP8_DYNAMIC_ACTS,
                },
                "group_1": {
                    "targets": [_EXPERT_TARGET],
                    "weights": _FP4_WEIGHTS,
                },
            },
        }
    )
    # What the model loader does with the model class's mapping.
    config.update_packed_modules_mapping(mapping)
    return config


class TestGraniteMoePackedModules(CustomTestCase):
    QKV = "model.layers.0.self_attn.qkv_proj"
    O = "model.layers.0.self_attn.o_proj"

    def test_qkv_proj_matches_per_projection_target(self):
        config = _config(GraniteMoeForCausalLM.packed_modules_mapping)
        scheme = config.get_scheme_dict(torch.nn.Linear(1, 1), self.QKV)
        self.assertIsNotNone(scheme)
        self.assertEqual(scheme["weights"].num_bits, 8)

    def test_unfused_o_proj_still_matches(self):
        config = _config(GraniteMoeForCausalLM.packed_modules_mapping)
        scheme = config.get_scheme_dict(torch.nn.Linear(1, 1), self.O)
        self.assertEqual(scheme["weights"].num_bits, 8)

    def test_without_mapping_qkv_proj_is_unmatched(self):
        """The failure this mapping fixes, kept as the control."""
        config = _config({})
        with self.assertRaisesRegex(ValueError, "Unable to find matching target"):
            config.get_scheme_dict(torch.nn.Linear(1, 1), self.QKV)

    def test_subclasses_inherit_mapping(self):
        for cls in (GraniteMoeSharedForCausalLM, GraniteMoeSWAForCausalLM):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(
                    cls.packed_modules_mapping["qkv_proj"],
                    ["q_proj", "k_proj", "v_proj"],
                )


if __name__ == "__main__":
    unittest.main()
