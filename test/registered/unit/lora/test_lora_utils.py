"""Unit tests for LoRA target-module name normalization in srt/lora/utils.py.

Covers the mapping of GDN (GatedDeltaNet) input-projection module names to
the fused ``in_proj_qkvz`` module. HF/PEFT adapters for models like Qwen3.5
target either the 2-way split (``in_proj_qkv`` + ``in_proj_z``) or the 4-way
split (``in_proj_q/k/v/z``); before the mapping existed, these raw names
reached ``LoRAMemoryPool.init_buffers`` and crashed the server with
``NotImplementedError: get_hidden_dim not implemented for in_proj_qkv``
(see issue #30168).

Usage:
    python -m pytest test/registered/unit/lora/test_lora_utils.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import unittest

from sglang.srt.lora.utils import get_normalized_target_modules, get_stacked_multiply
from sglang.test.test_utils import CustomTestCase


class TestGetNormalizedTargetModules(CustomTestCase):
    def test_gdn_two_way_split(self):
        self.assertEqual(
            get_normalized_target_modules(["in_proj_qkv", "in_proj_z"]),
            {"in_proj_qkvz"},
        )

    def test_gdn_two_way_split_prefixed(self):
        self.assertEqual(
            get_normalized_target_modules(
                [
                    "model.layers.5.linear_attn.in_proj_qkv",
                    "model.layers.5.linear_attn.in_proj_z",
                ]
            ),
            {"in_proj_qkvz"},
        )

    def test_gdn_four_way_split(self):
        self.assertEqual(
            get_normalized_target_modules(
                ["in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z"]
            ),
            {"in_proj_qkvz"},
        )

    def test_gdn_merged_name_unchanged(self):
        self.assertEqual(
            get_normalized_target_modules(["in_proj_qkvz"]), {"in_proj_qkvz"}
        )

    def test_mamba_in_proj_not_remapped(self):
        # Bare `in_proj` (Mamba/LFM2/NemotronH) is a different module and must
        # not be folded into the GDN mapping.
        self.assertEqual(get_normalized_target_modules(["in_proj"]), {"in_proj"})

    def test_gdn_ba_projections_not_remapped(self):
        # in_proj_b / in_proj_a belong to the separate in_proj_ba module and
        # remain unsupported (not silently absorbed into in_proj_qkvz).
        self.assertEqual(
            get_normalized_target_modules(["in_proj_b", "in_proj_a"]),
            {"in_proj_b", "in_proj_a"},
        )


class TestGetStackedMultiply(CustomTestCase):
    def test_stacked_multiply(self):
        self.assertEqual(get_stacked_multiply("in_proj_qkvz"), 4)
        self.assertEqual(get_stacked_multiply("in_proj"), 2)


if __name__ == "__main__":
    unittest.main()
