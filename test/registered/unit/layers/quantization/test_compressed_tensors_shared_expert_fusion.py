"""Compressed-tensors: a shared expert kept unquantized must not be fused into the quantized MoE — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.test.test_utils import CustomTestCase

_W4A16 = {
    "num_bits": 4,
    "type": "int",
    "strategy": "group",
    "group_size": 32,
    "symmetric": True,
    "dynamic": False,
}


def _config(ignore):
    return CompressedTensorsConfig.from_config(
        {
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(ignore),
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": _W4A16}},
        }
    )


class TestCompressedTensorsSharedExpertFusion(CustomTestCase):
    def test_fusable_without_shared_expert_in_ignore(self):
        self.assertTrue(_config(["lm_head", "re:.*mlp.gate$"]).can_fuse_shared_expert())

    def test_shared_expert_gate_alone_does_not_block(self):
        cfg = _config(["lm_head", "model.layers.0.mlp.shared_expert_gate"])
        self.assertTrue(cfg.can_fuse_shared_expert())

    def test_ignored_shared_expert_blocks_fusion(self):
        cfg = _config(
            [
                "lm_head",
                "model.layers.0.mlp.shared_expert.gate_proj",
                "model.layers.0.mlp.shared_expert.up_proj",
                "model.layers.0.mlp.shared_expert.down_proj",
            ]
        )
        self.assertFalse(cfg.can_fuse_shared_expert())

    def test_ignored_shared_experts_regex_blocks_fusion(self):
        # DeepSeek-family modules are "shared_experts"
        self.assertFalse(_config(["re:.*shared_experts.*"]).can_fuse_shared_expert())


if __name__ == "__main__":
    unittest.main()
