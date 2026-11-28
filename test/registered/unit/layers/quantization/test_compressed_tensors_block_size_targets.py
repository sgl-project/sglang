"""Compressed-tensors: the weight block size of block-FP8 checkpoints that list modules as targets — CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.test.test_utils import CustomTestCase


def _weights(block):
    return {
        "num_bits": 8,
        "type": "float",
        "strategy": "block",
        "block_structure": block,
        "symmetric": True,
        "dynamic": False,
    }


_ACTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "group",
    "group_size": 128,
    "symmetric": True,
    "dynamic": True,
}


def _config(groups):
    return CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized",
            "quant_method": "compressed-tensors",
            "ignore": ["lm_head"],
            "config_groups": {
                f"group_{i}": {
                    "targets": targets,
                    "weights": _weights(block),
                    "input_activations": _ACTS,
                }
                for i, (targets, block) in enumerate(groups)
            },
        }
    )


class TestCompressedTensorsBlockSizeTargets(CustomTestCase):
    def test_linear_target(self):
        self.assertEqual(
            _config([(["Linear"], [128, 128])]).weight_block_size, [128, 128]
        )

    def test_module_targets(self):
        targets = [
            "model.language_model.layers.0.mlp.down_proj",
            "model.language_model.layers.0.self_attn.q_proj",
        ]
        self.assertEqual(_config([(targets, [128, 128])]).weight_block_size, [128, 128])

    def test_mixed_block_sizes_raise(self):
        cfg = _config(
            [
                (["model.layers.0.mlp.down_proj"], [128, 128]),
                (["model.layers.0.self_attn.q_proj"], [64, 64]),
            ]
        )
        with self.assertRaises(ValueError):
            cfg.weight_block_size


if __name__ == "__main__":
    unittest.main()
