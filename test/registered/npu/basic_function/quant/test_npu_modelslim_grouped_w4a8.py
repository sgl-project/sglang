"""Regression coverage for grouped R1 ModelSlim W4A8 loading."""

import unittest
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="base-b-test-1-npu-a3")


class TestGroupedW4A8(unittest.TestCase):
    def schemes(self, group_size, tp_size=1):
        prefix = "model.layers.3.mlp.experts"
        description = {"group_size": group_size}
        for name in ["gate_proj", "up_proj", "down_proj"]:
            description[f"{prefix}.0.{name}.weight"] = "W4A8_DYNAMIC"
        layer = torch.nn.Module()
        layer.moe_tp_size = tp_size
        return ModelSlimConfig(description).get_moe_scheme(layer, prefix)

    def test_checkpoint_group_size_reaches_both_projections(self):
        for scheme in self.schemes(256):
            self.assertEqual(scheme.group_size, 256)
            self.assertFalse(scheme.is_per_channel_weight)

    def test_tensor_parallel_size_reaches_both_projections(self):
        for scheme in self.schemes(0, 4):
            self.assertEqual(scheme.tp_size, 4)

    def test_scales_use_unpacked_output_dimension(self):
        # Initialize the public quantization package before its backend module.
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            NPUW4A8Int8MoEMethod,
        )

        kernel = NPUW4A8Int8MoEMethod(is_per_channel_weight=False)
        weight = torch.empty((1, 16, 64), dtype=torch.int8)
        scale = torch.ones((1, 32, 1))
        group_scale = torch.full((1, 32, 4), 0.125)
        with patch.object(torch.Tensor, "npu", lambda tensor, *args, **kwargs: tensor):
            actual = kernel._process_scale(weight, scale, group_scale, False)
        self.assertEqual(tuple(actual.shape), (1, 4, 32))
        self.assertEqual(actual.dtype, torch.int64)
        self.assertTrue(torch.all(actual == 1040187392).item())


if __name__ == "__main__":
    unittest.main()
