"""CPU regression tests for ModelSlim MXFP8 weight-scale layouts."""

import unittest

import torch

from sglang.srt.layers.quantization.modelslim.schemes.modelslim_mxfp8 import (
    ModelSlimMXFP8Scheme,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestModelSlimMXFP8ScaleLayout(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.scheme = ModelSlimMXFP8Scheme()

    def test_scale_placeholder_rounds_up_partial_block(self):
        layer = torch.nn.Module()

        self.scheme.create_weights(
            layer=layer,
            input_size_per_partition=4304,
            output_partition_sizes=[2],
            input_size=4304,
            output_size=2,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(layer.weight_scale.shape, (2, 135))

    def test_post_load_pads_odd_scale_count_for_pair_layout(self):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty((2, 4304), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.ones((2, 135), dtype=torch.uint8), requires_grad=False
            ),
        )
        layer.register_parameter("bias", None)

        self.scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (4304, 2))
        self.assertEqual(layer.weight_scale_inv.shape, (68, 2, 2))
        self.assertTrue(torch.all(layer.weight_scale_inv[-1, :, 1] == 0))
        self.assertFalse(hasattr(layer, "weight_scale"))


if __name__ == "__main__":
    unittest.main()
