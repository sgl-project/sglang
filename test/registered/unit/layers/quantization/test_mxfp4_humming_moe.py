import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.mxfp4_humming_moe import (
    Mxfp4HummingMoEMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeFp8Method:
    def __init__(self):
        self.args = None
        self.kwargs = None

    def create_weights(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class TestMxfp4HummingMoEMethod(CustomTestCase):
    def _create_weights_and_get_scale_dtype(self, use_megamoe: bool):
        fp8_method = _FakeFp8Method()
        method = Mxfp4HummingMoEMethod(fp8_method, prefix="model.layers.0.mlp")
        layer = torch.nn.Module()

        with patch(
            "sglang.srt.layers.quantization.mxfp4_humming_moe.get_moe_a2a_backend"
        ) as get_backend:
            get_backend.return_value.is_megamoe.return_value = use_megamoe
            method.create_weights(
                layer,
                num_experts=8,
                hidden_size=128,
                intermediate_size_per_partition=64,
                params_dtype=torch.bfloat16,
                weight_loader="sentinel",
            )

        self.assertEqual(fp8_method.args[0], layer)
        self.assertEqual(fp8_method.kwargs["weight_loader"], "sentinel")
        return fp8_method.kwargs["fp4_scale_dtype"]

    def test_create_weights_uses_e8m0_scales_for_humming(self):
        self.assertEqual(
            self._create_weights_and_get_scale_dtype(use_megamoe=False),
            torch.float8_e8m0fnu,
        )

    def test_create_weights_keeps_fp32_scales_for_megamoe(self):
        self.assertEqual(
            self._create_weights_and_get_scale_dtype(use_megamoe=True),
            torch.float32,
        )


if __name__ == "__main__":
    unittest.main()
