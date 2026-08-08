"""Unit tests for serialized Marlin checkpoint configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import get_quantization_config
from sglang.srt.layers.quantization.marlin_utils import (
    MarlinConfig,
    MarlinLinearMethod,
    scalar_types,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMarlinConfig(CustomTestCase):
    @patch("sglang.srt.utils.is_cpu", return_value=False)
    def test_direct_quantization_resolution(self, _):
        self.assertIs(get_quantization_config("marlin"), MarlinConfig)

    def test_serialized_checkpoint_detection(self):
        for quant_config in (
            {"checkpoint_format": "marlin"},
            {"is_marlin_format": True},
        ):
            with self.subTest(quant_config=quant_config):
                self.assertEqual(
                    MarlinConfig.override_quantization_method(
                        quant_config, user_quant="marlin"
                    ),
                    "marlin",
                )

    def test_config_parsing(self):
        config = MarlinConfig.from_config({"group_size": 128, "lm_head": True})

        self.assertEqual(config.group_size, 128)
        self.assertTrue(config.lm_head_quantized)
        self.assertEqual(config.quant_type, scalar_types.uint4b8)

    @patch("sglang.srt.layers.quantization.marlin_utils.apply_gptq_marlin_linear")
    def test_linear_method_uses_native_marlin_kernel(self, apply_mock):
        config = MarlinConfig(group_size=128, lm_head_quantized=False)
        method = MarlinLinearMethod(config)
        layer = SimpleNamespace(
            B=torch.empty(8, 512, dtype=torch.int32),
            s=torch.empty(1, 64),
            zp=torch.empty(0, dtype=torch.int32),
            g_idx=torch.empty(0, dtype=torch.int32),
            g_idx_sort_indices=torch.empty(0, dtype=torch.int32),
            workspace=torch.empty(1, dtype=torch.int32),
        )
        x = torch.empty(2, 128)
        expected = torch.empty(2, 64)
        apply_mock.return_value = expected

        output = method.apply(layer, x)

        self.assertIs(output, expected)
        kwargs = apply_mock.call_args.kwargs
        self.assertIs(kwargs["input"], x)
        self.assertIs(kwargs["weight"], layer.B)
        self.assertIs(kwargs["weight_scale"], layer.s)
        self.assertIs(kwargs["wtype"], scalar_types.uint4b8)
        self.assertEqual(kwargs["input_size_per_partition"], 128)
        self.assertEqual(kwargs["output_size_per_partition"], 64)


if __name__ == "__main__":
    unittest.main()
