"""Tests for FP8 Marlin utilities."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import marlin_utils_fp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestFp8MarlinBias(CustomTestCase):
    def test_dense_bias_remains_in_logical_output_order(self):
        size_k = 32
        size_n = 32

        layer = torch.nn.Module()
        layer.input_size_per_partition = size_k
        layer.output_size_per_partition = size_n
        layer.orig_dtype = torch.float16
        layer.weight_block_size = None
        layer.weight = torch.nn.Parameter(
            torch.zeros((size_k, size_n), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones((size_n,), dtype=torch.float32), requires_grad=False
        )
        original_bias = torch.arange(size_n, dtype=torch.float16)
        layer.bias = torch.nn.Parameter(original_bias.clone(), requires_grad=False)

        with (
            patch.object(
                marlin_utils_fp8,
                "marlin_make_workspace",
                return_value=torch.empty(0, dtype=torch.int32),
            ),
            patch.object(
                marlin_utils_fp8,
                "gptq_marlin_repack",
                return_value=torch.empty(0, dtype=torch.int32),
                create=True,
            ),
        ):
            marlin_utils_fp8.prepare_fp8_layer_for_marlin(layer)

        torch.testing.assert_close(layer.bias, original_bias)


if __name__ == "__main__":
    unittest.main(verbosity=3)
