"""CPU unit tests for MXFP8 activation-preparation dispatch."""

import importlib
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.mxfp4 import (
    _prepare_flashinfer_mxfp8_activations,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

per_token_group_quant_module = importlib.import_module(
    "sglang.kernels.ops.quantization.per_token_group_quant"
)


class TestMxfp4FlashinferActivationPrep(CustomTestCase):
    def test_sm107_handoff_miss_uses_flashinfer_quantizer(self):
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        x_quant = torch.empty(3, 64, dtype=torch.float8_e4m3fn)
        x_scale = torch.arange(6, dtype=torch.uint8).reshape(3, 2)

        with (
            patch(
                "sglang.srt.layers.moe.route_quant_handoff.take", return_value=None
            ) as take,
            patch(
                "sglang.srt.layers.quantization.mxfp4._is_sm107_supported",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_quantize",
                return_value=(x_quant, x_scale),
                create=True,
            ) as quantize,
        ):
            actual_x, packed_topk, actual_quant, actual_scale = (
                _prepare_flashinfer_mxfp8_activations(x, 64)
            )

        take.assert_called_once_with(x)
        quantize.assert_called_once_with(x, False, alignment=64)
        self.assertIs(actual_x, x)
        self.assertIsNone(packed_topk)
        self.assertIs(actual_quant, x_quant)
        self.assertTrue(torch.equal(actual_scale.view(torch.uint8), x_scale))

    def test_sm107_fp32_handoff_miss_uses_sglang_quantizer(self):
        x = torch.randn(3, 64, dtype=torch.float32)
        x_quant = torch.empty(3, 64, dtype=torch.float8_e4m3fn)
        x_scale = torch.arange(6, dtype=torch.uint8).reshape(3, 2)

        with (
            patch(
                "sglang.srt.layers.moe.route_quant_handoff.take", return_value=None
            ) as take,
            patch(
                "sglang.srt.layers.quantization.mxfp4._is_sm107_supported",
                return_value=True,
            ),
            patch.object(
                per_token_group_quant_module,
                "per_token_group_quant",
                return_value=(x_quant, x_scale),
            ) as quantize,
            patch(
                "sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_quantize",
                create=True,
            ) as flashinfer_quantize,
        ):
            actual_x, packed_topk, actual_quant, actual_scale = (
                _prepare_flashinfer_mxfp8_activations(x, 64)
            )

        take.assert_called_once_with(x)
        quantize.assert_called_once_with(x, group_size=32, scale_ue8m0=True)
        flashinfer_quantize.assert_not_called()
        self.assertIs(actual_x, x)
        self.assertIsNone(packed_topk)
        self.assertIs(actual_quant, x_quant)
        self.assertTrue(torch.equal(actual_scale.view(torch.uint8), x_scale))

    def test_other_sm10x_handoff_miss_keeps_sglang_quantizer(self):
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        x_quant = torch.empty(3, 64, dtype=torch.float8_e4m3fn)
        x_scale = torch.arange(6, dtype=torch.uint8).reshape(3, 2)

        with (
            patch("sglang.srt.layers.moe.route_quant_handoff.take", return_value=None),
            patch(
                "sglang.srt.layers.quantization.mxfp4._is_sm107_supported",
                return_value=False,
            ),
            patch.object(
                per_token_group_quant_module,
                "per_token_group_quant",
                return_value=(x_quant, x_scale),
            ) as quantize,
            patch(
                "sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_quantize",
                create=True,
            ) as flashinfer_quantize,
        ):
            actual_x, packed_topk, actual_quant, actual_scale = (
                _prepare_flashinfer_mxfp8_activations(x, 64)
            )

        quantize.assert_called_once_with(x, group_size=32, scale_ue8m0=True)
        flashinfer_quantize.assert_not_called()
        self.assertIs(actual_x, x)
        self.assertIsNone(packed_topk)
        self.assertIs(actual_quant, x_quant)
        self.assertTrue(torch.equal(actual_scale.view(torch.uint8), x_scale))

    def test_padded_input_keeps_flashinfer_quantizer(self):
        """A group-aligned input must use hidden-size-aligned quantization."""
        x = torch.randn(3, 96, dtype=torch.bfloat16)
        x_quant = torch.empty(3, 128, dtype=torch.float8_e4m3fn)
        x_scale = torch.arange(12, dtype=torch.uint8)

        with (
            patch(
                "sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_quantize",
                return_value=(x_quant, x_scale),
                create=True,
            ) as quantize,
            patch("sglang.srt.layers.moe.route_quant_handoff.take") as take,
        ):
            actual_x, packed_topk, actual_quant, actual_scale = (
                _prepare_flashinfer_mxfp8_activations(x, 128)
            )

        take.assert_not_called()
        quantize.assert_called_once_with(x, False, alignment=128)
        self.assertIs(actual_x, x)
        self.assertIsNone(packed_topk)
        self.assertIs(actual_quant, x_quant)
        self.assertEqual(actual_scale.shape, torch.Size([3, 4]))

    def test_kimi_handoff_skips_flashinfer_quantizer(self):
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        packed_topk = torch.zeros(2, 4, dtype=torch.int32)
        x_quant = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
        x_scale = torch.arange(4, dtype=torch.uint8).reshape(2, 2)

        with (
            patch(
                "sglang.srt.layers.moe.route_quant_handoff.take",
                return_value=(packed_topk, x_quant, x_scale),
            ),
            patch(
                "sglang.srt.layers.quantization.fp8_utils.flashinfer_mxfp8_quantize",
                create=True,
            ) as quantize,
        ):
            actual_x, actual_packed, actual_quant, actual_scale = (
                _prepare_flashinfer_mxfp8_activations(x, 64)
            )

        quantize.assert_not_called()
        self.assertIs(actual_x, x)
        self.assertIs(actual_packed, packed_topk)
        self.assertIs(actual_quant, x_quant)
        self.assertTrue(torch.equal(actual_scale.view(torch.uint8), x_scale))


if __name__ == "__main__":
    unittest.main()
