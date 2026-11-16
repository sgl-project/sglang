"""CPU unit tests for MXFP8 activation-preparation dispatch."""

import importlib
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.mxfp4 import (
    Mxfp4MoEMethod,
    _prepare_flashinfer_mxfp8_activations,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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


class TestMxfp4FlashinferSituGuard(CustomTestCase):
    def _builds_runner(self, fi_kernel, activation):
        runner = object()
        method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
        method._fi_kernel = fi_kernel
        method.use_mega_moe = False
        with (
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_MXFP4,
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4.MoeRunner",
                return_value=runner,
            ),
        ):
            method.create_moe_runner(object(), MoeRunnerConfig(activation=activation))
        return method.runner is runner

    def test_cutlass_runners_reject_situ(self):
        """Kimi-K3 on SM90/SM120 loaded and served with GPT-OSS SwiGLU experts."""
        for fi_kernel in ("cutlass_sm90", "cutlass_sm120"):
            with self.subTest(fi_kernel=fi_kernel):
                with self.assertRaisesRegex(NotImplementedError, "SiTU"):
                    self._builds_runner(fi_kernel, "situ")

    def test_runner_built_when_kernel_has_the_activation(self):
        for fi_kernel, activation in (
            ("trtllm_sm100", "situ"),
            ("cutlass_sm90", "silu"),
            ("cutlass_sm120", "silu"),
        ):
            with self.subTest(fi_kernel=fi_kernel, activation=activation):
                self.assertTrue(self._builds_runner(fi_kernel, activation))


if __name__ == "__main__":
    unittest.main()
