import unittest
from types import SimpleNamespace

import sgl_kernel  # noqa: F401
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    static_quant_fp8,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (
    CompressedTensorsW8A16Fp8,
)
from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu-intel")


def _dequant_fp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()


def _ue8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    return (scale.to(torch.int32) << 23).view(torch.float32)


def _reference_w8a8_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    dequant_weight = weight.float() * weight_scale.float().view(-1, 1)
    out = x.float().matmul(dequant_weight.t())
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def _requantize_tensor_scale_int8(
    weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    max_w_scale = weight_scale.max()
    int8_info = torch.iinfo(torch.int8)
    requantized_weight = weight.clone()
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_dq = weight[start:end, :].float() * weight_scale[idx]
        requantized_weight[start:end, :] = torch.clamp(
            torch.round(weight_dq / max_w_scale),
            int8_info.min,
            int8_info.max,
        ).to(torch.int8)
        start = end
    return requantized_weight, max_w_scale.expand(weight.size(0)).clone()


class TestCPUFp8QuantOps(CustomTestCase):
    def test_per_token_group_quant_fp8_cpu(self):
        x = torch.linspace(-2.0, 2.0, steps=128, dtype=torch.bfloat16).reshape(2, 64)

        q, scale = per_token_group_quant_fp8(x, 32)
        ref_scale = x.float().view(2, 2, 32).abs().amax(-1).clamp(min=1e-10) / 448.0
        ref_q = torch.clamp(
            x.float() / ref_scale.repeat_interleave(32, dim=-1), -448.0, 448.0
        ).to(torch.float8_e4m3fn)

        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (2, 2))
        torch.testing.assert_close(scale, ref_scale)
        torch.testing.assert_close(q.float(), ref_q.float())
        torch.testing.assert_close(
            _dequant_fp8(q, scale.repeat_interleave(32, dim=-1)),
            x.float(),
            atol=0.08,
            rtol=0.08,
        )

    def test_scaled_fp8_quant_cpu_dynamic_and_static(self):
        x = torch.randn(3, 64, dtype=torch.bfloat16) / 4

        q, scale = scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        self.assertEqual(q.shape, x.shape)
        self.assertEqual(scale.shape, (3, 1))
        torch.testing.assert_close(
            _dequant_fp8(q, scale), x.float(), atol=0.02, rtol=0.02
        )

        static_scale = torch.tensor([0.01], dtype=torch.float32)
        static_q, returned_scale = static_quant_fp8(x, static_scale)
        self.assertIs(returned_scale, static_scale)
        torch.testing.assert_close(
            static_q.float(),
            torch.clamp(x.float() / static_scale, -448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .float(),
        )

    def test_static_quant_fp8_cpu_3d(self):
        """Regression: CPU static quant must preserve ndim >= 2 input shapes."""
        x = torch.randn(2, 3, 64, dtype=torch.bfloat16) / 4
        static_scale = torch.tensor([0.01], dtype=torch.float32)

        q, returned_scale = static_quant_fp8(x, static_scale)

        self.assertEqual(q.shape, x.shape)
        self.assertIs(returned_scale, static_scale)
        torch.testing.assert_close(
            q.float(),
            torch.clamp(x.float() / static_scale, -448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .float(),
        )

    def test_scaled_fp8_quant_cpu_padding(self):
        x = torch.randn(2, 32, dtype=torch.bfloat16) / 3

        q, scale = scaled_fp8_quant(
            x, num_token_padding=4, use_per_token_if_dynamic=True
        )

        self.assertEqual(q.shape, (4, 32))
        self.assertEqual(scale.shape, (4, 1))
        torch.testing.assert_close(
            _dequant_fp8(q[:2], scale[:2]), x.float(), atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(q[2:].float(), torch.zeros(2, 32))
        torch.testing.assert_close(scale[2:], torch.zeros(2, 1))

    def test_mxfp8_group_quantize_cpu(self):
        x = torch.randn(3, 64, dtype=torch.float32) / 5

        q, scale_u8 = mxfp8_group_quantize(x)

        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale_u8.dtype, torch.uint8)
        self.assertEqual(scale_u8.shape, (3, 2))
        dequant = q.float() * _ue8m0_to_float(scale_u8).repeat_interleave(32, dim=-1)
        torch.testing.assert_close(dequant, x, atol=0.04, rtol=0.04)

    def test_compressed_tensors_w8a16_fp8_cpu(self):
        x = torch.randn(5, 64, dtype=torch.bfloat16)
        weight = (torch.randn(32, 64) / 4).to(torch.float8_e4m3fn)
        weight_scale = torch.tensor([0.02], dtype=torch.float32)
        bias = torch.randn(32, dtype=torch.float32) / 10
        layer = SimpleNamespace(weight=weight, weight_scale=weight_scale)
        scheme = CompressedTensorsW8A16Fp8(QuantizationStrategy.TENSOR, False)

        out = scheme.apply_weights(layer, x, bias)
        ref = torch.ops.sgl_kernel.fp8_per_tensor_scaled_mm_cpu(
            x, weight, weight_scale, bias, x.dtype, False
        )

        torch.testing.assert_close(out, ref)

    def test_compressed_tensors_w8a16_fp8_cpu_rejects_multi_scale(self):
        weight = (torch.randn(32, 64) / 4).to(torch.float8_e4m3fn)
        layer = SimpleNamespace(
            weight=weight,
            weight_scale=torch.tensor([0.02, 0.03], dtype=torch.float32),
        )
        scheme = CompressedTensorsW8A16Fp8(QuantizationStrategy.TENSOR, False)

        with self.assertRaisesRegex(NotImplementedError, "multiple weight scales"):
            scheme.process_weights_after_loading(layer)

    def test_compressed_tensors_w8a8_fp8_cpu_unsupported(self):
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        weight = (torch.randn(5, 64) / 4).to(torch.float8_e4m3fn)
        weight_scale = torch.tensor(0.02, dtype=torch.float32)
        layer = SimpleNamespace(
            weight=weight, weight_scale=weight_scale, input_scale=None
        )
        quant_args = QuantizationArgs(num_bits=8, strategy=QuantizationStrategy.TENSOR)
        scheme = CompressedTensorsW8A8Fp8(quant_args, False)

        with self.assertRaisesRegex(NotImplementedError, "W8A8 FP8 linear"):
            scheme.apply_weights(layer, x, None)


class TestCPUInt8QuantOps(CustomTestCase):
    def test_per_token_quant_int8_python_dispatch_cpu(self):
        """Public CPU dispatch keeps the CUDA dtype and scale-shape contract."""
        for k in (64, 33):
            with self.subTest(k=k):
                x = torch.linspace(-1.0, 1.0, steps=2 * 3 * k).reshape(2, 3, k)
                x = x.to(torch.bfloat16)

                x_q, scales, x_sum = per_token_quant_int8(x, cal_sum=True)

                self.assertEqual(x_q.dtype, torch.int8)
                self.assertEqual(x_q.shape, x.shape)
                self.assertEqual(scales.shape, x.shape[:-1] + (1,))
                torch.testing.assert_close(
                    x_q.float() * scales.float(), x.float(), atol=0.02, rtol=0.02
                )
                torch.testing.assert_close(x_sum, x.sum(dim=-1))

    @unittest.skipUnless(
        cpu_has_amx_support(),
        "Compressed-tensors W8A8 INT8 CPU path requires the Intel AMX backend.",
    )
    def test_compressed_tensors_w8a8_int8_cpu(self):
        x = torch.linspace(-0.8, 0.9, steps=3 * 64, dtype=torch.float32).reshape(3, 64)
        x = x.to(torch.bfloat16)
        weight = (torch.arange(64 * 64).reshape(64, 64) % 33 - 16).to(torch.int8)
        weight_scale = torch.linspace(0.005, 0.02, steps=64, dtype=torch.float32)
        bias = torch.linspace(-0.1, 0.1, steps=64, dtype=torch.float32)
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale=torch.nn.Parameter(weight_scale, requires_grad=False),
        )
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, True)
        scheme.process_weights_after_loading(layer)
        self.assertTrue(layer.use_intel_amx_backend)

        out = scheme.apply_weights(layer, x, bias)
        ref = _reference_w8a8_int8_linear(x, weight, weight_scale, bias)

        torch.testing.assert_close(out, ref, atol=0.12, rtol=0.05)

    @unittest.skipUnless(
        cpu_has_amx_support(),
        "Compressed-tensors W8A8 INT8 CPU path requires the Intel AMX backend.",
    )
    def test_compressed_tensors_w8a8_int8_cpu_tensor_scales(self):
        x = torch.linspace(-0.7, 0.8, steps=3 * 64, dtype=torch.float32).reshape(3, 64)
        x = x.to(torch.bfloat16)
        weight = (torch.arange(64 * 64).reshape(64, 64) % 41 - 20).to(torch.int8)
        weight_scale = torch.tensor([0.01, 0.04], dtype=torch.float32)
        bias = torch.linspace(-0.05, 0.05, steps=64, dtype=torch.float32)
        logical_widths = [32, 32]
        ref_weight, ref_weight_scale = _requantize_tensor_scale_int8(
            weight, weight_scale, logical_widths
        )
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale=torch.nn.Parameter(weight_scale, requires_grad=False),
            logical_widths=logical_widths,
        )
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.TENSOR, False, True)
        scheme.process_weights_after_loading(layer)
        self.assertTrue(layer.use_intel_amx_backend)
        self.assertEqual(layer.weight_scale.numel(), weight.size(0))

        out = scheme.apply_weights(layer, x, bias)
        ref = _reference_w8a8_int8_linear(x, ref_weight, ref_weight_scale, bias)

        torch.testing.assert_close(out, ref, atol=0.16, rtol=0.06)

    def test_compressed_tensors_w8a8_int8_cpu_rejects_unpacked_weight(self):
        """Unsupported CPU dimensions fail before using the packed AMX contract."""
        x = torch.randn(3, 33, dtype=torch.bfloat16)
        weight = torch.randint(-16, 16, (16, 33), dtype=torch.int8)
        weight_scale = torch.rand(16, dtype=torch.float32) / 16
        layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            weight_scale=torch.nn.Parameter(weight_scale, requires_grad=False),
        )
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, True)
        scheme.process_weights_after_loading(layer)

        self.assertFalse(layer.use_intel_amx_backend)
        with self.assertRaisesRegex(NotImplementedError, "AMX-packed weights"):
            scheme.apply_weights(layer, x, None)

    def test_compressed_tensors_w8a8_int8_cpu_rejects_static_input(self):
        """CPU rejects checkpoint input scales instead of silently changing quantization semantics."""
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, True, True)

        with self.assertRaisesRegex(NotImplementedError, "dynamic symmetric"):
            scheme.process_weights_after_loading(SimpleNamespace())

    def test_compressed_tensors_w8a8_int8_cpu_rejects_asymmetric_input(self):
        """CPU rejects activation zero-points instead of silently changing quantization semantics."""
        scheme = CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL, False, False)

        with self.assertRaisesRegex(NotImplementedError, "dynamic symmetric"):
            scheme.process_weights_after_loading(SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
