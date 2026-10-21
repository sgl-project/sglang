"""CPU coverage of static INT8 loading and NPU API argument contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8Int8DynamicLinearMethod,
    NPUW8A8Int8StaticLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    NPUCompressedTensorsW8A8Int8,
)

KERNEL_MODULE = "sglang.srt.hardware_backend.npu.quantization.linear_method_npu"


def _make_layer(strategy="channel", widths=(4, 2, 2)):
    scheme = NPUCompressedTensorsW8A8Int8(strategy, True, True)
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_partition_sizes=list(widths),
        input_size_per_partition=8,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
        input_size=8,
        output_size=sum(widths),
    )
    layer.weight.data.copy_(
        torch.arange(sum(widths) * 8).reshape(sum(widths), 8).to(torch.int8) - 32
    )
    layer.input_scale.data.fill_(0.125)
    layer.weight_scale.data.fill_(0.03125)
    return scheme, layer


def _prepare(scheme, layer):
    with patch(f"{KERNEL_MODULE}.npu_format_cast", side_effect=lambda x: x):
        scheme.process_weights_after_loading(layer)


class TestNPUStaticInt8Linear(unittest.TestCase):
    def test_scheme_selection_and_asymmetric_rejection(self):
        self.assertIsInstance(
            NPUCompressedTensorsW8A8Int8("channel", True, True).kernel,
            NPUW8A8Int8StaticLinearMethod,
        )
        self.assertIsInstance(
            NPUCompressedTensorsW8A8Int8("channel", False, True).kernel,
            NPUW8A8Int8DynamicLinearMethod,
        )
        with self.assertRaisesRegex(NotImplementedError, "symmetric activations"):
            NPUCompressedTensorsW8A8Int8("channel", True, False)

    def test_fused_qkv_scale_loader_preserves_distinct_widths(self):
        scheme, layer = _make_layer()
        self.assertEqual(layer.input_scale.shape, (3,))
        for shard in ("q", "k", "v"):
            layer.input_scale.load_qkv_weight(torch.tensor([0.25]), shard_id=shard)
        original = layer.weight.detach().clone()
        _prepare(scheme, layer)
        torch.testing.assert_close(layer.weight, original.T.contiguous())
        torch.testing.assert_close(layer.input_scale, torch.tensor([0.25]))
        torch.testing.assert_close(layer.npu_static_quant_scale, torch.full((8,), 0.25))
        torch.testing.assert_close(
            layer.npu_static_quant_offset, torch.zeros(8, dtype=torch.int32)
        )

    def test_gate_up_scale_loader(self):
        scheme, layer = _make_layer(widths=(6, 6))
        for shard in (0, 1):
            layer.input_scale.load_merged_column_weight(
                torch.tensor([0.5]), shard_id=shard
            )
        _prepare(scheme, layer)
        self.assertEqual(layer.weight.shape, (8, 12))
        torch.testing.assert_close(layer.input_scale, torch.tensor([0.5]))

    def test_tensor_weight_scales_expand_without_requantization(self):
        scheme, layer = _make_layer("tensor")
        layer.weight_scale.data.copy_(torch.tensor([0.125, 0.25, 0.5]))
        original = layer.weight.detach().clone()
        _prepare(scheme, layer)
        torch.testing.assert_close(layer.weight, original.T.contiguous())
        torch.testing.assert_close(
            layer.weight_scale, torch.tensor([0.125] * 4 + [0.25] * 2 + [0.5] * 2)
        )

    def test_different_fused_input_scales_are_not_silently_merged(self):
        scheme, layer = _make_layer()
        layer.input_scale.data[1] = 0.25
        with self.assertRaisesRegex(ValueError, "identical input scales"):
            _prepare(scheme, layer)

    def test_invalid_calibration_is_rejected(self):
        for parameter in ("input_scale", "weight_scale"):
            for value in (float("nan"), float("inf"), 0.0, -1.0):
                with self.subTest(parameter=parameter, value=value):
                    scheme, layer = _make_layer()
                    getattr(layer, parameter).data.flatten()[0] = value
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        _prepare(scheme, layer)

    def test_tp_partition_is_explicitly_rejected(self):
        scheme = NPUCompressedTensorsW8A8Int8("channel", True, True)
        for extra in (
            {"input_size": 16, "output_size": 4},
            {"input_size": 8, "output_size": 8},
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(NotImplementedError, "TP=1"),
            ):
                scheme.create_weights(
                    torch.nn.Module(), [4], 8, torch.float16, lambda *a: None, **extra
                )
        layer = torch.nn.Module()
        layer.tp_group = SimpleNamespace(world_size=2)
        with self.assertRaisesRegex(NotImplementedError, "TP=1"):
            scheme.create_weights(layer, [4], 8, torch.float16, lambda *a: None)

    def test_empty_batch_does_not_launch_npu_ops(self):
        scheme, layer = _make_layer()
        _prepare(scheme, layer)
        ops = MagicMock()
        with patch.object(torch.ops, "npu", ops, create=True):
            result = scheme.apply_weights(
                layer, torch.empty(2, 0, 8, dtype=torch.bfloat16), None
            )
        self.assertEqual(result.shape, (2, 0, 8))
        ops.npu_quantize.assert_not_called()
        ops.npu_quant_matmul.assert_not_called()

    def test_fixed_scale_and_float_bias_follow_integer_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((1, 8), (5, 8), (2, 3, 8)):
                for with_bias in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, bias=with_bias):
                        scheme, layer = _make_layer("tensor", widths=(4, 4))
                        layer.weight_scale.data.copy_(torch.tensor([0.015625, 0.03125]))
                        weight = layer.weight.detach().to(torch.int64)
                        scales = torch.tensor([0.015625] * 4 + [0.03125] * 4)
                        _prepare(scheme, layer)
                        x = (
                            torch.linspace(-2.5, 2.5, torch.tensor(shape).prod().item())
                            .reshape(shape)
                            .to(dtype)
                        )
                        bias = (
                            torch.linspace(-0.03, 0.02, 8).to(dtype)
                            if with_bias
                            else None
                        )
                        q = (
                            torch.round(x.float() / 0.125)
                            .clamp(-128, 127)
                            .to(torch.int8)
                        )
                        expected = (
                            (q.reshape(-1, 8).to(torch.int64) @ weight.T).float()
                            * scales
                            * 0.125
                        )
                        if bias is not None:
                            expected += bias.float()
                        expected = expected.reshape(*shape[:-1], 8).to(dtype)

                        ops = MagicMock()
                        ops.npu_quantize.return_value = q.reshape(-1, 8)

                        def matmul(qx, w, scale, *, pertoken_scale, bias, output_dtype):
                            result = (qx.to(torch.int64) @ w.to(torch.int64)).float()
                            result = result * scale * pertoken_scale[:, None]
                            if bias is not None:
                                result += bias
                            return result.to(output_dtype)

                        ops.npu_quant_matmul.side_effect = matmul
                        with patch.object(torch.ops, "npu", ops, create=True):
                            actual = scheme.apply_weights(layer, x, bias)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        quant_call = ops.npu_quantize.call_args.args
                        self.assertEqual(quant_call[2].dtype, torch.int32)
                        torch.testing.assert_close(
                            quant_call[2], torch.zeros(8, dtype=torch.int32)
                        )
                        torch.testing.assert_close(
                            quant_call[1], torch.full((8,), 0.125)
                        )
                        self.assertEqual(quant_call[3:], (torch.qint8, -1, True))
                        mm_call = ops.npu_quant_matmul.call_args.kwargs
                        torch.testing.assert_close(
                            mm_call["pertoken_scale"],
                            torch.full((x.numel() // 8,), 0.125),
                        )
                        if with_bias:
                            self.assertEqual(mm_call["bias"].dtype, torch.float32)
                        ops.npu_dynamic_quant.assert_not_called()


if __name__ == "__main__":
    unittest.main()
