"""Actual Ascend static INT8 kernels against independent CPU integer math."""

import unittest
from itertools import product

import torch
import torch_npu  # noqa: F401

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    NPUCompressedTensorsW8A8Int8,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=15, suite="base-b-test-1-npu-a3")


def make_layer(strategy, widths, dtype):
    scheme = NPUCompressedTensorsW8A8Int8(strategy, True, True)
    with torch.device("npu"):
        layer = torch.nn.Module()
        scheme.create_weights(
            layer,
            list(widths),
            128,
            dtype,
            lambda *a: None,
            input_size=256,
            output_size=2 * sum(widths),
        )
    layer.weight.data.copy_(
        (torch.arange(sum(widths) * 128).reshape(sum(widths), 128) % 17 - 8).to(
            torch.int8
        )
    )
    layer.input_scale.data.fill_(0.125)
    if strategy == "tensor":
        layer.weight_scale.data.copy_(torch.arange(1, len(widths) + 1).float() / 64)
    else:
        layer.weight_scale.data.fill_(0.03125)
    return scheme, layer


class TestNPUStaticInt8Linear(CustomTestCase):
    def test_real_fixed_scale_linear(self):
        for strategy, dtype, widths, shape, bias_enabled in product(
            ("channel", "tensor"),
            (torch.float16, torch.bfloat16),
            ((256,), (128, 64, 64), (128, 128)),
            ((3, 128), (2, 3, 128)),
            (False, True),
        ):
            with self.subTest(
                strategy=strategy,
                dtype=dtype,
                widths=widths,
                shape=shape,
                bias=bias_enabled,
            ):
                scheme, layer = make_layer(strategy, widths, dtype)
                # Fused shards can have different activation calibration
                # scales; the shared input must use their maximum.
                input_scales = torch.arange(1, len(widths) + 1) * 0.125
                layer.input_scale.data.copy_(input_scales)
                input_scale = input_scales.max().item()
                weight = layer.weight.detach().cpu().long()
                scales = layer.weight_scale.detach().cpu().flatten()
                if strategy == "tensor":
                    scales = torch.cat([s.expand(w) for s, w in zip(scales, widths)])
                scheme.process_weights_after_loading(layer)
                x = (
                    torch.linspace(-70, 70, torch.tensor(shape).prod().item())
                    .reshape(shape)
                    .to(dtype)
                )
                bias = (
                    torch.linspace(-0.03, 0.02, sum(widths)).to(dtype)
                    if bias_enabled
                    else None
                )
                q = torch.round(x.float() / input_scale).clamp(-128, 127).long()
                expected = (q.reshape(-1, 128) @ weight.T).float() * (
                    scales * input_scale
                )
                if bias is not None:
                    expected += bias.float()
                actual = scheme.apply_weights(
                    layer, x.npu(), bias.npu() if bias is not None else None
                )
                torch.testing.assert_close(
                    actual.cpu(),
                    expected.reshape(*shape[:-1], sum(widths)).to(dtype),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    scheme.apply_weights(layer, x[:0].npu(), None).cpu(),
                    torch.empty(0, *shape[1:-1], sum(widths), dtype=dtype),
                )

    def test_unsupported_configuration(self):
        with self.assertRaisesRegex(NotImplementedError, "symmetric activations"):
            NPUCompressedTensorsW8A8Int8("channel", True, False)
        scheme = NPUCompressedTensorsW8A8Int8("channel", True, True)
        with self.assertRaisesRegex(ValueError, "float16 and bfloat16"):
            scheme.create_weights(torch.nn.Module(), [128], 128, torch.float32, None)


if __name__ == "__main__":
    unittest.main()
