"""Fused static calibration must load before backend-specific processing."""

import unittest
from itertools import product
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
    NPUCompressedTensorsW8A8Int8,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_layer(strategy, widths, symmetric=True, k=128):
    scheme = CompressedTensorsW8A8Int8(strategy, True, symmetric)
    layer = torch.nn.Module()
    scheme.create_weights(layer, list(widths), k, torch.bfloat16, lambda *a: None)
    layer.weight.data.fill_(1)
    layer.weight_scale.data.fill_(0.125)
    return scheme, layer


class TestStaticInt8Loading(CustomTestCase):
    def test_fused_calibration_and_postprocessing(self):
        for strategy, symmetric, widths in product(
            ("channel", "tensor"), (True, False), ((8,), (4, 4), (4, 2, 2))
        ):
            with self.subTest(strategy=strategy, symmetric=symmetric, widths=widths):
                scheme, layer = make_layer(strategy, widths, symmetric)
                scales = torch.arange(1, len(widths) + 1) * 0.125
                zps = torch.arange(len(widths), dtype=torch.int8) - 1
                for i in range(len(widths)):
                    load = (
                        "load_qkv_weight"
                        if len(widths) == 3
                        else "load_merged_column_weight"
                    )
                    shard = "qkv"[i] if len(widths) == 3 else i
                    getattr(layer.input_scale, load)(scales[i : i + 1], shard_id=shard)
                    if not symmetric:
                        getattr(layer.input_zero_point, load)(
                            zps[i : i + 1], shard_id=shard
                        )
                torch.testing.assert_close(layer.input_scale, scales)
                if not symmetric:
                    torch.testing.assert_close(layer.input_zero_point, zps)
                if strategy == "tensor":
                    continue  # Requantization invokes the CUDA FP8 helper.
                scheme.process_weights_after_loading(layer)
                if symmetric:
                    torch.testing.assert_close(layer.input_scale, scales.max())
                else:
                    lo, hi = (
                        (scales * (-128 - zps.int())).min(),
                        (scales * (127 - zps.int())).max(),
                    )
                    scale = (hi - lo) / 255
                    torch.testing.assert_close(layer.input_scale, scale)
                    torch.testing.assert_close(
                        layer.input_zero_point, (-128 - lo / scale).int()
                    )

    def test_npu_fused_scale_conversion(self):
        # The NPU conversion keeps each shard's exported int8 codes and
        # broadcasts the shard weight scale to its output channels.
        widths = [4, 2, 2]
        for strategy in ("channel", "tensor"):
            with self.subTest(strategy=strategy):
                scheme = NPUCompressedTensorsW8A8Int8(strategy, True, True)
                layer = torch.nn.Module()
                scheme.create_weights(layer, widths, 8, torch.bfloat16, lambda *a: None)
                weight = torch.arange(64, dtype=torch.int8).reshape(8, 8)
                layer.weight.data.copy_(weight)
                layer.input_scale.data.copy_(torch.tensor([0.125, 0.375, 0.25]))
                if strategy == "tensor":
                    layer.weight_scale.data.copy_(torch.tensor([0.25, 0.5, 1.0]))
                    expected_scales = torch.tensor([0.25] * 4 + [0.5] * 2 + [1.0] * 2)
                else:
                    expected_scales = torch.arange(1, 9).float() / 16
                    layer.weight_scale.data.copy_(expected_scales.reshape(-1, 1))
                with patch(
                    "sglang.srt.hardware_backend.npu.quantization.linear_method_npu.npu_format_cast",
                    side_effect=lambda tensor: tensor,
                ):
                    scheme.process_weights_after_loading(layer)
                torch.testing.assert_close(layer.weight, weight.T.contiguous())
                torch.testing.assert_close(layer.weight_scale, expected_scales)
                torch.testing.assert_close(layer.input_scale, torch.tensor(0.375))
                torch.testing.assert_close(
                    layer.aclnn_input_scale, torch.full((8,), 0.375)
                )
                torch.testing.assert_close(
                    layer.aclnn_input_offset, torch.zeros(8, dtype=torch.int32)
                )

                # Model the NPU's fused integer matmul and independently
                # calculate each output shard with the shared input scale.
                x = torch.tensor([[0.125, -0.125, 1.0, -1.0, 48.0, -50.0, 0.75, 3.0]])
                bias = torch.arange(8).float() / 32
                qx = torch.round(x / layer.input_scale).clamp(-128, 127).long()
                actual = (qx @ layer.weight.long()).float() * (
                    layer.input_scale * layer.weight_scale
                ) + bias
                reference = []
                offset = 0
                for width in widths:
                    shard = slice(offset, offset + width)
                    reference.append(
                        (qx @ weight[shard].long().T).float()
                        * (0.375 * expected_scales[shard])
                        + bias[shard]
                    )
                    offset += width
                torch.testing.assert_close(actual, torch.cat(reference, dim=-1))

                # max() intentionally changes rounding for shards originally
                # calibrated with a smaller input scale.
                qx_first = torch.round(x / 0.125).clamp(-128, 127).long()
                separate_first = (qx_first @ weight[:4].long().T).float() * (
                    0.125 * expected_scales[:4]
                ) + bias[:4]
                self.assertFalse(torch.equal(actual[:, :4], separate_first))


if __name__ == "__main__":
    unittest.main()
