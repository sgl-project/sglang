"""Fused static calibration must load before backend-specific processing."""

import unittest
from itertools import product

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
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

    def test_tp_weight_and_scale_shards(self):
        for strategy, mode in product(
            ("channel", "tensor"), ("column", "row", "qkv", "merged")
        ):
            widths = (
                (4, 2, 2) if mode == "qkv" else (4, 4) if mode == "merged" else (8,)
            )
            full_k = 16 if mode == "row" else 8
            full_widths = [
                w if mode == "row" or (mode == "qkv" and i > 0) else 2 * w
                for i, w in enumerate(widths)
            ]
            weights = [
                torch.arange(w * full_k).reshape(w, full_k).to(torch.int8)
                for w in full_widths
            ]
            for rank in (0, 1):
                with self.subTest(strategy=strategy, mode=mode, rank=rank):
                    _, layer = make_layer(strategy, widths, k=8)
                    offset = 0
                    for i, (width, weight) in enumerate(zip(widths, weights)):
                        scale = (
                            torch.arange(1, len(weight) + 1).float().reshape(-1, 1)
                            if strategy == "channel"
                            else torch.tensor([0.125 * (i + 1)])
                        )
                        kwargs = dict(tp_rank=rank)
                        method = (
                            f"load_{mode}_weight"
                            if mode == "qkv"
                            else f"load_{mode}_parallel_weight"
                            if mode in ("row", "column")
                            else "load_merged_column_weight"
                        )
                        if mode in ("qkv", "merged"):
                            kwargs.update(
                                shard_id="qkv"[i] if mode == "qkv" else i,
                                shard_offset=offset,
                                shard_size=width,
                                num_heads=2 if i else 1,
                            )
                        getattr(layer.weight, method)(weight, **kwargs)
                        if mode == "row":
                            expected = weight[:, rank * 8 : (rank + 1) * 8]
                            layer.weight_scale.load_row_parallel_weight(scale)
                        else:
                            start = 0 if mode == "qkv" and i else rank * width
                            expected = weight[start : start + width]
                            getattr(layer.weight_scale, method)(scale, **kwargs)
                        torch.testing.assert_close(
                            layer.weight[offset : offset + width], expected
                        )
                        if strategy == "channel":
                            torch.testing.assert_close(
                                layer.weight_scale[offset : offset + width],
                                scale
                                if mode == "row"
                                else scale[start : start + width],
                            )
                        else:
                            torch.testing.assert_close(layer.weight_scale[i], scale[0])
                        offset += width


if __name__ == "__main__":
    unittest.main()
