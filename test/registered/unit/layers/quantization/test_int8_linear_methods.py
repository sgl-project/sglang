"""Numerics for the INT8 dense-linear methods.

Real layer path vs a dequantized-reference matmul, in two formats:
channel W8A8 (W8A8Int8LinearMethod, per-channel weight scale + dynamic
per-token int8 activations) and blockwise (BlockInt8LinearMethod,
(128, 128) block weight scale).
"""

import unittest

import torch

from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    assert_output_close,
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

INT8_MAX = 127.0

# (M, N, K); channel int8 has no block-alignment constraints.
CHANNEL_SHAPES = [
    (64, 512, 512),
    (5, 160, 336),
    (128, 1024, 1024),
]

# (M, N, K), N and K multiples of the (128, 128) weight block.
BLOCK_SHAPES = [
    (64, 512, 512),
    (5, 384, 896),
    (128, 1024, 1024),
]


def _quantize_int8_channel(w: torch.Tensor):
    """Per-output-channel symmetric int8; returns checkpoint-format
    (w_int8 [N, K], scale fp32 [N, 1]) and the dequant reference."""
    amax = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / INT8_MAX
    w_int8 = torch.round(w.float() / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    w_dequant = w_int8.float() * scale
    return w_int8, scale, w_dequant


def _quantize_int8_block(w: torch.Tensor, block: int = 128):
    """Per (block, block) tile symmetric int8; returns checkpoint-format
    (w_int8 [N, K], scale_inv fp32 [N/block, K/block]) and the dequant reference."""
    n, k = w.shape
    tiles = w.float().reshape(n // block, block, k // block, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / INT8_MAX
    w_int8 = (
        torch.round(tiles / scale[:, None, :, None])
        .clamp(-INT8_MAX, INT8_MAX)
        .to(torch.int8)
    )
    w_dequant = (w_int8.float() * scale[:, None, :, None]).reshape(n, k)
    return w_int8.reshape(n, k), scale, w_dequant


class _Int8LinearCheck(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _check(self, shapes, build_layer):
        torch.manual_seed(7)
        for m, n, k in shapes:
            with self.subTest(shape=(m, n, k)):
                layer, w_dequant = build_layer(n, k)
                layer.quant_method.process_weights_after_loading(layer)

                x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
                out, _ = layer(x)

                ref = x.float() @ w_dequant.T
                # atol absorbs the dynamic per-token int8 activation quant,
                # which the reference does not mirror.
                assert_output_close(self, out, ref, rtol=5e-2, atol=1e-1)


@unittest.skipIf(
    get_device_sm() >= 100, "sgl-kernel int8_scaled_mm has no SM100+ kernel"
)
class TestW8A8Int8Linear(_Int8LinearCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        layer = make_tp1_column_parallel_linear(W8A8Int8Config({}), n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_int8, scale, w_dequant = _quantize_int8_channel(w)
        load_linear_weights(layer, weight=w_int8, weight_scale=scale)
        return layer, w_dequant

    def test_channel(self):
        self._check(CHANNEL_SHAPES, self._build_layer)


class TestBlockInt8Linear(_Int8LinearCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = BlockInt8Config(
            is_checkpoint_int8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        # create_weights reads get_parallel().tp_size, not the layer's argument.
        with get_parallel().override(tp_size=1, tp_rank=0):
            layer = make_tp1_column_parallel_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_int8, scale_inv, w_dequant = _quantize_int8_block(w)
        load_linear_weights(layer, weight=w_int8, weight_scale_inv=scale_inv)
        return layer, w_dequant

    def test_block(self):
        self._check(BLOCK_SHAPES, self._build_layer)


if __name__ == "__main__":
    unittest.main()
