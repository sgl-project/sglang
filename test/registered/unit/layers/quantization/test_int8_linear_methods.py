"""Numerics for the INT8 dense-linear methods.

Real layer path vs a dequantized-reference matmul, in two formats:
channel W8A8 (W8A8Int8LinearMethod, per-channel weight scale + dynamic
per-token int8 activations) and blockwise (BlockInt8LinearMethod,
(128, 128) block weight scale). Both channel W8A8 methods
(W8A8Int8LinearMethod and the compressed-tensors scheme) are also checked
against an exact INT32-accumulator reference on HIP.
"""

import unittest

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Int8,
)
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8Config
from sglang.srt.utils import get_device_sm, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.layer_ut_utils import (
    assert_output_close,
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="base-b", runner_config="1-gpu-small-amd")

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
        layer = make_tp1_column_parallel_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_int8, scale_inv, w_dequant = _quantize_int8_block(w)
        load_linear_weights(layer, weight=w_int8, weight_scale_inv=scale_inv)
        return layer, w_dequant

    def test_block(self):
        self._check(BLOCK_SHAPES, self._build_layer)


@unittest.skipUnless(is_hip(), "checks the HIP dispatch to triton_scaled_mm")
class TestW8A8Int8LinearInt32Reference(CustomTestCase):
    """Both channel W8A8 INT8 linear methods against an exact INT32 GEMM.

    Guards the HIP dispatch of these methods (sgl-project/sglang#50386: on
    ROCm the compressed-tensors scheme raised NameError on int8_scaled_mm),
    and the epilogue order: the bias is added before the single rounding to
    the output dtype.
    """

    SHAPES = [(1, 1152, 896), (7, 896, 896), (33, 4864, 896), (300, 896, 4864)]

    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    @staticmethod
    def _apply_w8a8_int8_linear(w_int8, w_scale, x, bias):
        layer = make_tp1_column_parallel_linear(
            W8A8Int8Config({}), w_int8.shape[0], w_int8.shape[1]
        )
        load_linear_weights(layer, weight=w_int8, weight_scale=w_scale)
        layer.quant_method.process_weights_after_loading(layer)
        return layer.quant_method.apply(layer, x, bias)

    @staticmethod
    def _apply_compressed_tensors(w_int8, w_scale, x, bias):
        scheme = CompressedTensorsW8A8Int8(
            strategy=QuantizationStrategy.CHANNEL,
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        layer = torch.nn.Module()
        scheme.create_weights(
            layer=layer,
            output_partition_sizes=[w_int8.shape[0]],
            input_size_per_partition=w_int8.shape[1],
            params_dtype=x.dtype,
            weight_loader=None,
        )
        layer.weight.data = w_int8.clone()
        layer.weight_scale.data = w_scale.clone()
        scheme.process_weights_after_loading(layer)
        return scheme.apply_weights(layer, x, bias)

    def _methods(self):
        return {
            "W8A8Int8LinearMethod": self._apply_w8a8_int8_linear,
            "CompressedTensorsW8A8Int8": self._apply_compressed_tensors,
        }

    def test_empty_batch(self):
        n, k = 1152, 896
        w_int8 = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        w_scale = 1e-4 + 1e-3 * torch.rand((n, 1), device="cuda", dtype=torch.float32)
        x = torch.empty((0, k), device="cuda", dtype=torch.bfloat16)
        for name, apply in self._methods().items():
            with self.subTest(method=name):
                out = apply(w_int8, w_scale, x, None)
                self.assertEqual(tuple(out.shape), (0, n))
                self.assertEqual(out.dtype, torch.bfloat16)

    def test_matches_int32_reference(self):
        for name, apply in self._methods().items():
            for m, n, k in self.SHAPES:
                for with_bias in (False, True):
                    with self.subTest(method=name, shape=(m, n, k), bias=with_bias):
                        torch.manual_seed(0)
                        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                        w_int8 = torch.randint(
                            -127, 128, (n, k), device="cuda", dtype=torch.int8
                        )
                        w_scale = 1e-4 + 1e-3 * torch.rand(
                            (n, 1), device="cuda", dtype=torch.float32
                        )
                        bias = (
                            4 * torch.randn((n,), device="cuda", dtype=torch.bfloat16)
                            if with_bias
                            else None
                        )

                        out = apply(w_int8, w_scale, x, bias).double()

                        # fp64 holds the INT32 accumulator exactly (|acc| < 2^53).
                        x_q, x_scale = per_token_quant_int8(x)
                        ref = (x_q.double() @ w_int8.double().t()) * x_scale.double()
                        ref = ref * w_scale.double().t()
                        if bias is not None:
                            ref = ref + bias.double()
                        # One bf16 ulp (2^-7 relative) at the reference, plus
                        # the smallest bf16 subnormal for outputs near zero.
                        bound = ref.abs() * 2.0**-7 + 2.0**-133
                        excess = (out - ref).abs() - bound
                        self.assertEqual(tuple(out.shape), (m, n))
                        self.assertLessEqual(
                            excess.max().item(),
                            0.0,
                            f"{(excess > 0).float().mean().item():.4f} of outputs "
                            "are more than 1 bf16 ulp from the INT32 reference",
                        )


if __name__ == "__main__":
    unittest.main()
