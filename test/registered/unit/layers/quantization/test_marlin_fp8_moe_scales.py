"""The Marlin W8A16 repack takes FP32 [E, K groups, N] scales; these adapters
fold every FP8 checkpoint scale layout into that contract, and the conversion
to the kernel's activation-dtype scales has a stated, checked domain."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    fp8_marlin_scale_compensation,
    marlin_fp8_moe_group_size,
    marlin_fp8_scales_at_group,
    marlin_fp8_scales_to_dtype,
    marlin_moe_fp8_block_scales,
    marlin_moe_fp8_channel_scales,
    marlin_moe_fp8_tensor_scales,
    marlin_moe_mxfp8_scales,
    prepare_moe_fp8_layer_for_marlin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMarlinFp8MoeScales(CustomTestCase):
    def test_block_scales_repeat_over_their_output_columns(self):
        """[E, ceil(N / block_n), K / block_k] becomes [E, K / block_k, N]: column
        n of K group g reads the block scale (n // block_n, g), and a partial
        last block row covers only the columns the layer has."""
        experts, block_n, block_k, size_n, size_k = 2, 128, 128, 300, 512
        scales = torch.arange(experts * 3 * 4, dtype=torch.float32).reshape(
            experts, 3, 4
        )
        groups = marlin_moe_fp8_block_scales(scales, size_n=size_n, block_n=block_n)
        self.assertEqual(tuple(groups.shape), (experts, size_k // block_k, size_n))
        n = torch.arange(size_n)
        for e in range(experts):
            for g in range(size_k // block_k):
                torch.testing.assert_close(groups[e, g], scales[e, n // block_n, g])

    def test_mxfp8_scales_are_the_powers_of_two_of_the_e8m0_bytes(self):
        """E8M0 byte b is the scale 2 ** (b - 127); the Marlin groups transpose
        the checkpoint's [E, N, K / 32] to [E, K / 32, N], from raw bytes or the
        e8m0 view alike."""
        raw = torch.tensor(
            [[[127, 126, 130], [0, 254, 120]]], dtype=torch.uint8
        )  # [1, N=2, K/32=3]
        expected = torch.tensor(
            [[[1.0, 2.0**-127], [0.5, 2.0**127], [8.0, 2.0**-7]]]
        )  # [1, 3, 2]
        for scales in (raw, raw.view(torch.float8_e8m0fnu)):
            groups = marlin_moe_mxfp8_scales(scales)
            self.assertEqual(groups.dtype, torch.float32)
            torch.testing.assert_close(groups, expected, rtol=0, atol=0)

    def test_channel_scales_become_one_group(self):
        scales = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5, 1)
        groups = marlin_moe_fp8_channel_scales(scales)
        self.assertEqual(tuple(groups.shape), (2, 1, 5))
        torch.testing.assert_close(groups[:, 0], scales[:, :, 0])

    def test_tensor_scales_keep_each_shard_its_own_scale(self):
        """The loader's [E, 2] gate / up scales cover their own halves of the
        gate-up columns; an [E] scale covers every column."""
        gate_up = torch.tensor([[1.0, 3.0], [2.0, 5.0]])
        groups = marlin_moe_fp8_tensor_scales(gate_up, size_n=8)
        self.assertEqual(tuple(groups.shape), (2, 1, 8))
        torch.testing.assert_close(
            groups[:, 0], torch.tensor([[1.0] * 4 + [3.0] * 4, [2.0] * 4 + [5.0] * 4])
        )
        down = marlin_moe_fp8_tensor_scales(torch.tensor([0.5, 0.25]), size_n=6)
        torch.testing.assert_close(down[:, 0], torch.tensor([[0.5] * 6, [0.25] * 6]))
        with self.assertRaisesRegex(ValueError, "equal scale shards"):
            marlin_moe_fp8_tensor_scales(gate_up, size_n=7)

    def test_single_group_scales_are_served_as_repeated_groups(self):
        """The kernel reads one scale group over K as channel-wise and, for E4M3
        weights, scales the FP32 accumulators after the K loop, where products
        that carry the 2**-120 dequantization shift have already underflowed;
        grouped scales are applied before the MMA. So channel-wise and
        per-tensor scales are repeated over 128-wide groups (32-wide when a
        stage's K would hold a single 128-wide group), block scales whose stage
        holds a single group are split the same way, MXFP8 stays at 32, and
        64-wide groups are refused as before."""
        channel13, channel2 = torch.rand(2, 1, 64), torch.rand(2, 1, 2048)
        self.assertEqual(
            marlin_fp8_moe_group_size(
                channel13, channel2, hidden_size=2048, intermediate_size=512
            ),
            128,
        )
        self.assertEqual(
            marlin_fp8_moe_group_size(
                channel13, channel2, hidden_size=256, intermediate_size=128
            ),
            32,
        )
        # 64-aligned intermediates the channel-wise path admitted keep serving,
        # through the 32-wide kernels.
        for inter, group in ((320, 32), (448, 32), (256, 128), (384, 128)):
            with self.subTest(intermediate=inter):
                self.assertEqual(
                    marlin_fp8_moe_group_size(
                        channel13, channel2, hidden_size=512, intermediate_size=inter
                    ),
                    group,
                )
        block13, block2 = torch.rand(2, 2, 256), torch.rand(2, 1, 256)
        self.assertEqual(
            marlin_fp8_moe_group_size(
                block13, block2, hidden_size=256, intermediate_size=128
            ),
            32,
        )
        self.assertEqual(
            marlin_fp8_moe_group_size(
                torch.rand(2, 8, 256),
                torch.rand(2, 4, 256),
                hidden_size=256,
                intermediate_size=128,
            ),
            32,
        )
        with self.assertRaisesRegex(NotImplementedError, "32- or 128-wide"):
            marlin_fp8_moe_group_size(
                torch.rand(2, 4, 256),
                torch.rand(2, 2, 256),
                hidden_size=256,
                intermediate_size=128,
            )
        repeated = marlin_fp8_scales_at_group(channel13, size_k=2048, group_size=128)
        self.assertEqual(tuple(repeated.shape), (2, 16, 64))
        for g in range(16):
            torch.testing.assert_close(repeated[:, g], channel13[:, 0])
        split = marlin_fp8_scales_at_group(block13, size_k=256, group_size=32)
        self.assertEqual(tuple(split.shape), (2, 8, 256))
        for g in range(8):
            torch.testing.assert_close(split[:, g], block13[:, g // 4])
        self.assertIs(
            marlin_fp8_scales_at_group(block13, size_k=256, group_size=128), block13
        )


def _e8m0(byte: int) -> float:
    return 2.0 ** (byte - 127)


def _kernel_dequant(code: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    """What the FP8 Marlin kernel computes for an E4M3 byte: its bits moved
    into the activation dtype's exponent field (the value times 2**-bias, where
    the dtype's bias minus E4M3's is 120 for BF16 and 8 for FP16) times the
    compensated scale, rounded once to the dtype; independent of the helpers."""
    bits = code.view(torch.uint8).to(torch.int32)
    sign, exponent, mantissa = bits >> 7, (bits >> 3) & 0xF, bits & 0x7
    if dtype == torch.bfloat16:
        shifted = (sign << 15) | (exponent << 7) | (mantissa << 4)
        value = shifted.to(torch.int16).view(torch.bfloat16)
    else:
        shifted = (sign << 15) | (exponent << 10) | (mantissa << 7)
        value = shifted.to(torch.int16).view(torch.float16)
    return value * scale.to(dtype)


class TestMarlinFp8ScaleDomain(CustomTestCase):
    """FP32 scales reach the kernel as activation-dtype scales with the exponent
    bias folded in; the fold happens in FP32, and the representable range is
    checked rather than clamped."""

    def test_compensation_matches_the_kernel_dequantization(self):
        """For every E4M3 byte and representative scales (MXFP8 exponents from
        2**-127 to 2**7, FP32 scales down to 2**-140), the kernel model with the
        compensated scale equals the FP32 dequantization rounded to the dtype
        within one rounding of the scale and one of the product."""
        codes = torch.arange(256, dtype=torch.int32)
        codes = codes[(codes & 0x7F) != 0x7F]  # E4M3 has no infinities; skip NaN
        values = codes.to(torch.uint8).view(torch.float8_e4m3fn).float()
        for dtype in (torch.bfloat16, torch.float16):
            exponents = [0, 60, 95, 120, 127, 130, 134]
            scales = [_e8m0(byte) for byte in exponents] + [2.0**-140, 1e-3, 0.0375]
            if dtype == torch.float16:
                scales = [scale for scale in scales if scale >= 2.0**-22]
            for scale in scales:
                with self.subTest(dtype=dtype, scale=scale):
                    scale_t = torch.full((1,), scale, dtype=torch.float32)
                    compensated = marlin_fp8_scales_to_dtype(scale_t, dtype)
                    self.assertEqual(compensated.dtype, dtype)
                    got = _kernel_dequant(
                        codes.to(torch.uint8), compensated.expand(values.numel()), dtype
                    ).float()
                    reference = (values * scale).to(dtype).float()
                    # Three unit roundoffs (2**-8 BF16, 2**-11 FP16): the scale's,
                    # the kernel product's and the reference product's; or one
                    # subnormal quantum where the product is subnormal.
                    torch.testing.assert_close(
                        got,
                        reference,
                        rtol=3 * 2.0 ** -(8 if dtype == torch.bfloat16 else 11),
                        atol=2.0 ** (-133 if dtype == torch.bfloat16 else -24),
                    )

    def test_fold_happens_before_the_cast(self):
        """A scale below BF16's smallest subnormal (2**-133) must still arrive:
        2**-140 times the compensation is 2**-20, and E4M3's 448 times 2**-140
        is a nonzero BF16 weight."""
        compensated = marlin_fp8_scales_to_dtype(
            torch.tensor([2.0**-140]), torch.bfloat16
        )
        self.assertEqual(float(compensated), 2.0**-20)
        self.assertEqual(float(torch.tensor([2.0**-140]).to(torch.bfloat16)), 0.0)
        weight = _kernel_dequant(
            torch.tensor([126], dtype=torch.uint8), compensated, torch.bfloat16
        )
        self.assertGreater(float(weight), 0.0)
        self.assertEqual(
            float(weight), float(torch.tensor(448 * 2.0**-140).to(torch.bfloat16))
        )

    def test_compensation_factors(self):
        self.assertEqual(fp8_marlin_scale_compensation(torch.bfloat16), 2.0**120)
        self.assertEqual(fp8_marlin_scale_compensation(torch.float16), 2.0**8)
        with self.assertRaisesRegex(ValueError, "BF16 or FP16"):
            fp8_marlin_scale_compensation(torch.float32)

    def test_domain_boundaries(self):
        """Every admitted scale keeps the dtype's normal precision. Power-of-two
        MXFP8 bytes: both dtypes admit 134 (2**7) and refuse 135 (2**8, overflow
        after the fold); FP16 admits byte 105 (2**-22, the smallest scale whose
        compensated value is a normal FP16) and refuses 104; every nonzero
        byte survives in BF16. Arbitrary FP32 scales at the same boundaries:
        BF16 admits 255 (255 * 2**120 is its largest finite value) and refuses
        255.5; FP16 admits 255.875 (65504 / 256) and refuses 255.9; FP16 admits
        2**-22 and 1.5 * 2**-22 and refuses 0.75 * 2**-22 and 3 * 2**-34 (whose
        compensated values are subnormal, the latter rounding to 2**-24 with a
        third of relative error); the FP32 minimum 2**-149 and 3 * 2**-140 are
        normal in BF16 after the fold. NaN, Inf and negative scales are refused."""
        cases = {
            (torch.bfloat16, 134): True,
            (torch.bfloat16, 135): False,
            (torch.bfloat16, 0): True,
            (torch.float16, 134): True,
            (torch.float16, 135): False,
            (torch.float16, 105): True,
            (torch.float16, 104): False,
        }
        for (dtype, byte), admitted in cases.items():
            with self.subTest(dtype=dtype, byte=byte):
                scales = marlin_moe_mxfp8_scales(
                    torch.full((1, 2, 1), byte, dtype=torch.uint8)
                )
                if admitted:
                    out = marlin_fp8_scales_to_dtype(scales, dtype)
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    self.assertTrue(bool((out != 0).all()))
                else:
                    with self.assertRaises(ValueError):
                        marlin_fp8_scales_to_dtype(scales, dtype)
        for bad in (float("nan"), float("inf"), -1.0):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                marlin_fp8_scales_to_dtype(torch.tensor([bad]), torch.bfloat16)
        arbitrary = {
            (torch.bfloat16, 255.0): True,
            (torch.bfloat16, 255.5): False,
            (torch.bfloat16, 2.0**-149): True,
            (torch.bfloat16, 3 * 2.0**-140): True,
            (torch.float16, 255.875): True,
            (torch.float16, 255.9): False,
            (torch.float16, 2.0**-22): True,
            (torch.float16, 1.5 * 2.0**-22): True,
            (torch.float16, 0.75 * 2.0**-22): False,
            (torch.float16, 3 * 2.0**-34): False,
        }
        for (dtype, scale), admitted in arbitrary.items():
            with self.subTest(dtype=dtype, scale=scale):
                if admitted:
                    out = marlin_fp8_scales_to_dtype(torch.tensor([scale]), dtype)
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    self.assertGreaterEqual(abs(float(out)), torch.finfo(dtype).tiny)
                else:
                    with self.assertRaisesRegex(ValueError, "range"):
                        marlin_fp8_scales_to_dtype(torch.tensor([scale]), dtype)
        # The FP32 minimum survives BF16 (2**-149 * 2**120 = 2**-29, a normal BF16).
        out = marlin_fp8_scales_to_dtype(torch.tensor([2.0**-149]), torch.bfloat16)
        self.assertEqual(float(out), 2.0**-29)

    def test_fp16_dequantized_weights_must_fit_fp16(self):
        """The grouped kernel multiplies code * scale in the activation dtype
        before the MMA, with the scale as the kernel holds it (the compensated
        scale rounded to FP16). With FP16 activations, E4M3 codes of 448 under a
        scale of 255 dequantize to 114240, which is FP16 infinity, although the
        GEMM result can be finite; the repack refuses that stage before any
        parameter moves, with the scale admitted on its own (255 <= 255.875).
        BF16 activations skip the check: 448 * 255 is far inside BF16's range."""
        experts, hidden, inter = 2, 256, 128
        codes = torch.nn.Parameter(
            torch.full((experts, 2 * inter, hidden), 448.0).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        w2 = torch.nn.Parameter(
            torch.ones((experts, hidden, inter)).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer = SimpleNamespace(
            w13_weight=codes,
            w2_weight=w2,
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            moe_runner_config=SimpleNamespace(
                is_gated=True, activation="silu", apply_router_weight_on_input=False
            ),
        )
        huge = torch.full((experts, 1, 2 * inter), 255.0)
        ones = torch.ones((experts, 1, hidden))
        with self.assertRaisesRegex(ValueError, "beyond the FP16 range"):
            prepare_moe_fp8_layer_for_marlin(
                layer, w13_scales=huge, w2_scales=ones, params_dtype=torch.float16
            )
        self.assertIs(layer.w13_weight, codes)
        self.assertFalse(hasattr(layer, "w13_weight_scale"))
        # The scale itself passes the scale check; only the product fails.
        marlin_fp8_scales_to_dtype(huge, torch.float16)

    def test_rejection_keeps_both_stages_untouched(self):
        """A scale outside the domain in either stage refuses the layer before
        any parameter is replaced: w13 admitted, w2 overflowing, nothing moves."""
        experts, hidden, inter = 2, 256, 128
        w13 = torch.nn.Parameter(
            torch.zeros((experts, 2 * inter, hidden), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        w2 = torch.nn.Parameter(
            torch.zeros((experts, hidden, inter), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer = SimpleNamespace(
            w13_weight=w13,
            w2_weight=w2,
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            moe_runner_config=SimpleNamespace(
                is_gated=True, activation="silu", apply_router_weight_on_input=False
            ),
        )
        good = torch.full((experts, 1, 2 * inter), 0.01)
        overflow = torch.full((experts, 1, hidden), 2.0**8)
        with self.assertRaisesRegex(ValueError, "w2 scales exceed"):
            prepare_moe_fp8_layer_for_marlin(
                layer, w13_scales=good, w2_scales=overflow, params_dtype=torch.bfloat16
            )
        self.assertIs(layer.w13_weight, w13)
        self.assertIs(layer.w2_weight, w2)
        self.assertFalse(hasattr(layer, "w13_weight_scale"))


if __name__ == "__main__":
    unittest.main()
