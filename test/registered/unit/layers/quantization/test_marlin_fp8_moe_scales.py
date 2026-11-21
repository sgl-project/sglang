"""FP8 checkpoint scales in Marlin's FP32 [E, K groups, N] repack layout."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    marlin_fp8_moe_group_size,
    marlin_fp8_scales_at_group,
    marlin_fp8_scales_to_bf16,
    marlin_moe_fp8_block_scales,
    marlin_moe_fp8_channel_scales,
    marlin_moe_fp8_tensor_scales,
    marlin_moe_mxfp8_scales,
    prepare_moe_fp8_layer_for_marlin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="weekly", runner_config="cpu")


class TestMarlinFp8MoeScales(CustomTestCase):
    def test_block_scales_repeat_over_their_output_columns(self):
        """Expand each scale across its output block, including a partial last block."""
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
        """E8M0 byte b supplies 2**(b - 127) in Marlin's [E, K/32, N] order."""
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
        """Each [E, 2] gate/up scale covers one half of the output columns."""
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
        """Force grouped scaling before accumulation, including non-128-aligned K."""
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
        # Intermediates aligned to 64 but not 128 use 32-wide groups.
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


def _kernel_dequant(code: torch.Tensor, scale: torch.Tensor):
    """Apply Marlin's E4M3-to-BF16 bit shift (2**-120) and compensated scale."""
    bits = code.view(torch.uint8).to(torch.int32)
    sign, exponent, mantissa = bits >> 7, (bits >> 3) & 0xF, bits & 0x7
    shifted = (sign << 15) | (exponent << 7) | (mantissa << 4)
    value = shifted.to(torch.int16).view(torch.bfloat16)
    return value * scale.to(torch.bfloat16)


class TestMarlinFp8ScaleDomain(CustomTestCase):
    """Pin scale rounding and overflow after folding the exponent bias."""

    def test_compensation_matches_the_kernel_dequantization(self):
        """Compare Marlin's bit-shifted E4M3 dequantization with a BF16 reference."""
        codes = torch.arange(256, dtype=torch.int32)
        codes = codes[(codes & 0x7F) != 0x7F]  # E4M3 has no infinities; skip NaN
        values = codes.to(torch.uint8).view(torch.float8_e4m3fn).float()
        exponents = [0, 60, 95, 120, 127, 130, 134]
        scales = [_e8m0(byte) for byte in exponents] + [2.0**-140, 1e-3, -0.0375]
        for scale in scales:
            with self.subTest(scale=scale):
                scale_t = torch.full((1,), scale, dtype=torch.float32)
                compensated = marlin_fp8_scales_to_bf16(scale_t)
                self.assertEqual(compensated.dtype, torch.bfloat16)
                got = _kernel_dequant(
                    codes.to(torch.uint8), compensated.expand(values.numel())
                ).float()
                reference = (values * scale).to(torch.bfloat16).float()
                # Three BF16 roundoffs (scale and both products) plus one subnormal quantum.
                torch.testing.assert_close(
                    got, reference, rtol=3 * 2.0**-8, atol=2.0**-133
                )

    def test_fold_happens_before_the_cast(self):
        """Preserve a scale that would underflow before exponent compensation."""
        compensated = marlin_fp8_scales_to_bf16(torch.tensor([2.0**-140]))
        self.assertEqual(float(compensated), 2.0**-20)
        self.assertEqual(float(torch.tensor([2.0**-140]).to(torch.bfloat16)), 0.0)
        weight = _kernel_dequant(torch.tensor([126], dtype=torch.uint8), compensated)
        self.assertGreater(float(weight), 0.0)
        self.assertEqual(
            float(weight), float(torch.tensor(448 * 2.0**-140).to(torch.bfloat16))
        )

    def test_domain_boundaries(self):
        """Check the smallest FP32 scales and the compensated BF16 overflow boundary."""
        for byte, admitted in ((134, True), (135, False), (0, True)):
            with self.subTest(byte=byte):
                scales = marlin_moe_mxfp8_scales(
                    torch.full((1, 2, 1), byte, dtype=torch.uint8)
                )
                if admitted:
                    out = marlin_fp8_scales_to_bf16(scales)
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    self.assertTrue(bool((out != 0).all()))
                else:
                    with self.assertRaises(ValueError):
                        marlin_fp8_scales_to_bf16(scales)
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                marlin_fp8_scales_to_bf16(torch.tensor([bad]))
        arbitrary = {
            255.0: True,
            255.5: False,
            -255.5: False,
            2.0**-149: True,
            3 * 2.0**-140: True,
        }
        for scale, admitted in arbitrary.items():
            with self.subTest(scale=scale):
                if admitted:
                    out = marlin_fp8_scales_to_bf16(torch.tensor([scale]))
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    self.assertGreaterEqual(
                        abs(float(out)), torch.finfo(torch.bfloat16).tiny
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "range"):
                        marlin_fp8_scales_to_bf16(torch.tensor([scale]))
        # The FP32 minimum survives BF16 (2**-149 * 2**120 = 2**-29, a normal BF16).
        out = marlin_fp8_scales_to_bf16(torch.tensor([2.0**-149]))
        self.assertEqual(float(out), 2.0**-29)

    def test_rejection_keeps_both_stages_untouched(self):
        """An invalid down-projection scale leaves both sets of weights intact."""
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
            prepare_moe_fp8_layer_for_marlin(layer, w13_scales=good, w2_scales=overflow)
        self.assertIs(layer.w13_weight, w13)
        self.assertIs(layer.w2_weight, w2)
        self.assertFalse(hasattr(layer, "w13_weight_scale"))


if __name__ == "__main__":
    unittest.main()
