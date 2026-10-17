"""Unit tests for the UltraQuant 4-bit KV cache format - CPU-only."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.ultraquant_tensor import (
    E2M1_LEVELS_SORTED,
    E2M1_MIDPOINTS,
    E2M1_VALUES,
    E2M1_ZERO_LEVEL_INDEX,
    GROUP_SIZE,
    UE8M0_BIAS,
    UE8M0_MAX_EXP,
    UE8M0_MIN_EXP,
    UltraQuantKVQuantizeUtil,
    code_bytes,
    hadamard_matrix,
    n_groups,
    pack_nibbles,
    sorted_index_to_e2m1_bits,
    ue8m0_decode,
    ue8m0_encode,
    unpack_nibbles,
)
from sglang.test.test_utils import CustomTestCase

HEAD_DIMS = (64, 128, 256)
CPU = torch.device("cpu")


class TestUltraQuantFormat(CustomTestCase):
    def test_e2m1_tables_are_self_consistent(self):
        self.assertEqual(len(E2M1_VALUES), 16)
        self.assertEqual(
            len(E2M1_LEVELS_SORTED), 15, "+0.0 and -0.0 collapse to one level"
        )
        self.assertEqual(list(E2M1_LEVELS_SORTED), sorted(E2M1_LEVELS_SORTED))
        self.assertEqual(len(E2M1_MIDPOINTS), len(E2M1_LEVELS_SORTED) - 1)

        # The closed-form remap must reproduce the ascending levels exactly, which
        # is what lets the kernels skip a lookup table.
        bits = sorted_index_to_e2m1_bits(torch.arange(len(E2M1_LEVELS_SORTED)))
        decoded = [E2M1_VALUES[int(b)] for b in bits]
        self.assertEqual(decoded, list(E2M1_LEVELS_SORTED))

    def test_hadamard_is_orthonormal(self):
        for head_dim in HEAD_DIMS:
            with self.subTest(head_dim=head_dim):
                h = hadamard_matrix(head_dim, CPU)
                torch.testing.assert_close(
                    h @ h.T, torch.eye(head_dim), rtol=0, atol=1e-6
                )

    def test_hadamard_rejects_non_power_of_two(self):
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            hadamard_matrix(96, CPU)

    def test_ue8m0_round_trips_and_clamps(self):
        exponents = torch.arange(UE8M0_MIN_EXP, UE8M0_MAX_EXP + 1)
        powers = torch.exp2(exponents.float())
        snapped, byte = ue8m0_encode(powers)

        # Exact powers of two must survive encoding untouched.
        torch.testing.assert_close(snapped, powers, rtol=0, atol=0)
        torch.testing.assert_close(
            byte.int(), (exponents + UE8M0_BIAS).int(), rtol=0, atol=0
        )
        torch.testing.assert_close(ue8m0_decode(byte), powers, rtol=0, atol=0)

        # Zero and non-finite inputs collapse to the byte-0 sentinel.
        special = torch.tensor([0.0, -1.0, float("inf"), float("nan")])
        snapped, byte = ue8m0_encode(special)
        self.assertEqual(int(byte.sum()), 0)
        self.assertEqual(float(snapped.abs().sum()), 0.0)

        # Exponents below the representable range clamp instead of wrapping.
        _, byte = ue8m0_encode(torch.tensor([2.0**-140]))
        self.assertEqual(int(byte.item()), UE8M0_MIN_EXP + UE8M0_BIAS)

    def test_pack_nibbles_round_trip(self):
        for head_dim in HEAD_DIMS:
            with self.subTest(head_dim=head_dim):
                codes = torch.randint(0, 16, (5, 3, head_dim), dtype=torch.uint8)
                packed = pack_nibbles(codes)
                self.assertEqual(packed.shape[-1], code_bytes(head_dim))
                torch.testing.assert_close(
                    unpack_nibbles(packed), codes, rtol=0, atol=0
                )

    def test_quantize_shapes_and_exact_values(self):
        for head_dim in HEAD_DIMS:
            for rotate in (True, False):
                with self.subTest(head_dim=head_dim, rotate=rotate):
                    x = torch.randn(4, 3, head_dim, dtype=torch.bfloat16)
                    codes, scales = UltraQuantKVQuantizeUtil.batched_quantize(
                        x, rotate=rotate
                    )
                    self.assertEqual(codes.shape, (4, 3, code_bytes(head_dim)))
                    self.assertEqual(scales.shape, (4, 3, n_groups(head_dim)))
                    self.assertEqual(codes.dtype, torch.uint8)
                    self.assertEqual(scales.dtype, torch.uint8)

            # Values already on the FP4 grid for their group must survive the
            # round trip exactly. Pin the group scale to 1.0 with a leading 8.0
            # -- that gives exponent round(log2(8 * 0.156)) == 0 -- then every
            # E2M1 level placed in the rest of the group must come back
            # unchanged. The 8.0 itself does not: 1 / CONSTANT_C is 6.41,
            # slightly past the 6.0 grid maximum, so the group maximum is
            # deliberately clipped. That clipping is what makes CONSTANT_C
            # MSE-optimal rather than range-preserving.
            with self.subTest(head_dim=head_dim, grid=True):
                grid = torch.zeros((1, 1, head_dim), dtype=torch.bfloat16)
                levels = torch.tensor(E2M1_LEVELS_SORTED, dtype=torch.bfloat16)
                for group in range(n_groups(head_dim)):
                    base = group * GROUP_SIZE
                    grid[0, 0, base] = 8.0
                    grid[0, 0, base + 1 : base + 1 + len(levels)] = levels

                codes, scales = UltraQuantKVQuantizeUtil.batched_quantize(
                    grid, rotate=False
                )
                recovered = UltraQuantKVQuantizeUtil.batched_dequantize(
                    codes, scales, dtype=torch.float32
                )
                self.assertEqual(int(scales.min()), UE8M0_BIAS)
                self.assertEqual(int(scales.max()), UE8M0_BIAS)
                on_grid = torch.ones(head_dim, dtype=torch.bool)
                on_grid[torch.arange(0, head_dim, GROUP_SIZE)] = False
                torch.testing.assert_close(
                    recovered[..., on_grid], grid.float()[..., on_grid], rtol=0, atol=0
                )

    def test_all_zero_input_uses_the_zero_sentinel(self):
        zeros = torch.zeros(2, 2, 256, dtype=torch.bfloat16)
        codes, scales = UltraQuantKVQuantizeUtil.batched_quantize(zeros, rotate=True)

        self.assertEqual(int(scales.sum()), 0, "zero groups must encode scale byte 0")
        expected_code = sorted_index_to_e2m1_bits(torch.tensor(E2M1_ZERO_LEVEL_INDEX))
        self.assertEqual(int(expected_code), 0)
        self.assertEqual(int(codes.sum()), 0)

        recovered = UltraQuantKVQuantizeUtil.batched_dequantize(codes, scales)
        self.assertEqual(float(recovered.abs().sum()), 0.0)

    def test_rotation_reduces_key_quantization_error(self):
        """The Hadamard rotation exists to tame per-channel outliers in keys."""
        torch.manual_seed(0)
        x = torch.randn(64, 8, 256, dtype=torch.bfloat16)
        x[..., 5] *= 30.0

        def relative_error(rotate: bool) -> float:
            codes, scales = UltraQuantKVQuantizeUtil.batched_quantize(x, rotate=rotate)
            recovered = UltraQuantKVQuantizeUtil.batched_dequantize(
                codes, scales, dtype=torch.float32
            )
            if rotate:
                # Undo the rotation so both variants are compared in the same basis.
                recovered = recovered @ hadamard_matrix(256, x.device)
            return ((recovered - x.float()).norm() / x.float().norm()).item()

        self.assertLess(
            relative_error(rotate=True), 0.75 * relative_error(rotate=False)
        )


if __name__ == "__main__":
    unittest.main()
