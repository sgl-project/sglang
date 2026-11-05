# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""A compact BF16 counterexample to expanding the AttnRes residual norm.

This models the FP32 reduction trees on synthetic rows only. The companion kernel
test checks the actual consumers; this CPU witness does not compile CuTe.
"""

import math
import struct

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _dot(a, b, consumer):
    warps = []
    for rank in range(7):
        row = []
        for warp in range(4):
            lanes = []
            for lane in range(32):
                start = (rank * 128 + warp * 32 + lane) * 8
                # All products and eight-element sums in this integer-valued
                # BF16 fixture are exact FP32, including either kernel's FMA tree.
                lanes.append(sum(a[i] * b[i] for i in range(start, start + 8)))
            for shift in (16, 8, 4, 2, 1):
                lanes = [_f32(lanes[i] + lanes[i ^ shift]) for i in range(32)]
            row.append(lanes[0])
        warps.append(row)
    if consumer == "oproj":
        sums = [0.0] * 4
        for row in warps:
            sums = [_f32(a + b) for a, b in zip(sums, row)]
        return _f32(_f32(sums[0] + sums[1]) + _f32(sums[2] + sums[3]))
    sums = [0.0, 0.0]
    for rank, row in enumerate(warps):
        cta = _f32(_f32(row[0] + row[1]) + _f32(row[2] + row[3]))
        sums[rank % 2] = _f32(sums[rank % 2] + cta)
    return _f32(sums[0] + sums[1])


class AttnResStability(CustomTestCase):
    def test_nearly_opposing_bf16_rows_need_a_direct_residual_norm(self):
        saved, prefix = [140.0] * 7168, [-140.0] * 7168
        prefix[0] = -142.0
        residual = [a + b for a, b in zip(saved, prefix)]
        for consumer in ("cluster", "oproj"):
            with self.subTest(consumer=consumer):
                saved_sq = _dot(saved, saved, consumer)
                cross = _dot(saved, prefix, consumer)
                prefix_sq = _dot(prefix, prefix, consumer)
                # Zero score-query weights give alpha=ep=1. Multiplication by
                # these powers of two is exact, so this also models the old FMAs.
                expanded = _f32(saved_sq + _f32(2 * cross + prefix_sq))
                direct = _dot(residual, residual, consumer)
                self.assertEqual(expanded, -16.0)
                self.assertEqual(direct, 4.0)
                self.assertEqual(direct, math.fsum(x * x for x in residual))
                for denominator in (2, 9):  # nb=1 and nb=8, including zero bank rows
                    eps = _f32(_f32(_f32(1.0e-5) * denominator) * denominator)
                    old_arg = _f32(expanded * _f32(1 / 7168) + eps)
                    new_arg = _f32(direct * _f32(1 / 7168) + eps)
                    self.assertLess(old_arg, 0)
                    self.assertGreater(new_arg, 0)
                    self.assertTrue(math.isfinite(1 / math.sqrt(new_arg)))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
