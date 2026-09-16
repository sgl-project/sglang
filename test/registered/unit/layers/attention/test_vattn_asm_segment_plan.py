"""Planner segment cap for the gfx950 assembly attention split.

The cap is the static grid.x of the planned split, so it bounds how much of the
device one long request may take. Deriving it from the exact equal-share CU
budget rather than a power-of-two floor of that share is what keeps a skewed
batch's long request from losing half its headroom at a batch discontinuity:
at 17 sequences on 256 CUs the pow2 form drops from 32 to 16, the exact form to
30. These tests pin that boundary, the 16/64 clamp, and monotonicity, so
restoring the rounding fails here instead of costing 1.2-1.8x on a skewed
batch.

    python -m pytest test/registered/unit/layers/attention/test_vattn_asm_segment_plan.py -v
"""

import unittest

from sglang.kernels.ops.attention.vattn_asm_gfx950 import mtp_verify_attn_seg_max
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# MI355X. Passed explicitly so the tests are device-independent and run on CPU.
NUM_CU = 256

# GQA 16 (TP2) and GQA 8 (TP4) are the only layouts the asm kernel serves.
KV_HEADS = (1, 2)


class TestVattnAsmSegmentPlanCap(CustomTestCase):
    def test_cap_is_twice_the_exact_uniform_cu_share(self):
        cases = [
            (8, 1, 64),  # share 32, clamped to 64
            (9, 1, 56),  # 28 is not a power of two
            (16, 1, 32),  # exact divisor: pow2 and exact agree
            (17, 1, 30),  # the discontinuity; the pow2 form gives 16
            (24, 1, 20),
            (29, 1, 16),  # share 8, clamped up to 16
            (9, 2, 28),
            (14, 2, 18),
            (15, 2, 16),
        ]
        for num_seqs, num_kv_heads, expected in cases:
            with self.subTest(num_seqs=num_seqs, num_kv_heads=num_kv_heads):
                self.assertEqual(
                    mtp_verify_attn_seg_max(num_seqs, num_kv_heads, num_cus=NUM_CU),
                    expected,
                )

    def test_cap_never_increases_with_more_sequences(self):
        """A cap that rose with the batch would hand each sequence more of the
        device exactly as the device got busier."""
        for num_kv_heads in KV_HEADS:
            with self.subTest(num_kv_heads=num_kv_heads):
                caps = [
                    mtp_verify_attn_seg_max(num_seqs, num_kv_heads, num_cus=NUM_CU)
                    for num_seqs in range(1, NUM_CU + 2)
                ]
                self.assertEqual(caps, sorted(caps, reverse=True))

    def test_cap_stays_within_the_kernel_bounds(self):
        """The partial buffers are allocated for the cap, and the assembly
        kernel's segment loop is built for at most 64."""
        for num_kv_heads in KV_HEADS:
            for num_seqs in range(1, NUM_CU + 2):
                cap = mtp_verify_attn_seg_max(num_seqs, num_kv_heads, num_cus=NUM_CU)
                with self.subTest(num_seqs=num_seqs, num_kv_heads=num_kv_heads):
                    self.assertGreaterEqual(cap, 16)
                    self.assertLessEqual(cap, 64)


if __name__ == "__main__":
    unittest.main()
