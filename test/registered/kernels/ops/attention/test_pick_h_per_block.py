"""Head-tile selection for the DSA fp8 partial decode kernel.

A pick that violates ``num_heads <= h or num_heads % h == 0`` crashes the
kernel. The policy is the largest safe tile whose grid still fills the GPU.
"""

import unittest

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd-mi35x")

CU = 256
BLOCK_PER_CU = 2
TOPK = 2048
BLOCK_I = 64
NUM_HEADS = 64
TILES = (64, 32, 16)


def _pick(seq, num_heads, n_groups, cu=CU, block_per_cu=BLOCK_PER_CU):
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import _pick_h_per_block

    return _pick_h_per_block(
        seq=seq,
        num_heads=num_heads,
        n_groups=n_groups,
        cu=cu,
        block_per_cu=block_per_cu,
    )


def _kernel_assert_holds(num_heads, h):
    return num_heads <= h or num_heads % h == 0


class TestPickHPerBlock(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_hip():
            raise unittest.SkipTest("tilelang DSA head tiles are the ROCm path")

    def test_pick_never_violates_the_kernel_assert(self):
        for num_heads in (1, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128, 192):
            for seq in (1, 12, 80, 512, 14472):
                for n_groups in (1, 8, 32):
                    with self.subTest(num_heads=num_heads, seq=seq, n_groups=n_groups):
                        h = _pick(seq, num_heads, n_groups)
                        self.assertIn(h, TILES)
                        self.assertTrue(_kernel_assert_holds(num_heads, h))

    def test_fallback_itself_satisfies_the_kernel_assert(self):
        """A fixed tile of 16 crashes when num_heads is not a multiple of 16."""
        for num_heads in range(1, 65):
            h = _pick(seq=1, num_heads=num_heads, n_groups=1)
            self.assertTrue(
                _kernel_assert_holds(num_heads, h),
                f"num_heads={num_heads} -> h={h}",
            )

    def test_pick_never_pads_past_num_heads(self):
        """A tile wider than num_heads keeps head_blocks at 1, so it only pads."""
        for num_heads in (16, 32, 64, 128):
            for seq in (1, 12, 80, 512, 14472):
                for n_groups in (1, 8, 32):
                    with self.subTest(num_heads=num_heads, seq=seq, n_groups=n_groups):
                        self.assertLessEqual(_pick(seq, num_heads, n_groups), num_heads)

    def test_production_shapes(self):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import _pick_inner_iter

        ni = TOPK // BLOCK_I
        for seq, expected in ((14472, 64), (80, 64), (12, 32)):
            with self.subTest(seq=seq):
                inner_iter = _pick_inner_iter(
                    seq=seq, ni=ni, cu=CU, block_per_cu=BLOCK_PER_CU
                )
                self.assertEqual(
                    _pick(
                        seq=seq,
                        num_heads=NUM_HEADS,
                        n_groups=ni // inner_iter,
                    ),
                    expected,
                )

    def test_empty_grid_falls_back_to_16(self):
        self.assertEqual(_pick(seq=1, num_heads=NUM_HEADS, n_groups=1), 16)

    def test_full_prefill_takes_the_widest_tile(self):
        self.assertEqual(_pick(seq=14472, num_heads=NUM_HEADS, n_groups=1), 64)


if __name__ == "__main__":
    unittest.main()
