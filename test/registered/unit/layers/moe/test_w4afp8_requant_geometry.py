"""CPU tests for the W4AFP8 low-latency requant launch geometry."""

import unittest

from sglang.kernels.ops.moe.ep_moe_kernels import requant_launch_geometry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DSV3_GROUPS = 7168 // 128  # 56
K3_GROUPS = 3584 // 128  # 28
PREVIOUS_FIXED_M_GRID = 32


class TestRequantLaunchGeometry(CustomTestCase):
    def test_unknown_row_count_keeps_the_previous_m_grid(self):
        """A caller with no row estimate keeps the historical m-grid.

        Only the m-grid: the tile width and warp count change for every caller.
        """
        for num_experts in (8, 56):
            _, m_grid = requant_launch_geometry(K3_GROUPS, num_experts)
            self.assertEqual(m_grid, PREVIOUS_FIXED_M_GRID)

    def test_tile_never_exceeds_the_payload(self):
        """A 512-wide hidden size is 4 groups; a wider tile would be mostly masked."""
        for num_groups in (1, 4, 12):
            for num_experts in (8, 56):
                g_block, _ = requant_launch_geometry(
                    num_groups, num_experts, expected_rows=64
                )
                self.assertLessEqual(g_block, num_groups)

    def test_m_grid_never_exceeds_the_previous_fixed_grid(self):
        """The estimate only ever shrinks the grid, so no batch can regress."""
        for expected_rows in (1, 8, 32, 33, 1024):
            for num_experts in (8, 56, 256):
                _, m_grid = requant_launch_geometry(
                    DSV3_GROUPS, num_experts, expected_rows=expected_rows
                )
                self.assertLessEqual(m_grid, PREVIOUS_FIXED_M_GRID)

    def test_m_grid_shrinks_as_the_expert_axis_fills_the_grid(self):
        """Hundreds of experts already saturate the grid without 32 programs each."""
        scarce = [
            requant_launch_geometry(DSV3_GROUPS, num_experts, expected_rows=32)[1]
            for num_experts in (8, 64, 128, 256)
        ]
        self.assertEqual(scarce, sorted(scarce, reverse=True))
        self.assertEqual(scarce[0], PREVIOUS_FIXED_M_GRID)
        self.assertLess(scarce[-1], scarce[0])

    def test_expert_cap_lifts_once_rows_carry_the_work(self):
        """Past the row threshold the extra programs are not just early exits."""
        for num_experts in (8, 128, 512):
            _, m_grid = requant_launch_geometry(
                DSV3_GROUPS, num_experts, expected_rows=1024
            )
            self.assertEqual(m_grid, PREVIOUS_FIXED_M_GRID)

    def test_dispatcher_round_up_does_not_bump_the_grid(self):
        """`dispatch_a` reports (rows + num_experts) // num_experts, i.e. one high.

        Rounding the grid up as well would launch twice the programs the sweep
        found optimal at every power of two.
        """
        for rows in (4, 8, 16, 32):
            exact = requant_launch_geometry(DSV3_GROUPS, 8, expected_rows=rows)[1]
            reported = requant_launch_geometry(DSV3_GROUPS, 8, expected_rows=rows + 1)[
                1
            ]
            self.assertEqual(reported, exact, f"rows={rows}")

    def test_m_grid_is_bounded_and_monotonic(self):
        previous = 0
        for expected_rows in range(1, 512):
            _, m_grid = requant_launch_geometry(
                K3_GROUPS, 8, expected_rows=expected_rows
            )
            # A floor keeps a one-row batch from serializing a whole expert into
            # one program, which measured several times slower than the cap does.
            self.assertGreaterEqual(m_grid, 4)
            self.assertLessEqual(m_grid, PREVIOUS_FIXED_M_GRID)
            # A larger batch never gets a smaller grid than a smaller one.
            self.assertGreaterEqual(m_grid, previous)
            previous = m_grid


if __name__ == "__main__":
    unittest.main()
