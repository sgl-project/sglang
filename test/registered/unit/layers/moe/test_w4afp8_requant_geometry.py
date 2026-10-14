"""CPU tests for the W4AFP8 low-latency requant launch geometry."""

import unittest

from sglang.kernels.ops.moe.ep_moe_kernels import requant_launch_geometry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

DSV3_GROUPS = 7168 // 128  # 56
K3_GROUPS = 3584 // 128  # 28
PREVIOUS_FIXED_M_GRID = 32
ROW_CAP_SLACK = 2


class TestRequantLaunchGeometry(CustomTestCase):
    def test_cap_leaves_ordinary_variation_to_the_owning_expert(self):
        """Rows below the cap stay on their expert; the shared path costs a
        lookup per row and only pays under real imbalance."""
        for expected_rows in (1, 4, 16, 64, 256):
            _, _, row_cap = requant_launch_geometry(
                DSV3_GROUPS, 64, expected_rows=expected_rows
            )
            self.assertGreaterEqual(row_cap, expected_rows * ROW_CAP_SLACK)

    def test_cap_never_exceeds_the_payload(self):
        """A cap past the padded rows would leave the shared path unreachable."""
        for max_rows in (1, 8, 128):
            for expected_rows in (1, 64, 4096):
                _, _, row_cap = requant_launch_geometry(
                    DSV3_GROUPS, 64, expected_rows=expected_rows, max_rows=max_rows
                )
                self.assertLessEqual(row_cap, max_rows)

    def test_unknown_row_count_keeps_every_row_with_its_expert(self):
        """With no estimate there is nothing to place a cap against."""
        for num_experts in (8, 56):
            _, m_grid, row_cap = requant_launch_geometry(
                K3_GROUPS, num_experts, max_rows=128
            )
            self.assertEqual(m_grid, PREVIOUS_FIXED_M_GRID)
            self.assertEqual(row_cap, 128)

    def test_m_grid_never_exceeds_the_previous_fixed_grid(self):
        """The estimate only ever shrinks the grid, so no batch can regress."""
        for expected_rows in (1, 8, 32, 33, 1024):
            for num_experts in (8, 56, 256):
                _, m_grid, _ = requant_launch_geometry(
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
            _, m_grid, _ = requant_launch_geometry(
                DSV3_GROUPS, num_experts, expected_rows=1024
            )
            self.assertEqual(m_grid, PREVIOUS_FIXED_M_GRID)

    def test_dispatcher_round_up_does_not_bump_the_grid(self):
        """dispatch_a reports (rows + num_experts) // num_experts, one high;
        rounding up as well would double the launch at every power of two."""
        for rows in (4, 8, 16, 32):
            exact = requant_launch_geometry(DSV3_GROUPS, 8, expected_rows=rows)[1]
            reported = requant_launch_geometry(DSV3_GROUPS, 8, expected_rows=rows + 1)[
                1
            ]
            self.assertEqual(reported, exact, f"rows={rows}")

    def test_m_grid_is_bounded_and_monotonic(self):
        previous = 0
        for expected_rows in range(1, 512):
            _, m_grid, _ = requant_launch_geometry(
                K3_GROUPS, 8, expected_rows=expected_rows
            )
            # The floor keeps a one-row batch from serializing an expert into
            # one program (measured several times slower).
            self.assertGreaterEqual(m_grid, 4)
            self.assertLessEqual(m_grid, PREVIOUS_FIXED_M_GRID)
            self.assertGreaterEqual(m_grid, previous)
            previous = m_grid

    def test_tile_never_exceeds_the_payload(self):
        """A 512-wide hidden size is 4 groups; a wider tile would be mostly masked."""
        for num_groups in (1, 4, 12):
            for num_experts in (8, 56):
                g_block, _, _ = requant_launch_geometry(
                    num_groups, num_experts, expected_rows=64
                )
                self.assertLessEqual(g_block, num_groups)

    def test_tile_holds_bytes_per_lane_across_warp_widths(self):
        """The tuned unit is bytes per lane: 2048 elements at warp 32 must become
        4096 at warp 64, or a wave64 part gets half the bytes per lane."""
        for warp_size, want_elems in ((32, 2048), (64, 4096)):
            for group_size in (64, 128, 256, 512):
                g_block, _, _ = requant_launch_geometry(
                    num_groups=7168 // group_size,
                    num_experts=56,
                    group_size=group_size,
                    expected_rows=16,
                    warp_size=warp_size,
                )
                self.assertEqual(
                    g_block * group_size, want_elems, f"warp_size={warp_size}"
                )

    def test_few_experts_halve_the_tile_on_either_warp_width(self):
        """A grid too small to fill the part buys k-blocks by halving the tile."""
        for warp_size, want_elems in ((32, 1024), (64, 2048)):
            g_block, _, _ = requant_launch_geometry(
                DSV3_GROUPS, 8, expected_rows=16, warp_size=warp_size
            )
            self.assertEqual(g_block * 128, want_elems, f"warp_size={warp_size}")

    def test_warp_width_does_not_move_the_m_grid(self):
        """The two knobs are independent: the m-grid answers to rows and experts."""
        for num_experts in (8, 56, 256):
            for expected_rows in (1, 8, 32, 1024):
                grids = {
                    requant_launch_geometry(
                        DSV3_GROUPS,
                        num_experts,
                        expected_rows=expected_rows,
                        warp_size=warp_size,
                    )[1]
                    for warp_size in (32, 64)
                }
                self.assertEqual(len(grids), 1, f"E={num_experts} rows={expected_rows}")


if __name__ == "__main__":
    unittest.main()
