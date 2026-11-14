import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestScaleKvCellSizePerTokenForDFlash(unittest.TestCase):
    """Sizing of the combined target+draft KV budget, including the HiSparse
    loc-space widening of the draft contribution."""

    def test_estimate_path_matches_historical_formula_at_scale_one(self):
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        for target_cell, t_layers, d_layers in [
            (1, 3, 1),
            (10, 3, 1),
            (10, 4, 2),
            (7, 3, 1),
            (1024, 45, 5),
            (2048, 78, 8),
        ]:
            with self.subTest(
                target_cell=target_cell, t_layers=t_layers, d_layers=d_layers
            ):
                got = scale_kv_cell_size_per_token_for_dflash(
                    target_cell_size_per_token=target_cell,
                    target_num_layers=t_layers,
                    draft_num_layers=d_layers,
                )
                total_layers = t_layers + d_layers
                historical = (target_cell * total_layers + t_layers - 1) // t_layers
                self.assertEqual(got, historical)

    def test_loc_space_scale_widens_only_the_draft_part(self):
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        target_cell, t_layers, d_layers, ratio = 2048, 45, 5, 2
        base = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=target_cell,
            target_num_layers=t_layers,
            draft_num_layers=d_layers,
        )
        draft_part = base - target_cell
        widened = scale_kv_cell_size_per_token_for_dflash(
            target_cell_size_per_token=target_cell,
            target_num_layers=t_layers,
            draft_num_layers=d_layers,
            draft_loc_space_scale=ratio,
        )
        self.assertEqual(widened, target_cell + ratio * draft_part)

    def test_loc_space_scale_applies_to_explicit_draft_cell_size(self):
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        self.assertEqual(
            scale_kv_cell_size_per_token_for_dflash(
                target_cell_size_per_token=100,
                target_num_layers=45,
                draft_num_layers=5,
                draft_cell_size_per_token=7,
                draft_loc_space_scale=3,
            ),
            100 + 3 * 7,
        )

    def test_non_positive_layer_counts_leave_cell_size_unchanged(self):
        from sglang.srt.speculative.dflash_utils import (
            scale_kv_cell_size_per_token_for_dflash,
        )

        for t_layers, d_layers in [(0, 5), (45, 0), (-1, 5)]:
            with self.subTest(t_layers=t_layers, d_layers=d_layers):
                self.assertEqual(
                    scale_kv_cell_size_per_token_for_dflash(
                        target_cell_size_per_token=123,
                        target_num_layers=t_layers,
                        draft_num_layers=d_layers,
                        draft_loc_space_scale=2,
                    ),
                    123,
                )


if __name__ == "__main__":
    unittest.main()
