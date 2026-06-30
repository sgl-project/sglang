"""Unit tests for ``read_mode_stat`` — no server, no model loading.

``read_mode_stat`` reads the offline dumps produced by
``ExpertDistributionRecorder`` when ``expert_distribution_recorder_mode`` is
``stat``/``stat_approx``. These tests synthesize dump files in the exact
on-disk format the recorder writes (one rank-0 file per ``dump_record`` call,
each holding ``[num_buffered_steps, num_layers, num_logical_experts]`` logical
counts plus a utilization-rate scalar) and assert the reader reconstructs the
full step-ordered time series.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from sglang.srt.eplb.eplb_simulator.reader import read_mode_stat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# Match the recorder's on-disk layout: dict(rank, logical_count, average_utilization_rate_over_window)
# with logical_count shaped [num_buffered_steps, num_layers, num_logical_experts].
_DEFAULT_FILE = "expert_distribution_recorder_{}.pt"


def _write_stat_dump(dir_data: Path, ts: float, logical_count: torch.Tensor, util):
    torch.save(
        dict(
            rank=0,
            logical_count=logical_count,
            average_utilization_rate_over_window=util,
        ),
        str(dir_data / _DEFAULT_FILE.format(ts)),
    )


class TestReadModeStat(CustomTestCase):
    def test_single_file_preserves_shape_and_values(self):
        with TemporaryDirectory() as d:
            dir_data = Path(d)
            lc = torch.arange(2 * 3 * 4, dtype=torch.int32).reshape(2, 3, 4)
            _write_stat_dump(dir_data, 1700000000.0, lc, 0.42)

            out = read_mode_stat(dir_data)

            self.assertEqual(out["logical_count_of_step"].shape, (2, 3, 4))
            self.assertTrue(torch.equal(out["logical_count_of_step"], lc))
            self.assertEqual(out["average_utilization_rate_of_step"], [0.42])

    def test_multiple_files_stacked_along_step_dim_in_order(self):
        # Two dumps; the reader must concatenate along dim 0 in filename order.
        with TemporaryDirectory() as d:
            dir_data = Path(d)
            a = torch.zeros((2, 3, 4), dtype=torch.int32)
            b = torch.ones((3, 3, 4), dtype=torch.int32) * 7
            # Distinct timestamps so filename order == chronological order.
            _write_stat_dump(dir_data, 1700000001.0, a, 0.1)
            _write_stat_dump(dir_data, 1700000002.0, b, 0.2)

            out = read_mode_stat(dir_data)

            self.assertEqual(out["logical_count_of_step"].shape, (5, 3, 4))
            expected = torch.cat([a, b], dim=0)
            self.assertTrue(torch.equal(out["logical_count_of_step"], expected))
            # Step-ordered utilization series.
            self.assertEqual(out["average_utilization_rate_of_step"], [0.1, 0.2])

    def test_filename_sort_determines_step_order(self):
        # Larger timestamp dumped first on disk; reader must still order by name.
        with TemporaryDirectory() as d:
            dir_data = Path(d)
            early = torch.full((1, 1, 2), 1, dtype=torch.int32)
            late = torch.full((1, 1, 2), 2, dtype=torch.int32)
            _write_stat_dump(dir_data, 1700000099.0, late, 0.9)  # written first
            _write_stat_dump(dir_data, 1700000001.0, early, 0.1)  # written second

            out = read_mode_stat(dir_data)

            self.assertEqual(out["average_utilization_rate_of_step"], [0.1, 0.9])
            self.assertEqual(out["logical_count_of_step"][0].tolist(), [[1, 1]])
            self.assertEqual(out["logical_count_of_step"][1].tolist(), [[2, 2]])

    def test_none_utilization_rate_is_preserved(self):
        # When EPLB metrics are disabled the recorder writes ``None``.
        with TemporaryDirectory() as d:
            dir_data = Path(d)
            _write_stat_dump(
                dir_data, 1700000000.0, torch.zeros((1, 2, 2), dtype=torch.int32), None
            )
            out = read_mode_stat(dir_data)
            self.assertEqual(out["average_utilization_rate_of_step"], [None])

    def test_empty_directory_returns_empty_tensor_and_list(self):
        with TemporaryDirectory() as d:
            out = read_mode_stat(Path(d))
            self.assertEqual(out["logical_count_of_step"].numel(), 0)
            self.assertEqual(out["average_utilization_rate_of_step"], [])

    def test_non_zero_rank_dump_is_rejected(self):
        # Stat dumps are rank-0 only; a stray rank-N file must raise.
        with TemporaryDirectory() as d:
            dir_data = Path(d)
            torch.save(
                dict(
                    rank=1,
                    logical_count=torch.zeros((1, 1, 1), dtype=torch.int32),
                    average_utilization_rate_over_window=0.5,
                ),
                str(dir_data / _DEFAULT_FILE.format(1700000000.0)),
            )
            with self.assertRaises(AssertionError):
                read_mode_stat(dir_data)


if __name__ == "__main__":
    unittest.main()
