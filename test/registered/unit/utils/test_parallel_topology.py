# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from sglang.srt.utils.parallel_topology import (
    calculate_rank_ranges,
    validate_standard_rank_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _unchecked_rank_ranges(nnodes, pp_size, tp_size, node_rank):
    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )
    tp_size_per_node = tp_size // nnodes_per_pp_rank
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_pp_rank),
        tp_size_per_node * (node_rank % nnodes_per_pp_rank + 1),
    )
    return pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node


def _expand_global_ranks(calculator, nnodes, pp_size, tp_size):
    ranks = []
    for node_rank in range(nnodes):
        pp_rank_range, tp_rank_range, _, _ = calculator(
            nnodes, pp_size, tp_size, node_rank
        )
        ranks.extend(
            tp_size * pp_rank + tp_rank
            for pp_rank in pp_rank_range
            for tp_rank in tp_rank_range
        )
    return sorted(ranks)


class TestParallelTopology(unittest.TestCase):
    def test_rejects_layout_that_generates_out_of_range_ranks(self):
        with self.assertRaises(ValueError) as context:
            calculate_rank_ranges(nnodes=4, pp_size=3, tp_size=8, node_rank=0)

        message = str(context.exception)
        self.assertIn("tp_size=8", message)
        self.assertIn("pp_size=3", message)
        self.assertIn("nnodes=4", message)

    def test_rejects_uneven_pp_stages_per_node(self):
        with self.assertRaisesRegex(ValueError, "pp_size must be divisible"):
            validate_standard_rank_layout(nnodes=2, pp_size=3, tp_size=2)

    def test_preserves_known_legal_layouts(self):
        layouts = [
            (1, 7, 5),
            (4, 1, 8),
            (2, 4, 1),
            (3, 3, 8),
            (4, 2, 8),
            (2, 4, 2),
        ]
        for nnodes, pp_size, tp_size in layouts:
            with self.subTest(nnodes=nnodes, pp_size=pp_size, tp_size=tp_size):
                self.assertEqual(
                    _expand_global_ranks(
                        calculate_rank_ranges, nnodes, pp_size, tp_size
                    ),
                    list(range(tp_size * pp_size)),
                )

    def test_validation_matches_rank_coverage_in_small_grid(self):
        for nnodes in range(1, 9):
            for pp_size in range(1, 9):
                for tp_size in range(1, 9):
                    with self.subTest(nnodes=nnodes, pp_size=pp_size, tp_size=tp_size):
                        unchecked_ranks = _expand_global_ranks(
                            _unchecked_rank_ranges, nnodes, pp_size, tp_size
                        )
                        expected_ranks = list(range(tp_size * pp_size))

                        if unchecked_ranks == expected_ranks:
                            self.assertEqual(
                                _expand_global_ranks(
                                    calculate_rank_ranges,
                                    nnodes,
                                    pp_size,
                                    tp_size,
                                ),
                                expected_ranks,
                            )
                        else:
                            with self.assertRaises(ValueError):
                                validate_standard_rank_layout(nnodes, pp_size, tp_size)

    def test_weight_cache_rejects_before_spawning_processes(self):
        from sglang.srt.weight_cache.daemon import launch_weight_cache_daemons

        with patch("subprocess.Popen") as popen:
            with self.assertRaises(ValueError):
                launch_weight_cache_daemons(
                    model_path="dummy",
                    tp_size=8,
                    pp_size=3,
                    nnodes=4,
                    node_rank=0,
                    dist_init_method="tcp://127.0.0.1:29500",
                )

        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
