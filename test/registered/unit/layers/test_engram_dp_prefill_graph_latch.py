"""The DP-sharded engram exchange latches the prefill-graph DP-gather flag. CPU only.

The exchange (all-gatherv of the hash ids, then a reduce-scatter or all-to-all)
records its per-rank counts into a captured prefill graph. MegaMoE may otherwise
replay prefill graphs with per-rank SUM_LEN buckets, which lets an idle DP rank
run the step eagerly while its peer replays the graph, and the two sides issue
different collectives.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.engram_config import EngramLookup
from sglang.srt.layers import engram
from sglang.srt.layers.engram import EngramEmbedding
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DIM = 4
ROWS = 2


def _embedding(lookup, sizes):
    """An embedding whose exchange returns zeros; only the collectives' bookkeeping
    runs for real."""
    rows = torch.zeros(ROWS, DIM)
    return SimpleNamespace(
        lookup=lookup,
        dim=DIM,
        new_rows=lambda indices: rows.new_zeros((*indices.shape, DIM)),
        _dp_sizes=lambda indices, batch: sizes,
        _dp_global_buffer_lookup=lambda indices, batch: rows.view(ROWS, 1, DIM),
        _dp_reduce_scatter_lookup=lambda ids, sizes: rows,
        _dp_all_to_all_lookup=lambda ids, sizes: rows,
    )


def _lookup(embed, capturing):
    group = SimpleNamespace(all_gatherv=lambda tensor, sizes, output: output)
    indices = torch.zeros(ROWS, 1, dtype=torch.int64)
    with (
        patch.object(
            engram, "get_parallel", return_value=SimpleNamespace(tp_group=group)
        ),
        get_flags().dp.override(
            capturing_prefill_graph=capturing, prefill_graph_has_dp_gather=False
        ),
    ):
        EngramEmbedding._dp_sharded_lookup(embed, indices, SimpleNamespace())
        return get_flags().dp.prefill_graph_has_dp_gather


class TestEngramDpPrefillGraphLatch(CustomTestCase):
    def test_exchange_latches_while_capturing(self):
        for lookup in (EngramLookup.REDUCE_SCATTER, EngramLookup.ALL_TO_ALL):
            with self.subTest(lookup=lookup):
                embed = _embedding(lookup, sizes=[ROWS, ROWS])
                self.assertTrue(_lookup(embed, capturing=True))

    def test_exchange_leaves_the_flag_outside_capture(self):
        embed = _embedding(EngramLookup.ALL_TO_ALL, sizes=[ROWS, ROWS])
        self.assertFalse(_lookup(embed, capturing=False))

    def test_global_buffer_path_is_left_to_the_dp_gather(self):
        # dp_gather_replicate latches on its own; this path adds no exchange.
        embed = _embedding(EngramLookup.ALL_TO_ALL, sizes=None)
        self.assertFalse(_lookup(embed, capturing=True))


if __name__ == "__main__":
    unittest.main()
