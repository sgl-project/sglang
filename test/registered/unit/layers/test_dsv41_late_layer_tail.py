"""The DeepSeek-V4.1 late-layer tail that the late-layer CUDA graphs capture, padded to a bucket of rows."""

import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    LateLayerTail,
    _prefill_graph_tail_row_buckets,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _padded_tail(token_indices, *, graph_rows: int) -> LateLayerTail:
    padding = graph_rows - len(token_indices)
    return LateLayerTail(
        token_indices=torch.tensor(token_indices + [0] * padding),
        positions=torch.zeros(graph_rows, dtype=torch.int64),
        extend_seq_lens=torch.tensor([len(token_indices)], dtype=torch.int32),
        extend_seq_lens_cpu=[len(token_indices)],
        swa_out_cache_loc=torch.tensor(token_indices + [0] * padding),
        padded=True,
    )


class TestGraphPaddedLateLayerTail(CustomTestCase):
    def test_every_tail_fits_a_bucket_that_pads_it_by_less_than_half(self):
        """A tail of any size must land in a captured bucket, or its batch could not
        replay, and the bucket must stay close, or the late layers waste their win."""
        for max_rows in (1, SWA_WINDOW, 300, 4096, 16384, 20000):
            buckets = _prefill_graph_tail_row_buckets(max_rows)
            self.assertEqual(buckets, sorted(set(buckets)))
            self.assertGreaterEqual(buckets[-1], max_rows)
        buckets = _prefill_graph_tail_row_buckets(16384)
        for tail_rows in range(SWA_WINDOW, 16385, 37):
            bucket = min(rows for rows in buckets if rows >= tail_rows)
            self.assertLess(bucket, 1.5 * tail_rows)

    def test_replay_refresh_keeps_the_captured_index_tensors(self):
        """A late-layer graph reads the tail's positions and write slots by address,
        so a replay must refill those tensors, not rebind them."""
        captured = _padded_tail([1, 2, 3], graph_rows=4)
        live = _padded_tail([2, 3], graph_rows=4)

        refreshed = captured.refresh_for_breakable_cuda_graph_replay_(live)

        self.assertIs(refreshed.token_indices, captured.token_indices)
        self.assertIs(refreshed.positions, captured.positions)
        self.assertIs(refreshed.swa_out_cache_loc, captured.swa_out_cache_loc)
        self.assertEqual(refreshed.token_indices.tolist(), [2, 3, 0, 0])
        self.assertEqual(refreshed.swa_out_cache_loc.tolist(), [2, 3, 0, 0])
        # The DSpark target reads only the live rows of what the graph returns.
        self.assertEqual(refreshed.live_rows(torch.arange(4)).tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
