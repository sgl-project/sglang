"""DeepSeek-V4.1 bounded replay state the prefill CUDA graphs capture, padded to fixed row counts."""

import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    SWA_WINDOW,
    LateLayerTail,
    _prefill_graph_tail_row_buckets,
)
from sglang.srt.mem_cache.dsv41_request_window import window_layout
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


class TestGraphPaddedWindowLayout(CustomTestCase):
    def test_padding_rows_leave_every_request_window_unchanged(self):
        """Two requests padded to a graph bucket and to more window groups than
        requests: the live rows keep their unpadded layout shifted by the extra
        history rows, and padding rows read nothing and commit nowhere."""
        req = torch.tensor([3, 3, 3, 7, 7])
        pos = torch.tensor([200, 201, 202, 5, 6])
        window, live, groups, padded = 4, 5, 3, 8
        plain = window_layout(req, pos, window=window, capacity=8, num_groups=2)
        graph = window_layout(
            req,
            pos,
            window=window,
            capacity=8,
            num_groups=groups,
            padded_rows=padded,
        )

        extra_history = (groups - 2) * window
        self.assertEqual(graph.size, groups * window + padded)
        self.assertEqual(
            graph.write_loc[:live].tolist(), (plain.write_loc + extra_history).tolist()
        )
        # History slots keep their place; this batch's own rows move past the padding groups.
        shifted = torch.where(
            plain.indices >= 2 * window, plain.indices + extra_history, plain.indices
        )
        self.assertEqual(graph.indices[:live].tolist(), shifted.tolist())
        self.assertEqual(graph.commit_mask[:live].tolist(), plain.commit_mask.tolist())
        self.assertEqual(
            graph.history_valid[: 2 * window].tolist(), plain.history_valid.tolist()
        )
        self.assertFalse(graph.history_valid[2 * window :].any())

        self.assertFalse(graph.commit_mask[live:].any())
        self.assertEqual(graph.lengths[live:].tolist(), [0] * (padded - live))
        self.assertEqual(len(set(graph.write_loc.tolist())), padded)


if __name__ == "__main__":
    unittest.main()
