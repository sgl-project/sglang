"""The DeepSeek-V4.1 late-layer tail a prefill CUDA graph captures, padded to a fixed row count."""

import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import LateLayerTail
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _padded_tail(token_indices, *, graph_rows: int, num_tokens: int) -> LateLayerTail:
    padding = graph_rows - len(token_indices)
    return LateLayerTail(
        token_indices=torch.tensor(token_indices + [num_tokens] * padding),
        positions=torch.zeros(graph_rows, dtype=torch.int64),
        extend_seq_lens=torch.tensor([len(token_indices)], dtype=torch.int32),
        extend_seq_lens_cpu=[len(token_indices)],
        swa_out_cache_loc=torch.tensor(token_indices + [0] * padding),
        spare_row=num_tokens,
    )


class TestGraphPaddedLateLayerTail(CustomTestCase):
    def test_padding_rows_never_overwrite_a_tail_row(self):
        """The tail ends on the extend's last row, the row padding reads. Scattering
        the padded tail back must still return every tail row, not a padding row's."""
        num_tokens = 8
        tail = _padded_tail([5, 6, 7], graph_rows=6, num_tokens=num_tokens)
        full = torch.arange(num_tokens, dtype=torch.float32)[:, None].repeat(1, 2)

        rows = tail.rows(full)
        self.assertEqual(rows.shape[0], 6)
        # A late layer changes padding rows too; they must not reach the output.
        rows = rows.clone()
        rows[3:] = -1.0
        scattered = tail.scatter(rows, num_tokens)

        self.assertEqual(scattered.shape[0], num_tokens)
        self.assertEqual(scattered[5:, 0].tolist(), [5.0, 6.0, 7.0])

    def test_replay_refresh_keeps_the_captured_index_tensors(self):
        """The graph reads the tail's row indices, positions and write slots by
        address, so a replay must refill those tensors, not rebind them."""
        captured = _padded_tail([1, 2, 3], graph_rows=4, num_tokens=4)
        live = _padded_tail([2, 3], graph_rows=4, num_tokens=4)

        refreshed = captured.refresh_for_breakable_cuda_graph_replay_(live)

        self.assertIs(refreshed.token_indices, captured.token_indices)
        self.assertIs(refreshed.positions, captured.positions)
        self.assertEqual(refreshed.token_indices.tolist(), [2, 3, 4, 4])
        self.assertIs(refreshed.swa_out_cache_loc, captured.swa_out_cache_loc)
        self.assertEqual(refreshed.swa_out_cache_loc.tolist(), [2, 3, 0, 0])
        self.assertEqual(refreshed.extend_seq_lens_cpu, [2])


if __name__ == "__main__":
    unittest.main()
