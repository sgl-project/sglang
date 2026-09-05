"""A tracked SSM position that sits on the global chunk grid must be read from
`h`, not rebuilt: rebuilding it is numerically fine but not bit-exact, which
makes the cached state drift with the prefill batch composition.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import Mamba2AttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CHUNK = 128


def _backend():
    backend = object.__new__(Mamba2AttnBackend)
    backend.device = "cpu"
    backend._mamba_chunk_size = CHUNK
    return backend


def _forward_batch(extend_lens, prefix_lens, track_seqlens, track_mask):
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(extend_lens),
        extend_prefix_lens=torch.tensor(prefix_lens),
        mamba_track_seqlens=torch.tensor(track_seqlens),
        mamba_track_mask=torch.tensor(track_mask),
        mamba_track_indices=torch.arange(100, 100 + len(extend_lens)),
    )


def _split(extend_lens, prefix_lens, track_seqlens, track_mask):
    backend = _backend()
    cache_indices = torch.arange(len(extend_lens))
    (
        h_src,
        h_dst,
        _final_src,
        _final_dst,
        seq_idx,
        end_locs,
        recompute_dst,
    ) = backend._init_track_ssm_indices(
        cache_indices,
        _forward_batch(extend_lens, prefix_lens, track_seqlens, track_mask),
    )
    return h_src, h_dst, seq_idx, end_locs, recompute_dst


class TestMamba2TrackSsmIndices(unittest.TestCase):
    def test_on_grid_request_reads_h(self):
        # req0 is chunk-aligned, so req1 starts at flat 256 and its tracked
        # position 256 + 128 is on the grid.
        h_src, h_dst, seq_idx, _end_locs, recompute_dst = _split(
            extend_lens=[256, 320],
            prefix_lens=[0, 0],
            track_seqlens=[0, 200],
            track_mask=[False, True],
        )
        self.assertEqual(h_src.tolist(), [3])
        self.assertEqual(h_dst.tolist(), [101])
        self.assertEqual(seq_idx.numel(), 0)
        self.assertEqual(recompute_dst.numel(), 0)

    def test_off_grid_request_is_rebuilt(self):
        # req0 is not chunk-aligned, so req1 starts at flat 192 and its tracked
        # position 192 + 128 = 320 is not a multiple of 128.
        h_src, _h_dst, seq_idx, end_locs, recompute_dst = _split(
            extend_lens=[192, 320],
            prefix_lens=[0, 0],
            track_seqlens=[0, 200],
            track_mask=[False, True],
        )
        self.assertEqual(h_src.numel(), 0)
        self.assertEqual(seq_idx.tolist(), [1])
        self.assertEqual(end_locs.tolist(), [320])
        self.assertEqual(recompute_dst.tolist(), [101])

    def test_mixed_batch_splits_without_overlap(self):
        # req0 on grid (flat 0), req1 off grid (flat 192), req2 on grid (flat 512).
        h_src, h_dst, seq_idx, end_locs, recompute_dst = _split(
            extend_lens=[192, 320, 448],
            prefix_lens=[0, 0, 0],
            track_seqlens=[150, 200, 300],
            track_mask=[True, True, True],
        )
        self.assertEqual(h_src.tolist(), [1, 6])
        self.assertEqual(h_dst.tolist(), [100, 102])
        self.assertEqual(seq_idx.tolist(), [1])
        self.assertEqual(end_locs.tolist(), [320])
        self.assertEqual(recompute_dst.tolist(), [101])
        self.assertEqual(
            sorted(h_dst.tolist() + recompute_dst.tolist()), [100, 101, 102]
        )

    def test_end_locs_are_chunk_boundaries_of_the_flat_batch(self):
        # h is indexed by flat chunk, so a read index must satisfy
        # h_src * CHUNK == the flat token the state belongs to.
        starts = [0, 192, 512]
        h_src, h_dst, seq_idx, end_locs, _recompute_dst = _split(
            extend_lens=[192, 320, 448],
            prefix_lens=[0, 0, 0],
            track_seqlens=[150, 200, 300],
            track_mask=[True, True, True],
        )
        want = {100: starts[0] + 128, 102: starts[2] + 256}
        for dst, src in zip(h_dst.tolist(), h_src.tolist()):
            self.assertEqual(src * CHUNK, want[dst])
        for i, end in zip(seq_idx.tolist(), end_locs.tolist()):
            self.assertNotEqual(end % CHUNK, 0)
            self.assertGreaterEqual(end, starts[i])


if __name__ == "__main__":
    unittest.main()
