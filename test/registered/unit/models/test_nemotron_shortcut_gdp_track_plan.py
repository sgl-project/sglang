"""Extend-time GDP state snapshots for the extra_buffer radix cache."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.linear.gdp_backend import (  # noqa: E402
    build_gdp_track_plan,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _list(t):
    return None if t is None else t.tolist()


class TestGDPTrackPlan(CustomTestCase):
    def test_rows_split_at_their_snapshot_point(self):
        # chunk 64. Row 0 snapshots inside its extend (T = 64 of 100), row 1 at
        # its end (T == len), row 2 before any extend token (T == 0), row 3 is
        # not tracked. track_seqlen carries the +1 that forces an h snapshot.
        plan = build_gdp_track_plan(
            track_rows=[0, 1, 2],
            track_seqlens=[65, 128, 130, -1],
            extend_lens=[100, 64, 10, 5],
            prefix_lens=[0, 64, 128, 0],
            chunk_size=64,
            cache_indices=torch.tensor([10, 11, 12, 13]),
            track_dst=torch.tensor([20, 21, 22, 23]),
        )
        self.assertEqual(_list(plan.conv_dst), [20, 21, 22])
        self.assertEqual((_list(plan.split_src), _list(plan.split_dst)), ([10], [20]))
        self.assertEqual((_list(plan.post_src), _list(plan.post_dst)), ([11], [21]))
        self.assertEqual((_list(plan.pre_src), _list(plan.pre_dst)), ([12], [22]))
        # First pass: the split row up to T, every other row whole.
        self.assertEqual(
            _list(plan.first_tokens),
            list(range(0, 64)) + list(range(100, 179)),
        )
        self.assertEqual(_list(plan.first_cu_seqlens), [0, 64, 128, 138, 143])
        # Second pass: only the split row's remaining tokens.
        self.assertEqual(_list(plan.second_tokens), list(range(64, 100)))
        self.assertEqual(_list(plan.second_cu_seqlens), [0, 36])

    def test_no_split_row_runs_one_pass(self):
        plan = build_gdp_track_plan(
            track_rows=[0],
            track_seqlens=[64],
            extend_lens=[64, 3],
            prefix_lens=[0, 0],
            chunk_size=64,
            cache_indices=torch.tensor([4, 5]),
            track_dst=torch.tensor([8, 9]),
        )
        self.assertIsNone(plan.first_tokens)
        self.assertIsNone(plan.pre_src)
        self.assertEqual((_list(plan.post_src), _list(plan.post_dst)), ([4], [8]))


if __name__ == "__main__":
    unittest.main()
