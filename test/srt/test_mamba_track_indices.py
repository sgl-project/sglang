"""set_mamba_track_indices_from_reqs gathers one buffer index per request."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(mapping: torch.Tensor, req_pool_indices: list[int], positions: list[int]):
    reqs = [
        SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=p)) for p in positions
    ]
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
        ),
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        reqs=reqs,
        mamba_track_buffer_indices=None,
        mamba_track_indices=None,
    )


class TestSetMambaTrackIndices(unittest.TestCase):
    def setUp(self):
        # 6 request slots x 3 ping-pong positions of buffer indices.
        self.mapping = torch.arange(18, dtype=torch.int64).reshape(6, 3) * 10

    def test_staged_positions_select_one_buffer_per_request(self):
        req_pool_indices = [4, 0, 5]
        positions = [2, 0, 1]
        batch = _batch(self.mapping, req_pool_indices, positions)
        set_mamba_track_indices_from_reqs(
            batch, positions, torch.tensor(positions, dtype=torch.int32)
        )
        expected = torch.tensor(
            [self.mapping[r, p].item() for r, p in zip(req_pool_indices, positions)],
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(batch.mamba_track_indices, expected))
        self.assertEqual(batch.mamba_track_indices.dtype, torch.int64)
        self.assertEqual(batch.mamba_track_buffer_indices, positions)

    def test_positions_default_to_the_requests_next_track_idx(self):
        req_pool_indices = [1, 3]
        positions = [1, 2]
        batch = _batch(self.mapping, req_pool_indices, positions)
        # Stage the same positions on device so the CPU-only test needs no pinning.
        set_mamba_track_indices_from_reqs(
            batch, None, torch.tensor(positions, dtype=torch.int64)
        )
        expected = torch.tensor(
            [self.mapping[r, p].item() for r, p in zip(req_pool_indices, positions)],
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(batch.mamba_track_indices, expected))
        self.assertEqual(batch.mamba_track_buffer_indices, positions)


if __name__ == "__main__":
    unittest.main()
