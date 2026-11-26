"""Check the host snapshot of an attention-DP exchange."""

import unittest
from dataclasses import FrozenInstanceError, replace

import torch

from sglang.srt.model_executor.mlp_sync_metadata import MLPSyncBatchMetadata
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class TestMLPSyncBatchMetadata(unittest.TestCase):
    def test_mixed_cohort_is_an_independent_immutable_snapshot(self):
        rows = torch.zeros((3, 16), dtype=torch.int64)
        rows[:, 0] = torch.tensor([128, 4, 0])
        rows[:, 5] = torch.tensor([3, 2, 4])
        rows[:, -3:] = torch.tensor([[8, 2, 4096], [4, 4, 8193], [0, 0, 0]])
        snapshot = MLPSyncBatchMetadata.from_exchange(rows)
        rows.zero_()

        self.assertEqual(snapshot.forward_modes, (3, 2, 4))
        self.assertEqual(snapshot.num_tokens, (128, 4, 0))
        self.assertEqual(snapshot.num_requests, (8, 4, 0))
        self.assertEqual(snapshot.num_decode_requests, (2, 4, 0))
        self.assertEqual(snapshot.max_prefix_tokens, (4096, 8193, 0))
        self.assertEqual(replace(snapshot), snapshot)
        with self.assertRaises(FrozenInstanceError):
            snapshot.num_requests = ()


if __name__ == "__main__":
    unittest.main()
