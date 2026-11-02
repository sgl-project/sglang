import unittest

import torch

from sglang.srt.layers.attention.mamba.replay_state_indices_validator import (
    validate_replay_state_indices_cpu,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestReplayStateIndicesValidator(unittest.TestCase):
    def test_valid_unique_live_slots_and_padding(self):
        validate_replay_state_indices_cpu(
            torch.tensor([0, 2, 9, -1, -1], dtype=torch.int32),
            valid_bs=3,
            total_bs=5,
            num_state_slots=10,
        )

    def test_rejects_duplicate_live_slot(self):
        with self.assertRaisesRegex(AssertionError, r"duplicate_slots=\[7\]"):
            validate_replay_state_indices_cpu(
                torch.tensor([7, 2, 7, -1], dtype=torch.int32),
                valid_bs=3,
                total_bs=4,
                num_state_slots=10,
            )

    def test_rejects_out_of_range_live_slots(self):
        for bad_slot in (-1, -2, 10):
            with self.subTest(bad_slot=bad_slot):
                with self.assertRaisesRegex(AssertionError, "live rows"):
                    validate_replay_state_indices_cpu(
                        torch.tensor([3, bad_slot, -1], dtype=torch.int64),
                        valid_bs=2,
                        total_bs=3,
                        num_state_slots=10,
                    )

    def test_rejects_non_sentinel_padding(self):
        with self.assertRaisesRegex(AssertionError, "padded rows"):
            validate_replay_state_indices_cpu(
                torch.tensor([3, 5, 5], dtype=torch.int32),
                valid_bs=2,
                total_bs=3,
                num_state_slots=10,
            )

    def test_requires_cpu_tensor(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        with self.assertRaisesRegex(ValueError, "copied to CPU"):
            validate_replay_state_indices_cpu(
                torch.tensor([1], dtype=torch.int32, device="cuda"),
                valid_bs=1,
                total_bs=1,
                num_state_slots=2,
            )


if __name__ == "__main__":
    unittest.main()
