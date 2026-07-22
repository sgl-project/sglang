import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.dflash_worker_v2 import (
    DFlashWorkerV2,
    _compute_linear_mamba_verify_commit_steps,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashMambaCommit(CustomTestCase):
    def test_commit_steps_cover_real_dflash_boundary_cases(self):
        last_steps, track_steps = _compute_linear_mamba_verify_commit_steps(
            seq_lens_pre_verify=torch.tensor([256, 250, 500], dtype=torch.int32),
            commit_lens=torch.tensor([4, 10, 12], dtype=torch.int32),
            track_interval=256,
        )

        torch.testing.assert_close(last_steps, torch.tensor([3, 9, 11]))
        torch.testing.assert_close(track_steps, torch.tensor([-1, 5, 11]))

    def test_commit_steps_use_post_verify_length(self):
        _, track_steps = _compute_linear_mamba_verify_commit_steps(
            seq_lens_pre_verify=torch.tensor([500, 511], dtype=torch.int64),
            commit_lens=torch.tensor([12, 1], dtype=torch.int32),
            track_interval=256,
        )

        torch.testing.assert_close(track_steps, torch.tensor([11, 0]))

    @patch(
        "sglang.srt.speculative.dflash_worker_v2.get_exec",
        return_value=SimpleNamespace(mamba=SimpleNamespace(mamba_track_interval=256)),
    )
    def test_worker_commits_active_and_radix_track_states(self, _mock_get_exec):
        update_mamba_state = Mock()
        model = object()
        worker = object.__new__(DFlashWorkerV2)
        worker._need_mamba_verify_commit = True
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(
                    update_mamba_state_after_mtp_verify=update_mamba_state
                ),
                model=model,
            )
        )
        track_indices = torch.tensor([7, 8, 9], dtype=torch.int64)
        batch = SimpleNamespace(mamba_track_indices=track_indices)

        worker._update_target_mamba_state_after_verify(
            batch=batch,
            seq_lens_pre_verify=torch.tensor([250, 256, 500], dtype=torch.int32),
            commit_lens=torch.tensor([10, 4, 12], dtype=torch.int32),
        )

        kwargs = update_mamba_state.call_args.kwargs
        torch.testing.assert_close(
            kwargs["last_correct_step_indices"], torch.tensor([9, 3, 11])
        )
        torch.testing.assert_close(
            kwargs["mamba_steps_to_track"], torch.tensor([5, -1, 11])
        )
        self.assertIs(kwargs["mamba_track_indices"], track_indices)
        self.assertIs(kwargs["model"], model)

    def test_worker_without_extra_buffer_still_commits_active_state(self):
        update_mamba_state = Mock()
        worker = object.__new__(DFlashWorkerV2)
        worker._need_mamba_verify_commit = True
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(
                    update_mamba_state_after_mtp_verify=update_mamba_state
                ),
                model=object(),
            )
        )

        worker._update_target_mamba_state_after_verify(
            batch=SimpleNamespace(mamba_track_indices=None),
            seq_lens_pre_verify=torch.tensor([100], dtype=torch.int32),
            commit_lens=torch.tensor([3], dtype=torch.int32),
        )

        kwargs = update_mamba_state.call_args.kwargs
        torch.testing.assert_close(
            kwargs["last_correct_step_indices"], torch.tensor([2])
        )
        self.assertIsNone(kwargs["mamba_steps_to_track"])


if __name__ == "__main__":
    unittest.main()
