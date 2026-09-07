import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashMambaCheckpointBoundary(CustomTestCase):
    def test_tracks_checkpoints_from_post_verify_lengths(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        update_state = MagicMock()
        worker = SimpleNamespace(
            _need_mamba_verify_commit=True,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    attn_backend=SimpleNamespace(
                        update_mamba_state_after_mtp_verify=update_state
                    ),
                    model=MagicMock(),
                )
            ),
        )
        for dtype in (torch.int32, torch.int64):
            for interval in (64, 128, 256):
                with self.subTest(dtype=dtype, interval=interval):
                    batch = SimpleNamespace(
                        seq_lens=torch.tensor(
                            [interval - 1, interval - 6, interval, interval - 2],
                            dtype=dtype,
                        ),
                        mamba_track_indices=torch.tensor(
                            [7, 8, 9, 10], dtype=torch.int64
                        ),
                        tree_cache=SimpleNamespace(page_size=1),
                        req_pool_indices=torch.tensor([3, 4, 5, 6], dtype=torch.int64),
                    )

                    update_state.reset_mock()
                    with patch(
                        "sglang.srt.speculative.dflash_worker_v2.mamba_track_grid",
                        return_value=interval,
                    ):
                        DFlashWorkerV2._update_target_mamba_state_after_verify(
                            worker,
                            batch=batch,
                            seq_lens_pre_verify=batch.seq_lens.clone(),
                            seq_lens_post_verify=torch.tensor(
                                [interval, interval + 2, interval + 1, interval - 1],
                                dtype=dtype,
                            ),
                            commit_lens=torch.tensor([1, 8, 1, 1], dtype=dtype),
                        )

                    call_args = update_state.call_args.kwargs
                    self.assertTrue(
                        torch.equal(
                            call_args["mamba_steps_to_track"],
                            torch.tensor([0, 5, -1, -1], dtype=torch.int64),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            call_args["last_correct_step_indices"],
                            torch.tensor([0, 7, 0, 0], dtype=torch.int64),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
