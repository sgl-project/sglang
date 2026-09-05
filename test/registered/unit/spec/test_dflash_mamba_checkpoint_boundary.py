import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashMambaCheckpointBoundary(CustomTestCase):
    def test_tracks_checkpoint_crossed_by_committed_tokens(self):
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
        batch = SimpleNamespace(
            seq_lens=torch.tensor([255], dtype=torch.int32),
            mamba_track_indices=torch.tensor([7], dtype=torch.int64),
            tree_cache=SimpleNamespace(page_size=1),
            req_pool_indices=torch.tensor([3], dtype=torch.int64),
        )

        with patch(
            "sglang.srt.speculative.dflash_worker_v2.mamba_track_grid",
            return_value=256,
        ):
            DFlashWorkerV2._update_target_mamba_state_after_verify(
                worker,
                batch=batch,
                seq_lens_pre_verify=batch.seq_lens.clone(),
                commit_lens=torch.tensor([1], dtype=torch.int32),
            )

        self.assertTrue(
            torch.equal(
                update_state.call_args.kwargs["mamba_steps_to_track"],
                torch.tensor([0], dtype=torch.int64),
            )
        )


if __name__ == "__main__":
    unittest.main()
