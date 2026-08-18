import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ReqToTokenPool:
    def __init__(self):
        self.seen_req_indices = None

    def get_mamba_indices(self, req_indices):
        self.seen_req_indices = req_indices.clone()
        return req_indices.to(torch.int32) + 100

    def translate_mamba_indices(self, mamba_indices):
        return mamba_indices


def _batch():
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([10, 20], dtype=torch.int64),
        req_pool_indices=torch.tensor([3, 7], dtype=torch.int64),
        mamba_track_indices=None,
    )


def _worker(*, pp_is_last_rank=False, need_commit=True):
    worker = object.__new__(DSparkWorkerV2)
    worker._pp_enabled = True
    worker._pp_is_last_rank = pp_is_last_rank
    worker._need_mamba_verify_commit = need_commit
    pool = _ReqToTokenPool()
    worker.model_runner = SimpleNamespace(req_to_token_pool=pool)
    worker._commit_target_mamba_states_after_verify = Mock()
    return worker, pool


class TestDSparkPPMambaCommit(CustomTestCase):
    def test_non_last_rank_commits_with_explicit_state_indices(self):
        worker, pool = _worker()
        batch = _batch()

        worker.commit_pp_mamba_states_after_verify(
            batch=batch,
            commit_lens=torch.tensor([3, 5], dtype=torch.int64),
        )

        torch.testing.assert_close(
            pool.seen_req_indices, torch.tensor([3, 7], dtype=torch.int64)
        )
        kwargs = worker._commit_target_mamba_states_after_verify.call_args.kwargs
        torch.testing.assert_close(
            kwargs["seq_lens_pre_verify"], torch.tensor([10, 20])
        )
        torch.testing.assert_close(
            kwargs["seq_lens_post_verify"], torch.tensor([13, 25])
        )
        torch.testing.assert_close(
            kwargs["state_indices_tensor"], torch.tensor([103, 107], dtype=torch.int32)
        )
        torch.testing.assert_close(
            kwargs["conv_source_indices_tensor"],
            torch.tensor([3, 7], dtype=torch.int64),
        )

    def test_last_rank_skips_deferred_commit(self):
        worker, pool = _worker(pp_is_last_rank=True)

        worker.commit_pp_mamba_states_after_verify(
            batch=_batch(),
            commit_lens=torch.tensor([3, 5], dtype=torch.int64),
        )

        self.assertIsNone(pool.seen_req_indices)
        worker._commit_target_mamba_states_after_verify.assert_not_called()

    def test_non_mamba_rank_is_noop(self):
        worker, pool = _worker(need_commit=False)

        worker.commit_pp_mamba_states_after_verify(
            batch=_batch(),
            commit_lens=torch.tensor([3, 5], dtype=torch.int64),
        )

        self.assertIsNone(pool.seen_req_indices)
        worker._commit_target_mamba_states_after_verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
