import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.penalties import PenaltyStatusPool, StepPenalties
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_utils import commit_block_accept_tokens_to_penalties
from sglang.srt.speculative.eagle_utils import commit_accept_tokens_to_penalties
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB_SIZE = 16


def _req(num, **sampling_params):
    return SimpleNamespace(
        rid=f"r{num}",
        penalty_index=None,
        output_ids=[],
        sampling_params=SamplingParams(**sampling_params),
    )


def _info(reqs, pool):
    bs = len(reqs)
    info = SamplingBatchInfo(
        temperatures=torch.ones(bs, 1),
        top_ps=torch.ones(bs),
        top_ks=torch.ones(bs, dtype=torch.int32),
        min_ps=torch.zeros(bs),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB_SIZE,
        device="cpu",
    )
    info.step_penalties = StepPenalties.build(reqs, pool)
    return info


class TestVerifyPenaltyCommit(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.pool = PenaltyStatusPool(VOCAB_SIZE, "cpu")
        self.reqs = [
            _req(1, frequency_penalty=1.0),
            _req(2),
            _req(3, frequency_penalty=1.0),
        ]
        for req in self.reqs:
            self.pool.alloc([req])
            self.pool.reset_history([req])

    def _counts(self):
        additive, _ = _info(self.reqs, self.pool).step_penalties.logit_penalties()
        return (-additive).long()

    def _expected(self, histories):
        return torch.stack(
            [
                torch.bincount(torch.tensor(h, dtype=torch.long), minlength=VOCAB_SIZE)
                for h in histories
            ]
        )

    def test_tree_verify_commits_accept_index_rows(self):
        predict = torch.tensor([4, 4, 6, 1, 2, 3, 5, 9, 9])
        accept_index = torch.tensor([[0, 1, -1], [3, 4, 5], [6, -1, -1]])
        commit_accept_tokens_to_penalties(
            _info(self.reqs, self.pool), predict, accept_index
        )
        torch.testing.assert_close(self._counts(), self._expected([[4, 4], [], [5]]))

    def test_block_verify_commits_the_committed_prefix(self):
        out_tokens = torch.tensor([[4, 4, 6], [1, 2, 3], [5, 9, 9]])
        commit_lens = torch.tensor([2, 3, 1], dtype=torch.int32)
        commit_block_accept_tokens_to_penalties(
            _info(self.reqs, self.pool), out_tokens, commit_lens
        )
        torch.testing.assert_close(self._counts(), self._expected([[4, 4], [], [5]]))


if __name__ == "__main__":
    unittest.main()
