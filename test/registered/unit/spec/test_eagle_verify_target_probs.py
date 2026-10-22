"""Unit tests for the EAGLE verify target distribution (_filter_verify_target_probs).

Speculative sampling is lossless only if every verify row's target
distribution is the one the sampler draws from without speculation. These
tests compare it with the pytorch sampling backend's filtered weights
(top_k_top_p_min_p_sampling_from_probs_torch): the top-k and top-p cuts are
joint, min-p is relative to the most likely token, and rows without a filter
keep their probabilities.

Requires a GPU: the top-k / top-p renorms are FlashInfer kernels.
"""

import unittest
from types import SimpleNamespace

import torch
from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.srt.speculative.eagle_utils import _filter_verify_target_probs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

DRAFT_TOKEN_NUM = 2
VOCAB = 8
PROBS = [0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05]


def _sampling_info(top_ks, top_ps, min_ps):
    return SimpleNamespace(
        top_ks=torch.tensor(top_ks, dtype=torch.int32, device="cuda"),
        top_ps=torch.tensor(top_ps, dtype=torch.float32, device="cuda"),
        min_ps=torch.tensor(min_ps, dtype=torch.float32, device="cuda"),
        need_top_k_sampling=any(k < VOCAB for k in top_ks),
        need_top_p_sampling=any(p < 1.0 for p in top_ps),
        need_min_p_sampling=any(p > 0.0 for p in min_ps),
    )


def _filtered(info, probs):
    return _filter_verify_target_probs(
        probs,
        info,
        DRAFT_TOKEN_NUM,
        top_k_renorm_prob=top_k_renorm_probs,
        top_p_renorm_prob=top_p_renorm_probs,
    )


def _sampler_distribution(info, probs):
    """The normalized weights the pytorch backend samples each row from."""

    def per_row(values):
        return torch.repeat_interleave(values, DRAFT_TOKEN_NUM, dim=0)

    _, weights, token_ids, _ = top_k_top_p_min_p_sampling_from_probs_torch(
        probs.clone(),
        per_row(info.top_ks),
        per_row(info.top_ps),
        per_row(info.min_ps),
        info.need_min_p_sampling,
        None,
        torch.zeros(probs.shape[0], dtype=torch.int64, device="cuda"),
        return_filtered_probs=True,
    )
    dist = torch.zeros_like(probs).scatter_(1, token_ids.long(), weights)
    return dist / dist.sum(dim=-1, keepdim=True)


class TestVerifyTargetProbs(CustomTestCase):
    def _probs(self, num_reqs):
        return torch.tensor(
            [PROBS] * (num_reqs * DRAFT_TOKEN_NUM), dtype=torch.float32, device="cuda"
        )

    def test_matches_the_sampler_for_every_filter(self):
        # request 0: top-k and top-p; 1: min-p; 2: top-k, top-p and min-p; 3: none
        info = _sampling_info(
            top_ks=[3, VOCAB, 4, VOCAB],
            top_ps=[0.6, 1.0, 0.7, 1.0],
            min_ps=[0.0, 0.45, 0.4, 0.0],
        )
        probs = self._probs(4)
        got = _filtered(info, probs)
        torch.testing.assert_close(got, _sampler_distribution(info, probs))
        torch.testing.assert_close(got[6:], probs[6:])  # no filter: unchanged

    def test_top_p_cuts_the_unfiltered_probabilities(self):
        # The top-3 tokens hold 0.65 >= top_p, so the sampler keeps all three.
        # A top-p cut after the top-k renorm (0.46, 0.31, 0.23) keeps two.
        info = _sampling_info(top_ks=[3], top_ps=[0.6], min_ps=[0.0])
        got = _filtered(info, self._probs(1))
        self.assertEqual((got[0] > 0).nonzero().flatten().tolist(), [0, 1, 2])
        torch.testing.assert_close(
            got[0, :3], torch.tensor([0.3, 0.2, 0.15], device="cuda") / 0.65
        )

    def test_min_p_drops_tokens_below_its_share_of_the_top_token(self):
        info = _sampling_info(top_ks=[VOCAB], top_ps=[1.0], min_ps=[0.45])
        got = _filtered(info, self._probs(1))
        # threshold 0.45 * 0.3 = 0.135: tokens 0, 1, 2 stay
        self.assertEqual((got[1] > 0).nonzero().flatten().tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
