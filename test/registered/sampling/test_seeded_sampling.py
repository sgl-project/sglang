"""Per-request sampling seeds in the torch sampler."""

import unittest

import torch

from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

VOCAB_SIZE = 4096
BATCH_SIZE = 8


class TestSeededSampling(CustomTestCase):
    def test_seeded_min_p_matches_renormalized_probs(self):
        """Seeded Gumbel-max sampling is invariant to a row's scale, so min-p's
        unnormalized weights draw what their renormalized distribution draws."""
        gen = torch.Generator(device="cuda").manual_seed(0)
        probs = torch.softmax(
            torch.randn(BATCH_SIZE, VOCAB_SIZE, generator=gen, device="cuda") * 3.0,
            dim=-1,
        )
        min_ps = torch.tensor([0.0, 0.01, 0.05, 0.1] * 2, device="cuda")
        seeds = torch.arange(BATCH_SIZE, dtype=torch.int64, device="cuda") * 7919
        positions = torch.arange(BATCH_SIZE, dtype=torch.int64, device="cuda") + 100
        no_top_k = torch.full(
            (BATCH_SIZE,), VOCAB_SIZE, dtype=torch.int32, device="cuda"
        )
        no_top_p = torch.ones(BATCH_SIZE, device="cuda")

        min_p_ids = top_k_top_p_min_p_sampling_from_probs_torch(
            probs.clone(),
            no_top_k,
            no_top_p,
            min_ps,
            True,
            seeds,
            positions,
        )
        threshold = probs.max(dim=-1, keepdim=True).values * min_ps[:, None]
        kept = probs * (probs >= threshold)
        renormalized_ids = top_k_top_p_min_p_sampling_from_probs_torch(
            kept / kept.sum(dim=-1, keepdim=True),
            no_top_k,
            no_top_p,
            torch.zeros(BATCH_SIZE, device="cuda"),
            False,
            seeds,
            positions,
        )
        torch.testing.assert_close(min_p_ids, renormalized_ids, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
