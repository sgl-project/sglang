"""Seeded min-p sampling draws from the renormalized min-p distribution.

The seeded draw is a Gumbel-max over log weights, so scaling a row by a constant
does not change the argmax and the min-p filtered weights need no renormalizing.
"""

import unittest

import torch

from sglang.srt.layers.sampler import (
    multinomial_with_seed,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=16, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

VOCAB = 4096
ROWS = 4096
MIN_P = 0.3


def _seeded_min_p_sample(
    probs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    rows, vocab = probs.shape
    return top_k_top_p_min_p_sampling_from_probs_torch(
        probs.clone(),
        torch.full((rows,), vocab, dtype=torch.int32, device="cuda"),
        torch.ones(rows, device="cuda"),
        torch.full((rows,), MIN_P, device="cuda"),
        True,
        seeds,
        positions,
    )


def _min_p_filtered(probs: torch.Tensor) -> torch.Tensor:
    thresholds = probs.max(dim=-1, keepdim=True).values * MIN_P
    return probs.masked_fill(probs < thresholds, 0.0)


class TestSeededMinPSampling(CustomTestCase):
    def test_matches_a_seeded_draw_from_renormalized_weights(self):
        generator = torch.Generator(device="cpu").manual_seed(0)
        probs = torch.softmax(
            torch.randn(ROWS, VOCAB, generator=generator) * 4, dim=-1
        ).cuda()
        seeds = torch.arange(1, ROWS + 1, device="cuda")
        positions = torch.arange(ROWS, device="cuda")

        filtered = _min_p_filtered(probs)
        renormalized = filtered / filtered.sum(dim=-1, keepdim=True)
        sorted_weights, sorted_token_ids = renormalized.sort(dim=-1, descending=True)
        sorted_index = multinomial_with_seed(
            sorted_weights.to(torch.float64).log(), seeds, positions
        )
        expected = torch.gather(sorted_token_ids, 1, sorted_index).view(-1)

        actual = _seeded_min_p_sample(probs, seeds, positions)
        self.assertTrue(torch.equal(actual.long(), expected))

    def test_draws_follow_the_min_p_distribution(self):
        probs = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05], device="cuda")
        draws = 200_000
        token_ids = _seeded_min_p_sample(
            probs.expand(draws, -1).contiguous(),
            torch.arange(draws, device="cuda"),
            torch.zeros(draws, dtype=torch.int64, device="cuda"),
        )

        frequencies = torch.bincount(token_ids.long(), minlength=5).double() / draws
        filtered = _min_p_filtered(probs).double()
        target = filtered / filtered.sum()
        self.assertEqual(frequencies[3:].tolist(), [0.0, 0.0])
        self.assertLess((frequencies - target).abs().max().item(), 0.01)


if __name__ == "__main__":
    unittest.main()
