"""Unit tests for the threshold-based top-k / top-p renormalization."""

import unittest

import torch

from sglang.kernels.ops.sampling.renorm_threshold import (
    top_k_renorm_prob,
    top_p_renorm_prob,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _scatter_reference(
    probs: torch.Tensor,
    top_ks: torch.Tensor = None,
    top_ps: torch.Tensor = None,
) -> torch.Tensor:
    """The sorted-mask + scatter form the threshold code replaces."""
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    if top_ks is not None:
        ranks = torch.arange(probs.shape[-1], device=probs.device).view(1, -1)
        sorted_probs = sorted_probs.masked_fill(ranks >= top_ks.view(-1, 1), 0.0)
    if top_ps is not None:
        cumulative = sorted_probs.cumsum(dim=-1)
        sorted_probs = sorted_probs.masked_fill(
            cumulative - sorted_probs > top_ps.view(-1, 1), 0.0
        )
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)


def _random_probs(batch: int, vocab: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20261110)
    logits = torch.randn(batch, vocab, generator=generator) * 3.0
    return torch.softmax(logits, dim=-1)


class TestThresholdRenorm(unittest.TestCase):
    def setUp(self):
        self.probs = _random_probs(batch=8, vocab=1024)

    def test_top_k_matches_scatter_form(self):
        batch = self.probs.shape[0]
        for k in (1, 2, 16, 512, 1024):
            with self.subTest(k=k):
                top_ks = torch.full((batch,), k, dtype=torch.long)
                expected = _scatter_reference(self.probs, top_ks=top_ks)
                actual = top_k_renorm_prob(self.probs, top_ks)
                self.assertLess((actual - expected).abs().max().item(), 1e-6)
                torch.testing.assert_close(
                    actual.sum(dim=-1), torch.ones(batch), atol=1e-6, rtol=0
                )

    def test_top_p_matches_scatter_form(self):
        batch = self.probs.shape[0]
        for p in (0.1, 0.5, 0.9, 1.0):
            with self.subTest(p=p):
                top_ps = torch.full((batch,), p)
                expected = _scatter_reference(self.probs, top_ps=top_ps)
                actual = top_p_renorm_prob(self.probs, top_ps)
                self.assertLess((actual - expected).abs().max().item(), 1e-6)
                torch.testing.assert_close(
                    actual.sum(dim=-1), torch.ones(batch), atol=1e-6, rtol=0
                )

    def test_scalar_thresholds(self):
        top_k = top_k_renorm_prob(self.probs, 32)
        top_k_batch = top_k_renorm_prob(
            self.probs, torch.full((self.probs.shape[0],), 32, dtype=torch.long)
        )
        torch.testing.assert_close(top_k, top_k_batch)

        top_p = top_p_renorm_prob(self.probs, 0.8)
        top_p_batch = top_p_renorm_prob(
            self.probs, torch.full((self.probs.shape[0],), 0.8)
        )
        torch.testing.assert_close(top_p, top_p_batch)

    def test_input_is_not_mutated(self):
        probs = self.probs.clone()
        top_k_renorm_prob(probs, 64)
        top_p_renorm_prob(probs, 0.7)
        torch.testing.assert_close(probs, self.probs)

    def test_top_k_out_of_range_is_clamped(self):
        batch = self.probs.shape[0]
        clamped = top_k_renorm_prob(self.probs, torch.zeros(batch, dtype=torch.long))
        single = top_k_renorm_prob(self.probs, torch.ones(batch, dtype=torch.long))
        torch.testing.assert_close(clamped, single)


if __name__ == "__main__":
    unittest.main()
