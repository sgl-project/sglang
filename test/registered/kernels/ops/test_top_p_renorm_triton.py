"""Correctness and validation tests for the Triton top-p fallback."""

import unittest

import torch

from sglang.kernels.ops.sampling.top_p_renorm_triton import (
    top_p_renorm_probs_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


def _reference(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    probs_fp32 = probs.float()
    sorted_probs = probs_fp32.sort(dim=-1).values
    cdf = sorted_probs.cumsum(dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_p).unsqueeze(1)).squeeze(1)
    cutoff.clamp_(max=probs.shape[1] - 1)
    pivot = sorted_probs.gather(1, cutoff[:, None])
    expected = torch.where(probs_fp32 >= pivot, probs_fp32, 0)
    return expected / expected.sum(dim=-1, keepdim=True)


class TestTopPRenormTriton(CustomTestCase):
    def test_matches_reference(self):
        for batch_size, vocab_size in ((1, 7), (3, 1024), (2, 157184)):
            with self.subTest(batch_size=batch_size, vocab_size=vocab_size):
                torch.manual_seed(batch_size * 1000000 + vocab_size)
                logits = torch.randn(
                    batch_size, vocab_size, device="cuda", dtype=torch.float32
                )
                probs = logits.softmax(dim=-1)
                top_p = torch.linspace(
                    0.5, 0.95, batch_size, device="cuda", dtype=torch.float32
                )

                actual = top_p_renorm_probs_triton(probs, top_p)
                expected = _reference(probs, top_p)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-8)

    def test_scalar_and_tied_probabilities(self):
        probs = torch.tensor(
            [[0.05, 0.05, 0.10, 0.20, 0.20, 0.20, 0.20]],
            device="cuda",
            dtype=torch.float32,
        )
        actual = top_p_renorm_probs_triton(probs, 0.6)
        expected = _reference(
            probs, torch.tensor([0.6], device="cuda", dtype=torch.float32)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(actual).item(), 4)

    def test_empty_inputs(self):
        empty_batch = torch.empty((0, 7), device="cuda")
        empty_vocab = torch.empty((2, 0), device="cuda")
        self.assertEqual(top_p_renorm_probs_triton(empty_batch, 0.9).shape, (0, 7))
        self.assertEqual(top_p_renorm_probs_triton(empty_vocab, 0.9).shape, (2, 0))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "probs must be 2D"):
            top_p_renorm_probs_triton(torch.ones(4, device="cuda"), 0.9)
        with self.assertRaisesRegex(ValueError, r"top_p values must be in \(0, 1\]"):
            top_p_renorm_probs_triton(torch.full((2, 4), 0.25, device="cuda"), 0.0)
        with self.assertRaisesRegex(ValueError, "one value per row"):
            top_p_renorm_probs_triton(
                torch.full((2, 4), 0.25, device="cuda"),
                torch.tensor([0.8, 0.9, 1.0], device="cuda"),
            )


if __name__ == "__main__":
    unittest.main()
