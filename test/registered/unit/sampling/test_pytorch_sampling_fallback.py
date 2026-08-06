"""Unit tests for the top-k sampling fallback in srt/layers/sampler.py — no server, no model loading.

Regression test: the torch fallback must keep every token tied at the k-th
largest probability, matching the CUDA kernels (sgl_kernel.top_k_renorm_prob,
flashinfer). Previously it truncated by rank, which could sample from a
strictly smaller candidate set than the kernel backends.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.test_utils import CustomTestCase


class TestTopKTopPTieHandling(CustomTestCase):

    def _filtered_probs(self, probs, top_ks):
        """Run the fallback and capture the filtered probs passed to multinomial.

        torch.multinomial is patched out so the test asserts deterministically on
        the filtered distribution instead of depending on sampling randomness.
        """
        captured = {}

        def fake_multinomial(input, num_samples, *args, **kwargs):
            captured["filtered"] = input.clone()
            return torch.zeros((input.shape[0], num_samples), dtype=torch.long)

        top_ps = torch.ones(probs.shape[0])
        min_ps = torch.zeros(probs.shape[0])
        positions = torch.arange(probs.shape[0], dtype=torch.long)
        with patch("torch.multinomial", side_effect=fake_multinomial):
            top_k_top_p_min_p_sampling_from_probs_torch(
                probs,
                top_ks,
                top_ps,
                min_ps,
                need_min_p_sampling=False,
                sampling_seed=None,
                positions=positions,
            )
        return captured["filtered"]

    def _kept(self, probs, top_ks):
        filtered = self._filtered_probs(probs, top_ks)
        return [round(v, 6) for v in filtered[filtered > 0].tolist()]

    def _batch(self, rows):
        return torch.tensor(rows, dtype=torch.float32)

    def test_keeps_ties_at_kth_boundary(self):
        probs = self._batch([[0.5, 0.3, 0.1, 0.1]])
        top_ks = torch.tensor([3])
        kept = self._kept(probs, top_ks)
        self.assertEqual(len(kept), 4)
        self.assertEqual(kept, [0.5, 0.3, 0.1, 0.1])

    def test_truncates_when_no_ties(self):
        probs = self._batch([[0.5, 0.3, 0.15, 0.05]])
        top_ks = torch.tensor([3])
        kept = self._kept(probs, top_ks)
        self.assertEqual(len(kept), 3)
        self.assertEqual(kept, [0.5, 0.3, 0.15])

    def test_excludes_ties_below_kth_boundary(self):
        # Ties below the k-th largest value must still be dropped.
        probs = self._batch([[0.7, 0.1, 0.05, 0.05, 0.05]])
        top_ks = torch.tensor([2])
        kept = self._kept(probs, top_ks)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept, [0.7, 0.1])

    def test_top_k_all_keeps_whole_vocab(self):
        probs = self._batch([[0.5, 0.3, 0.1, 0.1]])
        top_ks = torch.tensor([TOP_K_ALL])
        kept = self._kept(probs, top_ks)
        self.assertEqual(len(kept), 4)

    def test_top_k_larger_than_vocab_keeps_all(self):
        probs = self._batch([[0.5, 0.3, 0.1, 0.1]])
        top_ks = torch.tensor([100])
        kept = self._kept(probs, top_ks)
        self.assertEqual(len(kept), 4)

    def test_mixed_batch_per_row_top_k(self):
        probs = self._batch([[0.5, 0.3, 0.1, 0.1], [0.5, 0.3, 0.15, 0.05]])
        top_ks = torch.tensor([3, 2])
        filtered = self._filtered_probs(probs, top_ks)
        row0_kept = (filtered[0] > 0).sum().item()
        row1_kept = (filtered[1] > 0).sum().item()
        self.assertEqual(row0_kept, 4)
        self.assertEqual(row1_kept, 2)


if __name__ == "__main__":
    unittest.main()
