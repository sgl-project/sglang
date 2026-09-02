"""Unit tests for deterministic sampling behavior in srt/layers/sampler.py."""

import unittest

import torch

from sglang.srt.layers.sampler import (
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestSeededMinPSampling(CustomTestCase):
    def test_seeded_min_p_sampling_is_reproducible_and_filtered(self):
        batch_size = 128
        device = "cuda"
        probs = torch.tensor([[0.05, 0.55, 0.10, 0.30]], device=device).repeat(
            batch_size, 1
        )
        kwargs = dict(
            probs=probs,
            top_ks=torch.full((batch_size,), 4, dtype=torch.int32, device=device),
            top_ps=torch.ones(batch_size, device=device),
            min_ps=torch.full((batch_size,), 0.5, device=device),
            need_min_p_sampling=True,
            sampling_seed=torch.arange(batch_size, dtype=torch.int64, device=device),
            positions=torch.full((batch_size,), 17, dtype=torch.int64, device=device),
        )

        first = top_k_top_p_min_p_sampling_from_probs_torch(**kwargs)
        second = top_k_top_p_min_p_sampling_from_probs_torch(**kwargs)

        self.assertTrue(torch.equal(first, second))
        # min_p=0.5 retains only probabilities >= 0.5 * max_prob. The
        # surviving original token IDs are 1 and 3, despite sorting internally.
        self.assertEqual(set(first.tolist()), {1, 3})


if __name__ == "__main__":
    unittest.main()
