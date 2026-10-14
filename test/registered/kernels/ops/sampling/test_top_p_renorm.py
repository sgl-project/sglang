"""Top-p probability renormalization parity."""

import unittest

import torch

from sglang.kernels.ops.sampling.top_p_renorm_triton import (
    top_p_renorm_probs_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestTopPRenorm(CustomTestCase):
    def test_top_p_renorm(self):
        torch.manual_seed(3)
        probs = torch.randn(3, 1024, device="cuda").softmax(-1)
        top_p = torch.tensor([0.5, 0.8, 0.95], device="cuda")
        sorted_probs = probs.sort(-1).values
        cutoff = torch.searchsorted(
            sorted_probs.cumsum(-1), (1 - top_p).unsqueeze(1)
        ).squeeze(1)
        cutoff.clamp_(max=probs.shape[1] - 1)
        pivot = sorted_probs.gather(1, cutoff[:, None])
        expected = torch.where(probs >= pivot, probs, 0)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(
            top_p_renorm_probs_triton(probs, top_p),
            expected,
            rtol=2e-6,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
