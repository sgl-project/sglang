"""The u == 1.0 gumbel bucket must neither go +inf (NaN -> samples vocab-masked
tokens) nor exceed the hash spacing's natural maximum (dominates every row)."""

import unittest

import torch

from sglang.srt.layers.sampler import sampling_from_probs_torch
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=14, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

VOCAB = 248320
CUTOFF = 248077
# murmur_hash32(seed, position, 248146) == 0xFFFFFFFF
SEED, POSITION = 6469398791980356130, 7371
U1_COLUMN = 248146


def _sample(logits: torch.Tensor) -> int:
    probs = torch.softmax(logits, dim=-1)
    return int(
        sampling_from_probs_torch(
            probs,
            sampling_seed=torch.tensor([SEED], device="cuda"),
            positions=torch.tensor([POSITION], device="cuda"),
        ).item()
    )


class TestDeterministicGumbelU1(CustomTestCase):
    def test_never_samples_masked_token(self):
        torch.manual_seed(0)
        logits = torch.randn(1, VOCAB, device="cuda", dtype=torch.float32) * 4
        logits[:, CUTOFF:] = float("-inf")
        self.assertLess(_sample(logits), CUTOFF)

    def test_u1_bucket_does_not_dominate(self):
        # -40 keeps the column's softmax prob representable in fp32 (~1.7e-23);
        # anything much lower underflows to 0 and the column is effectively masked
        logits = torch.zeros(1, VOCAB, device="cuda", dtype=torch.float32)
        logits[:, U1_COLUMN] = -40.0
        self.assertNotEqual(
            _sample(logits), U1_COLUMN, "u==1 gumbel outlier overrode a ~-52 logprob"
        )


if __name__ == "__main__":
    unittest.main()
