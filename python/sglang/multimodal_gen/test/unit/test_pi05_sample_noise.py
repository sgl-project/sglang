"""Regression test for Pi0.5 request generators without loading weights."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class TestPi05SampleNoise(CustomTestCase):
    def test_request_generators(self):
        from sglang.multimodal_gen.runtime.models.vlas.pi05_policy import (
            Pi05PolicyModel,
        )

        # sample_noise only uses config and device; no checkpoint is needed.
        policy = Pi05PolicyModel.__new__(Pi05PolicyModel)
        torch.nn.Module.__init__(policy)
        policy.config = SimpleNamespace(action_horizon=4, action_dim=8)
        policy.device = torch.device("cuda")

        def sample(device, seed):
            generator = torch.Generator(device=device).manual_seed(seed)
            noise = policy.sample_noise(2, generator=generator)
            self.assertEqual(noise.device.type, "cuda")
            self.assertEqual(noise.shape, (2, 4, 8))
            self.assertEqual(noise.dtype, torch.float32)
            return noise

        first = sample("cpu", 42)
        self.assertTrue(torch.equal(first, sample("cpu", 42)))
        self.assertFalse(torch.equal(first, sample("cpu", 43)))
        sample("cuda", 42)


if __name__ == "__main__":
    unittest.main()
