"""DiffusionGemma denoiser statistics and sampling parity."""

import unittest

import torch

from sglang.srt.dllm.algorithm.gemma4_renoise import (
    _compiled_denoiser_statistics,
    _denoiser_statistics,
    _sample_denoiser,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGemma4SamplingCUDA(CustomTestCase):
    def test_full_vocabulary_statistics_and_sampling(self):
        generator = torch.Generator(device="cuda").manual_seed(123)
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                logits = torch.randn(
                    batch_size, 16, 262144, device="cuda", generator=generator
                )
                temperatures = torch.linspace(0.4, 0.8, batch_size, device="cuda")
                expected = _denoiser_statistics(logits, temperatures)
                actual = _compiled_denoiser_statistics(logits, temperatures)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
                a = torch.Generator(device="cuda").manual_seed(42)
                b = torch.Generator(device="cuda").manual_seed(42)
                probabilities = actual[0].reshape(-1, logits.shape[-1])
                torch.testing.assert_close(
                    _sample_denoiser(probabilities, a),
                    torch.multinomial(probabilities, 1, generator=b).squeeze(-1),
                )
                torch.testing.assert_close(a.get_state(), b.get_state())


if __name__ == "__main__":
    unittest.main()
