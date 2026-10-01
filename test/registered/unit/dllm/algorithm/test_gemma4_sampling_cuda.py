"""DiffusionGemma denoiser statistics and sampling parity."""

import unittest

import torch

from sglang.srt.dllm.algorithm.gemma4_renoise import (
    _denoiser_statistics,
    _denoiser_statistics_cuda,
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
                actual = _denoiser_statistics_cuda(logits, temperatures)
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

    def test_sharp_full_vocabulary_statistics(self):
        from sglang.kernels.ops.sampling.denoiser_statistics import denoiser_statistics

        generator = torch.Generator(device="cuda").manual_seed(20261001)
        logits = torch.randn(1, 33, 262144, device="cuda", generator=generator) * 30
        temperatures = torch.tensor([0.4], device="cuda")
        expected = _denoiser_statistics(logits, temperatures)
        actual = denoiser_statistics(logits, temperatures)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
        a = torch.Generator(device="cuda").manual_seed(42)
        b = torch.Generator(device="cuda").manual_seed(42)
        torch.testing.assert_close(
            _sample_denoiser(actual[0][0], a), _sample_denoiser(expected[0][0], b)
        )
        torch.testing.assert_close(a.get_state(), b.get_state())

    def test_chunk_boundaries_and_graph_reuse(self):
        from sglang.kernels.ops.sampling.denoiser_statistics import denoiser_statistics

        logits = torch.zeros(2, 33, 65537, device="cuda")
        temperatures = torch.tensor([0.4, 0.8], device="cuda")
        denoiser_statistics(logits, temperatures)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = denoiser_statistics(logits, temperatures)
        for kind in (
            "uniform",
            "ties",
            "masked_chunk",
            "all_masked",
            "infinity",
            "nan",
        ):
            with self.subTest(kind=kind):
                logits.zero_()
                temperatures.copy_(temperatures.flip(0))
                if kind == "ties":
                    logits[:, :, 0] = 5
                    logits[:, :, -1] = 5
                elif kind == "masked_chunk":
                    logits[:, :, :8192] = -float("inf")
                elif kind == "all_masked":
                    logits.fill_(-float("inf"))
                elif kind == "infinity":
                    logits[:, :, 123] = float("inf")
                elif kind == "nan":
                    logits[:, :, 123] = float("nan")
                    logits[:, :, 45000] = float("nan")
                graph.replay()
                expected = _denoiser_statistics(logits, temperatures)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(
                        got, want, rtol=2e-5, atol=2e-6, equal_nan=True
                    )


if __name__ == "__main__":
    unittest.main()
