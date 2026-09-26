from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")

import unittest

import torch

from sglang.kernels.ops.speculative.temperature_softmax import temperature_softmax
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestTemperatureSoftmax(CustomTestCase):
    @staticmethod
    def temperatures(rows: int) -> torch.Tensor:
        return torch.linspace(
            0.5, 1.5, rows, dtype=torch.float32, device="cuda"
        ).unsqueeze(1)

    def assert_matches_torch(self, logits: torch.Tensor, temperatures: torch.Tensor):
        expected = torch.softmax(logits / temperatures, dim=-1)
        actual = temperature_softmax(logits, temperatures)

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)

    def test_dense_logits(self):
        generator = torch.Generator(device="cuda").manual_seed(0)
        for rows, vocab_size in (
            (0, 1024),
            (1, 0),
            (1, 1024),
            (1, 32000),
            (7, 65537),
            (32, 128257),
            (33, 4096),
        ):
            with self.subTest(rows=rows, vocab_size=vocab_size):
                logits = torch.randn(
                    (rows, vocab_size),
                    dtype=torch.float32,
                    device="cuda",
                    generator=generator,
                )
                self.assert_matches_torch(logits, self.temperatures(rows))

    def test_noncontiguous_logits_fall_back(self):
        logits = torch.randn((4096, 3), dtype=torch.float32, device="cuda").T
        self.assert_matches_torch(logits, self.temperatures(logits.shape[0]))

    def test_sparse_grammar_mask_with_fully_masked_splits(self):
        vocab_size = 65537
        allowed = torch.tensor(
            [0, vocab_size // 2, vocab_size - 1], device="cuda", dtype=torch.long
        )

        for rows in (1, 3, 16, 32):
            with self.subTest(rows=rows):
                logits = torch.full(
                    (rows, vocab_size),
                    -float("inf"),
                    dtype=torch.float32,
                    device="cuda",
                )
                values = torch.arange(
                    rows * allowed.numel(), dtype=torch.float32, device="cuda"
                ).reshape(rows, allowed.numel())
                values = values.remainder(7).sub_(3)
                logits[:, allowed] = values
                self.assert_matches_torch(logits, self.temperatures(rows))


if __name__ == "__main__":
    unittest.main()
