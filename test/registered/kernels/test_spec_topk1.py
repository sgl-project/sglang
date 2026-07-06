from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.speculative.triton_ops.topk1 import draft_topk1_postprocess
from sglang.test.test_utils import CustomTestCase


def _make_logits_with_unique_argmax(
    batch_size: int,
    vocab_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        (batch_size, vocab_size), dtype=dtype, device=device, generator=g
    )
    expected_index = (
        torch.arange(batch_size, dtype=torch.long, device=device) * 9973 + 17
    ) % vocab_size
    logits.scatter_(1, expected_index[:, None], 1000.0)
    return logits, expected_index[:, None]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestSpecTopk1Triton(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def test_draft_topk1_postprocess_matches_argmax_and_position_add(self):
        configs = [
            (1, 127, torch.float32),
            (4, 8192, torch.float16),
            (7, 8193, torch.bfloat16),
            (3, 50000, torch.float32),
        ]
        for batch_size, vocab_size, dtype in configs:
            with self.subTest(
                batch_size=batch_size, vocab_size=vocab_size, dtype=dtype
            ):
                logits, expected_index = _make_logits_with_unique_argmax(
                    batch_size,
                    vocab_size,
                    dtype=dtype,
                    device=self.device,
                    seed=vocab_size,
                )
                positions = torch.arange(
                    batch_size, dtype=torch.long, device=self.device
                )
                expected_positions = positions + 1

                topk_p, topk_index = draft_topk1_postprocess(logits, positions)

                torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
                torch.testing.assert_close(
                    topk_p,
                    torch.ones((batch_size, 1), dtype=torch.float32, device=self.device),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    positions, expected_positions, rtol=0, atol=0
                )

    def test_draft_topk1_postprocess_can_write_draft_token_column(self):
        batch_size = 17
        vocab_size = 4097
        logits, expected_index = _make_logits_with_unique_argmax(
            batch_size,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=0,
        )
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        backing = torch.full(
            (batch_size, 5), -1, dtype=torch.long, device=self.device
        )
        draft_tokens = backing[:, 1:4]

        topk_p, topk_index = draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column=2
        )

        torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        torch.testing.assert_close(
            draft_tokens[:, 2], expected_index[:, 0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            draft_tokens[:, :2],
            torch.full_like(draft_tokens[:, :2], -1),
            rtol=0,
            atol=0,
        )

    def test_empty_batch(self):
        logits = torch.empty((0, 1024), dtype=torch.float32, device=self.device)
        positions = torch.empty((0,), dtype=torch.long, device=self.device)
        draft_tokens = torch.empty((0, 3), dtype=torch.long, device=self.device)

        topk_p, topk_index = draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column=1
        )

        self.assertEqual(topk_p.shape, (0, 1))
        self.assertEqual(topk_index.shape, (0, 1))
        self.assertEqual(draft_tokens.numel(), 0)

    def test_non_contiguous_inputs_raise(self):
        logits = torch.empty((16, 4), dtype=torch.float32, device=self.device).t()
        positions = torch.arange(8, dtype=torch.long, device=self.device)[::2]

        with self.assertRaises(AssertionError):
            draft_topk1_postprocess(logits, torch.empty(4, dtype=torch.long, device=self.device))
        with self.assertRaises(AssertionError):
            draft_topk1_postprocess(torch.empty((4, 16), device=self.device), positions)


if __name__ == "__main__":
    unittest.main()
