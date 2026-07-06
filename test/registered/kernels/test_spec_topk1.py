from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.speculative.triton_ops.topk1 import (
    draft_topk1_postprocess,
    select_top_k_tokens_topk1_later,
)
from sglang.test.test_utils import CustomTestCase


def _make_logits_with_unique_argmax(
    batch_size: int,
    vocab_size: int,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        (batch_size, vocab_size), dtype=torch.float32, device=device, generator=g
    )
    expected_index = (
        torch.arange(batch_size, dtype=torch.long, device=device) * 9973 + 17
    ) % vocab_size
    logits.scatter_(1, expected_index[:, None], 1000.0)
    return logits, expected_index[:, None]


def _ref_select_topk1_later(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
):
    next_scores = scores * topk_p
    parents = torch.full_like(topk_index, i)
    tree_info = (next_scores.unsqueeze(1), topk_index, parents)
    return topk_index.flatten(), hidden_states, next_scores, tree_info


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestSpecTopk1Triton(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def test_draft_topk1_postprocess_matches_argmax_and_position_add(self):
        configs = [
            (1, 127),
            (4, 8192),
            (7, 8193),
            (3, 50000),
        ]
        for batch_size, vocab_size in configs:
            with self.subTest(batch_size=batch_size, vocab_size=vocab_size):
                logits, expected_index = _make_logits_with_unique_argmax(
                    batch_size, vocab_size, device=self.device, seed=vocab_size
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

    def test_select_topk1_later_matches_eager_reference(self):
        configs = [
            (1, 1),
            (8, 1),
            (33, 2),
        ]
        for batch_size, stride in configs:
            with self.subTest(batch_size=batch_size, stride=stride):
                topk_p = torch.rand(
                    (batch_size, 1), dtype=torch.float32, device=self.device
                )
                base_index = torch.arange(
                    batch_size * (stride + 1), dtype=torch.long, device=self.device
                ).view(batch_size, stride + 1)
                topk_index = base_index[:, :1]
                hidden_states = torch.randn(
                    (batch_size, 16), dtype=torch.float32, device=self.device
                )
                scores = torch.rand(
                    (batch_size, 1), dtype=torch.float32, device=self.device
                )

                got = select_top_k_tokens_topk1_later(
                    2, topk_p, topk_index, hidden_states, scores
                )
                ref = _ref_select_topk1_later(
                    2, topk_p, topk_index, hidden_states, scores
                )

                torch.testing.assert_close(got[0], ref[0], rtol=0, atol=0)
                self.assertIs(got[1], hidden_states)
                torch.testing.assert_close(got[2], ref[2], rtol=0, atol=0)
                for got_tree, ref_tree in zip(got[3], ref[3]):
                    torch.testing.assert_close(got_tree, ref_tree, rtol=0, atol=0)

    def test_select_topk1_later_materializes_draft_token_columns(self):
        batch_size = 17
        steps = 3
        hidden_states = torch.randn(
            (batch_size, 16), dtype=torch.float32, device=self.device
        )
        first_base = torch.arange(
            100, 100 + batch_size * 2, dtype=torch.long, device=self.device
        ).view(batch_size, 2)
        first_topk_index = first_base[:, :1]
        topk_indices = [
            first_topk_index,
            first_topk_index + 1000,
            first_topk_index + 2000,
        ]
        topk_ps = [
            torch.rand((batch_size, 1), dtype=torch.float32, device=self.device)
            for _ in range(steps)
        ]

        draft_tokens = torch.full(
            (batch_size, steps), -1, dtype=torch.long, device=self.device
        )
        scores = topk_ps[0]
        _, _, scores, tree_info = select_top_k_tokens_topk1_later(
            1,
            topk_ps[1],
            topk_indices[1],
            hidden_states,
            scores,
            draft_tokens,
            first_topk_index,
        )
        torch.testing.assert_close(
            tree_info[2], torch.full_like(topk_indices[1], 1), rtol=0, atol=0
        )
        _, _, scores, tree_info = select_top_k_tokens_topk1_later(
            2,
            topk_ps[2],
            topk_indices[2],
            hidden_states,
            scores,
            draft_tokens,
        )

        expected = torch.cat(topk_indices, dim=1)
        torch.testing.assert_close(draft_tokens, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            scores, topk_ps[0] * topk_ps[1] * topk_ps[2], rtol=0, atol=0
        )
        torch.testing.assert_close(
            tree_info[2], torch.full_like(topk_indices[2], 2), rtol=0, atol=0
        )

    def test_empty_batch(self):
        hidden_states = torch.empty((0, 16), dtype=torch.float32, device=self.device)
        topk_p = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        topk_index = torch.empty((0, 1), dtype=torch.long, device=self.device)
        scores = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        draft_tokens = torch.empty((0, 3), dtype=torch.long, device=self.device)

        got = select_top_k_tokens_topk1_later(
            1, topk_p, topk_index, hidden_states, scores, draft_tokens, topk_index
        )
        self.assertEqual(got[0].numel(), 0)
        self.assertEqual(got[2].numel(), 0)
        self.assertEqual(draft_tokens.numel(), 0)


if __name__ == "__main__":
    unittest.main()
