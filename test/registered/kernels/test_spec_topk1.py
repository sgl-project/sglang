from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func
from sglang.kernels.ops.speculative.topk1 import (
    TargetVerifyTopk1Output,
    draft_topk1_postprocess,
    target_verify_topk1_postprocess,
)
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
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


def _make_topk1_chain(batch_size: int, num_tokens: int, device: torch.device):
    retrieve_index = torch.arange(
        batch_size * num_tokens, dtype=torch.long, device=device
    ).view(batch_size, num_tokens)
    retrieve_next_token = torch.arange(
        1, num_tokens + 1, dtype=torch.long, device=device
    ).repeat(batch_size, 1)
    retrieve_next_token[:, -1] = -1
    return retrieve_index, retrieve_next_token


def _eager_target_verify_topk1(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    seq_lens: torch.Tensor,
) -> TargetVerifyTopk1Output:
    batch_size, num_tokens = candidates.shape
    predict = torch.zeros(
        (batch_size * num_tokens,), dtype=torch.int32, device=logits.device
    )
    accept_index = torch.full(
        (batch_size, num_tokens), -1, dtype=torch.int32, device=logits.device
    )
    num_correct_drafts = torch.empty(
        (batch_size,), dtype=torch.int32, device=logits.device
    )
    target_predict = torch.argmax(logits, dim=-1).view(batch_size, num_tokens)
    verify_tree_greedy_func(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=num_correct_drafts,
        candidates=candidates,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=torch.full_like(retrieve_next_token, -1),
        target_predict=target_predict,
        topk=1,
    )

    accept_lens = num_correct_drafts + 1
    new_seq_lens = seq_lens + accept_lens
    accept_tokens = predict[accept_index]
    bonus_tokens = torch.empty_like(accept_lens)
    fill_bonus_tokens_func(
        accept_tokens,
        accept_lens,
        bonus_tokens,
        num_tokens,
        batch_size,
    )
    select_index = (
        torch.arange(
            0,
            batch_size * num_tokens,
            num_tokens,
            device=logits.device,
        )
        + accept_lens
        - 1
    )
    return TargetVerifyTopk1Output(
        predict=predict,
        num_correct_drafts=accept_lens - 1,
        accept_lens=accept_lens,
        accept_index=accept_index,
        bonus_tokens=bonus_tokens,
        new_seq_lens=new_seq_lens,
        select_index=select_index,
        draft_input_ids=predict.to(torch.int64),
    )


def _assert_target_verify_matches_eager(
    actual: TargetVerifyTopk1Output,
    expected: TargetVerifyTopk1Output,
):
    for field in TargetVerifyTopk1Output._fields:
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field), rtol=0, atol=0
        )


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
                    torch.ones(
                        (batch_size, 1), dtype=torch.float32, device=self.device
                    ),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    positions, expected_positions, rtol=0, atol=0
                )

    def test_draft_topk1_postprocess_can_write_draft_token_column(self):
        batch_size = 17
        # Multi-split vocab so the fused write composes with the split reduction.
        vocab_size = 50000
        logits, expected_index = _make_logits_with_unique_argmax(
            batch_size,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=0,
        )
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        backing = torch.full((batch_size, 5), -1, dtype=torch.long, device=self.device)
        draft_tokens = backing[:, 1:4]

        topk_p, topk_index = draft_topk1_postprocess(
            logits, positions, draft_tokens, draft_token_column=2
        )

        torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        # Exactly one backing column is written; both neighbors stay untouched.
        expected_backing = torch.full_like(backing, -1)
        expected_backing[:, 3] = expected_index[:, 0]
        torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.ones_like(positions), rtol=0, atol=0
        )

    def test_row_strided_logits_view_matches_argmax(self):
        batch_size = 5
        vocab_size = 8193
        # Poison the padding columns: if the kernel used the dense vocab width
        # as the row stride it would read them and pick the wrong index.
        backing = torch.full(
            (batch_size, vocab_size + 64),
            2000.0,
            dtype=torch.float32,
            device=self.device,
        )
        logits, expected_index = _make_logits_with_unique_argmax(
            batch_size,
            vocab_size,
            dtype=torch.float32,
            device=self.device,
            seed=1,
        )
        backing[:, :vocab_size] = logits
        strided_logits = backing[:, :vocab_size]
        self.assertFalse(strided_logits.is_contiguous())
        positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        topk_p, topk_index = draft_topk1_postprocess(strided_logits, positions)

        torch.testing.assert_close(topk_index, expected_index, rtol=0, atol=0)
        torch.testing.assert_close(topk_p, torch.ones_like(topk_p), rtol=0, atol=0)
        torch.testing.assert_close(
            positions, torch.ones_like(positions), rtol=0, atol=0
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
            draft_topk1_postprocess(
                logits, torch.empty(4, dtype=torch.long, device=self.device)
            )
        with self.assertRaises(AssertionError):
            draft_topk1_postprocess(torch.empty((4, 16), device=self.device), positions)

    def test_target_verify_fused_matches_eager_exactly(self):
        batch_size, num_tokens, vocab_size = 4, 6, 154880
        accepted_drafts = [0, 1, num_tokens - 2, num_tokens - 1]
        dense_logits, target_ids = _make_logits_with_unique_argmax(
            batch_size * num_tokens,
            vocab_size,
            dtype=torch.bfloat16,
            device=self.device,
            seed=3,
        )
        backing = torch.full(
            (batch_size * num_tokens, vocab_size + 17),
            2000.0,
            dtype=torch.bfloat16,
            device=self.device,
        )
        logits = backing[:, :vocab_size]
        logits.copy_(dense_logits)
        target_ids = target_ids.view(batch_size, num_tokens)

        candidates = torch.zeros(
            (batch_size, num_tokens), dtype=torch.long, device=self.device
        )
        candidates[:, 1:] = (target_ids[:, :-1] + 1) % vocab_size
        for batch_idx, num_accepted in enumerate(accepted_drafts):
            candidates[batch_idx, 1 : num_accepted + 1] = target_ids[
                batch_idx, :num_accepted
            ]
        retrieve_index, retrieve_next_token = _make_topk1_chain(
            batch_size, num_tokens, self.device
        )
        seq_lens = torch.arange(
            100, 100 + batch_size, dtype=torch.long, device=self.device
        )

        fused = target_verify_topk1_postprocess(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )
        eager = _eager_target_verify_topk1(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )

        self.assertFalse(logits.is_contiguous())
        _assert_target_verify_matches_eager(fused, eager)
        torch.testing.assert_close(
            fused.num_correct_drafts,
            torch.tensor(accepted_drafts, dtype=torch.int32, device=self.device),
            rtol=0,
            atol=0,
        )

    def test_target_verify_leftmost_ties_and_all_negative_infinity(self):
        batch_size, num_tokens, vocab_size = 1, 3, 8193
        logits = torch.full(
            (num_tokens, vocab_size),
            -float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        logits[0, 7] = 1.0
        logits[0, 8192] = 1.0
        # Row 1 remains all -inf, whose leftmost argmax is zero.
        logits[2, 8191] = 1.0
        logits[2, 8192] = 1.0
        candidates = torch.tensor([[0, 7, 0]], device=self.device)
        retrieve_index, retrieve_next_token = _make_topk1_chain(
            batch_size, num_tokens, self.device
        )
        seq_lens = torch.tensor([41], dtype=torch.int32, device=self.device)

        fused = target_verify_topk1_postprocess(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )
        eager = _eager_target_verify_topk1(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )

        _assert_target_verify_matches_eager(fused, eager)
        torch.testing.assert_close(
            fused.predict,
            torch.tensor([7, 0, 8191], dtype=torch.int32, device=self.device),
            rtol=0,
            atol=0,
        )

    def test_target_verify_trivial_chain(self):
        logits = torch.tensor([[0.0, 3.0, 1.0]], device=self.device)
        candidates = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        retrieve_index, retrieve_next_token = _make_topk1_chain(1, 1, self.device)
        seq_lens = torch.tensor([9], dtype=torch.long, device=self.device)

        fused = target_verify_topk1_postprocess(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )
        eager = _eager_target_verify_topk1(
            logits,
            candidates,
            retrieve_index,
            retrieve_next_token,
            seq_lens,
        )

        _assert_target_verify_matches_eager(fused, eager)
        torch.testing.assert_close(
            fused.bonus_tokens,
            torch.tensor([1], dtype=torch.int32, device=self.device),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
