"""Unit tests for the chain speculative-sampling torch reference.

Covers ``chain_speculative_sampling_torch`` (the device fallback for
``chain_speculative_sampling_triton`` used by DFlash2/DSpark verify on
devices without Triton): analytic acceptance/resampling cases, buffer
layouts, the statistical losslessness property, and CUDA parity.
"""

import unittest

import torch

from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_torch,
    chain_speculative_sampling_triton,
)
from sglang.kernels.ops.speculative.triton_compat import cdiv, next_power_of_2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_buffers(bs, num_slots, vocab, device="cpu"):
    """Chain-verify buffers matching _get_or_create_chain_verify_buffers."""
    retrieve_index = torch.arange(
        bs * num_slots, dtype=torch.int64, device=device
    ).view(bs, num_slots)
    row_next = torch.arange(1, num_slots + 1, dtype=torch.int64, device=device)
    row_next[-1] = -1
    retrieve_next_token = row_next.unsqueeze(0).expand(bs, -1).clone()
    retrieve_next_sibling = torch.full(
        (bs, num_slots), -1, dtype=torch.int64, device=device
    )
    predicts = torch.zeros(bs * num_slots, dtype=torch.int32, device=device)
    accept_index = torch.zeros(bs, num_slots, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros(bs, dtype=torch.int32, device=device)
    return (
        predicts,
        accept_index,
        accept_token_num,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    )


def _run(fn, target_probs, draft_probs, candidates, coins, coin_final, device="cpu"):
    (
        predicts,
        accept_index,
        accept_token_num,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    ) = _make_buffers(
        candidates.shape[0], candidates.shape[1], target_probs.shape[-1], device
    )
    fn(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        coins,
        coin_final,
        target_probs,
        draft_probs,
        1.0,
        1.0,
        True,
    )
    num_slots = candidates.shape[1]
    return (
        predicts.view(candidates.shape[0], num_slots),
        accept_index,
        accept_token_num,
    )


def _one_hot(rows, vocab, token, value=1.0, device="cpu"):
    probs = torch.zeros(rows, vocab, dtype=torch.float32, device=device)
    probs[:, token] = value
    return probs


class TestChainSpeculativeSamplingTorch(CustomTestCase):
    def test_all_accepted_one_hot_final(self):
        # q is a point mass on each candidate and p agrees, so every coin
        # accepts; the all-accepted final sample comes from the last target
        # row, which is one-hot on the bonus token.
        bs, num_slots, vocab = 1, 4, 6
        candidates = torch.tensor([[0, 3, 4, 5]])
        target_probs = torch.zeros(num_slots, vocab)
        draft_probs = torch.zeros(num_slots - 1, vocab)
        for row, token in enumerate([3, 4, 5]):
            target_probs[row, token] = 1.0
            draft_probs[row, token] = 1.0
        target_probs[num_slots - 1, 2] = 1.0  # final row: bonus = token 2
        coins = torch.full((bs, num_slots - 1), 0.99)
        coin_final = torch.full((bs,), 0.5)

        predicts, accept_index, accept_token_num = _run(
            chain_speculative_sampling_torch,
            target_probs,
            draft_probs,
            candidates,
            coins,
            coin_final,
        )

        self.assertEqual(accept_token_num.tolist(), [3])
        self.assertEqual(accept_index[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(predicts[0].tolist(), [3, 4, 5, 2])

    def test_reject_residual_point_mass_bonus(self):
        # Accept step 1, reject step 2 (p(c2) == 0), then the residual
        # max(p - q, 0) of row 1 is a point mass on token 5: q covers token 4
        # exactly, leaving only token 5.
        bs, num_slots, vocab = 1, 4, 6
        candidates = torch.tensor([[0, 3, 4, 5]])
        target_probs = torch.zeros(num_slots, vocab)
        draft_probs = torch.zeros(num_slots - 1, vocab)
        # Step 1: q(c=3)=1, p(3)=1 -> accepted.
        target_probs[0, 3] = 1.0
        draft_probs[0, 3] = 1.0
        # Step 2 (row 1): p(c=4)=0 -> coin*q < 0 never holds -> rejected.
        # Residual row: p = {4: .5, 5: .5}, q = {4: .5} -> point mass on 5.
        target_probs[1, 4] = 0.5
        target_probs[1, 5] = 0.5
        draft_probs[1, 4] = 0.5
        coins = torch.full((bs, num_slots - 1), 0.5)
        coin_final = torch.full((bs,), 0.9)  # 0.9 * 0.5 = 0.45 -> token 5

        predicts, accept_index, accept_token_num = _run(
            chain_speculative_sampling_torch,
            target_probs,
            draft_probs,
            candidates,
            coins,
            coin_final,
        )

        self.assertEqual(accept_token_num.tolist(), [1])
        self.assertEqual(predicts[0].tolist(), [3, 5, 0, 0])
        self.assertEqual(accept_index[0, :2].tolist(), [0, 1])

    def test_nan_draft_is_zeroed_in_residual(self):
        # A NaN draft entry counts as 0 (kernel guard): it rejects the step
        # (coin*NaN < p is false) and contributes its full p to the residual.
        bs, num_slots, vocab = 1, 2, 4
        candidates = torch.tensor([[0, 1]])
        target_probs = torch.zeros(num_slots, vocab)
        draft_probs = torch.zeros(num_slots - 1, vocab)
        target_probs[0, 2] = 1.0
        draft_probs[0, 2] = float("nan")
        coins = torch.full((bs, num_slots - 1), 0.5)
        coin_final = torch.full((bs,), 0.5)

        predicts, accept_index, accept_token_num = _run(
            chain_speculative_sampling_torch,
            target_probs,
            draft_probs,
            candidates,
            coins,
            coin_final,
        )

        self.assertEqual(accept_token_num.tolist(), [0])
        # Residual = p (NaN q -> 0), one-hot on token 2.
        self.assertEqual(predicts[0, 0].item(), 2)

    def test_mixed_batch_matches_single_row_runs(self):
        # Rows of the batch see independent chains: row 0 all-accepted, row 1
        # rejected at step 2, row 2 rejected at step 1.
        bs, num_slots, vocab = 3, 3, 5
        candidates = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        target_probs = torch.zeros(bs, num_slots, vocab)
        draft_probs = torch.zeros(bs, num_slots - 1, vocab)
        # Row 0: accept both.
        target_probs[0, 0, 1] = 1.0
        draft_probs[0, 0, 1] = 1.0
        target_probs[0, 1, 2] = 1.0
        draft_probs[0, 1, 2] = 1.0
        target_probs[0, 2, 3] = 1.0  # final row
        # Row 1: accept step 1, reject step 2; residual one-hot on 4.
        target_probs[1, 0, 1] = 1.0
        draft_probs[1, 0, 1] = 1.0
        target_probs[1, 1, 4] = 1.0
        # Row 2: reject immediately; residual one-hot on 3.
        target_probs[2, 0, 3] = 1.0
        coins = torch.full((bs, num_slots - 1), 0.5)
        coin_final = torch.full((bs,), 0.5)

        predicts, accept_index, accept_token_num = _run(
            chain_speculative_sampling_torch,
            target_probs,
            draft_probs,
            candidates,
            coins,
            coin_final,
        )

        self.assertEqual(accept_token_num.tolist(), [2, 1, 0])
        # predicts[row, slot]: accepted drafts then the bonus at the last
        # accepted slot.
        self.assertEqual(predicts[0].tolist(), [1, 2, 3])
        self.assertEqual(predicts[1].tolist(), [1, 4, 0])
        self.assertEqual(predicts[2].tolist(), [3, 0, 0])
        self.assertTrue(
            torch.equal(accept_index[:, 0], torch.arange(bs, dtype=torch.int32))
        )

    def test_first_token_distribution_matches_target(self):
        # Losslessness property: with one draft slot, the first committed
        # token (accepted draft or resampled bonus) is distributed exactly
        # like the target row, whatever the draft distribution is.
        torch.manual_seed(0)
        vocab, bs = 8, 20000
        target = torch.softmax(torch.randn(vocab), dim=-1)
        draft = torch.softmax(torch.randn(vocab) * 2.0, dim=-1)

        candidates = torch.multinomial(draft, bs, replacement=True).unsqueeze(1)
        target_probs = target.view(1, 1, vocab).expand(bs, 2, vocab).contiguous()
        draft_probs = draft.view(1, 1, vocab).expand(bs, 1, vocab).contiguous()
        coins = torch.rand(bs, 1)
        coin_final = torch.rand(bs)

        predicts, _, _ = _run(
            chain_speculative_sampling_torch,
            target_probs,
            draft_probs,
            candidates,
            coins,
            coin_final,
        )

        counts = torch.bincount(predicts[:, 0].to(torch.int64), minlength=vocab)
        empirical = counts.float() / bs
        total_variation = 0.5 * (empirical - target).abs().sum().item()
        self.assertLess(total_variation, 0.02)

    def test_triton_compat_helpers(self):
        # Pure-python fallbacks used when triton is not installed.
        self.assertEqual(next_power_of_2(1), 1)
        self.assertEqual(next_power_of_2(5), 8)
        self.assertEqual(next_power_of_2(16), 16)
        self.assertEqual(cdiv(10, 3), 4)
        self.assertEqual(cdiv(9, 3), 3)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA parity check needs a GPU")
    def test_cuda_parity_with_triton_kernel(self):
        # Distributions are built with wide margins so fp32 summation-order
        # differences between the kernel and torch cannot flip an outcome.
        torch.manual_seed(1)
        device = "cuda"
        bs, num_slots, vocab = 16, 5, 64
        target_probs = torch.softmax(
            torch.randn(bs, num_slots, vocab, device=device) * 6.0, dim=-1
        )
        draft_probs = torch.softmax(
            torch.randn(bs, num_slots - 1, vocab, device=device) * 6.0, dim=-1
        )
        candidates = torch.randint(
            0, vocab, (bs, num_slots), dtype=torch.int64, device=device
        )
        # Snap each candidate to a high-probability token of its draft row so
        # accept/reject margins are decisive.
        for row in range(bs):
            for step in range(1, num_slots):
                candidates[row, step] = draft_probs[row, step - 1].argmax()
        coins = torch.rand(bs, num_slots - 1, device=device)
        coin_final = torch.rand(bs, device=device)

        torch_out = _run(
            chain_speculative_sampling_torch,
            target_probs,
            draft_probs,
            candidates,
            coins,
            coin_final,
            device=device,
        )
        triton_out = _run(
            chain_speculative_sampling_triton,
            target_probs,
            draft_probs,
            candidates,
            coins,
            coin_final,
            device=device,
        )

        for t, k in zip(torch_out, triton_out):
            self.assertTrue(torch.equal(t, k))


if __name__ == "__main__":
    unittest.main()
