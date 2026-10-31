"""Unit tests for EAGLE3 rejection sampling with a reduced (hot) draft vocabulary.

The draft head emits ``draft_vocab_size`` columns and ``hot_token_id`` maps them
onto target token ids, so ``q`` has to be lifted to the target vocabulary before
the chain kernel -- which indexes ``q`` by target token id -- can read it.
``eagle_utils`` rejects a ``draft_probs`` whose last dim does not match
``target_probs``, so the width contract is what these tests pin down.
"""

import unittest

import torch

from sglang.srt.speculative.spec_utils import scatter_draft_probs_to_target_vocab
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")


class TestScatterDraftProbsToTargetVocab(CustomTestCase):
    def setUp(self):
        self.target_vocab = 16
        self.hot_token_id = torch.tensor([1, 3, 4, 9, 15], dtype=torch.int64)
        self.draft_vocab = self.hot_token_id.numel()

    def test_hot_columns_land_on_their_target_ids(self):
        draft_probs = torch.rand(2, 3, self.draft_vocab)
        draft_probs /= draft_probs.sum(-1, keepdim=True)

        out = scatter_draft_probs_to_target_vocab(
            draft_probs, self.hot_token_id, self.target_vocab
        )

        self.assertEqual(out.shape, (2, 3, self.target_vocab))
        torch.testing.assert_close(out[..., self.hot_token_id], draft_probs)

    def test_non_hot_mass_is_exactly_zero(self):
        # q(x) = 0 off the hot set is the true proposal mass, not an
        # approximation: the draft can only ever propose a hot token. That is
        # what lets relu(p - q) keep the full p on non-hot tokens.
        draft_probs = torch.rand(4, 2, self.draft_vocab)
        out = scatter_draft_probs_to_target_vocab(
            draft_probs, self.hot_token_id, self.target_vocab
        )

        cold = torch.ones(self.target_vocab, dtype=torch.bool)
        cold[self.hot_token_id] = False
        self.assertEqual(out[..., cold].count_nonzero().item(), 0)

    def test_total_mass_is_preserved(self):
        draft_probs = torch.rand(3, 5, self.draft_vocab)
        draft_probs /= draft_probs.sum(-1, keepdim=True)
        out = scatter_draft_probs_to_target_vocab(
            draft_probs, self.hot_token_id, self.target_vocab
        )
        torch.testing.assert_close(out.sum(-1), torch.ones(3, 5), rtol=1e-6, atol=1e-6)

    def test_output_is_fp32_for_any_input_dtype(self):
        # The eager and CUDA-graph paths must hand the kernel the same dtype,
        # so the output dtype is pinned rather than inherited from the input.
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            out = scatter_draft_probs_to_target_vocab(
                torch.rand(2, 1, self.draft_vocab, dtype=dtype),
                self.hot_token_id,
                self.target_vocab,
            )
            self.assertEqual(out.dtype, torch.float32, msg=f"input dtype {dtype}")

    def test_no_buffer_reuse_across_calls(self):
        # The returned tensor outlives the call under overlap scheduling, so a
        # cached buffer would alias across iterations.
        a = scatter_draft_probs_to_target_vocab(
            torch.zeros(1, 1, self.draft_vocab), self.hot_token_id, self.target_vocab
        )
        b = scatter_draft_probs_to_target_vocab(
            torch.ones(1, 1, self.draft_vocab), self.hot_token_id, self.target_vocab
        )
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertEqual(a.count_nonzero().item(), 0)

    def test_width_matches_target_probs_contract(self):
        # eagle_utils raises when draft_probs.shape[-1] != target_probs.shape[-1].
        target_probs = torch.rand(2, 3, self.target_vocab)
        out = scatter_draft_probs_to_target_vocab(
            torch.rand(2, 3, self.draft_vocab), self.hot_token_id, self.target_vocab
        )
        self.assertEqual(out.shape[-1], target_probs.shape[-1])

    def test_chain_rejection_sampling_stays_unbiased(self):
        """A truncated q does not bias the sampler, it only lowers acceptance.

        Monte-Carlo the accept test the chain kernel runs -- accept with
        ``min(1, p/q)``, otherwise resample from ``relu(p - q)`` -- with q
        supported only on the hot set, and check the emitted distribution is p.
        """
        torch.manual_seed(0)
        n = 400_000
        p = torch.rand(self.target_vocab)
        p /= p.sum()
        hot_q = torch.rand(self.draft_vocab)
        hot_q /= hot_q.sum()
        q = scatter_draft_probs_to_target_vocab(
            hot_q.view(1, 1, -1), self.hot_token_id, self.target_vocab
        ).view(-1)

        proposed = self.hot_token_id[torch.multinomial(hot_q, n, replacement=True)]
        accepted = torch.rand(n) < (p[proposed] / q[proposed]).clamp(max=1.0)

        counts = torch.zeros(self.target_vocab)
        counts.index_add_(0, proposed[accepted], torch.ones(int(accepted.sum())))
        residual = (p - q).clamp(min=0)
        residual /= residual.sum()
        n_rejected = int((~accepted).sum())
        counts.index_add_(
            0,
            torch.multinomial(residual, n_rejected, replacement=True),
            torch.ones(n_rejected),
        )

        empirical = counts / n
        standard_error = (p * (1 - p) / n).sqrt().max()
        self.assertLess(float((empirical - p).abs().max()), 4 * float(standard_error))


if __name__ == "__main__":
    unittest.main()
