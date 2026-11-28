"""Unit tests for EAGLE3 rejection sampling with a reduced (hot) draft vocabulary.

The draft head emits ``draft_vocab_size`` columns and ``hot_token_id`` maps them
onto target token ids, so ``q`` has to be lifted to the target vocabulary before
the chain kernel -- which indexes ``q`` by target token id -- can read it.
``eagle_utils`` rejects a ``draft_probs`` whose last dim does not match
``target_probs``, so the width contract is what these tests pin down.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.speculative.eagle_worker_v2 as eagle_worker_v2
from sglang.srt.environ import envs
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_utils import (
    prune_quantized_lm_head,
    scatter_draft_probs_to_target_vocab,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


class TestVerifyDraftProbsVocabularyMapping(CustomTestCase):
    def make_worker(self, hot_token_id, target_vocab_size):
        worker = object.__new__(EagleDraftWorker)
        worker.hot_token_id = hot_token_id
        worker.target_worker = SimpleNamespace(
            model_config=SimpleNamespace(vocab_size=target_vocab_size)
        )
        worker._init_draft_probs_vocab_scatter()
        return worker

    def test_reduced_identity_prefix_is_expanded(self):
        """A zero-offset reduced map still needs target-width zero padding."""
        worker = self.make_worker(torch.arange(4, dtype=torch.int64), 8)
        rows = [torch.rand(2, 4) for _ in range(3)]
        tokens = torch.tensor([[0, 1, 2], [1, 2, 3]])

        out = worker._build_verify_draft_probs(rows, tokens)

        self.assertEqual(out.shape, (2, 3, 8))
        torch.testing.assert_close(out[..., :4], torch.stack(rows, dim=1))
        self.assertEqual(out[..., 4:].count_nonzero().item(), 0)

    def test_full_width_permutation_is_scattered(self):
        """A full-width permutation must not take the identity-map fast path."""
        worker = self.make_worker(torch.tensor([1, 0, 2, 3], dtype=torch.int64), 4)
        rows = [torch.rand(2, 4) for _ in range(3)]
        tokens = worker.hot_token_id[torch.tensor([[0, 1, 2], [1, 2, 3]])]

        out = worker._build_verify_draft_probs(rows, tokens)

        expected = torch.stack(rows, dim=1).scatter(
            -1,
            worker.hot_token_id.view(1, 1, -1).expand(2, 3, -1),
            torch.stack(rows, dim=1),
        )
        torch.testing.assert_close(out, expected)

    def test_full_width_identity_map_returns_stacked_distribution(self):
        """Identity maps must not allocate and scatter a duplicate q tensor."""
        worker = self.make_worker(torch.arange(4, dtype=torch.int64), 4)
        rows = [torch.rand(2, 4) for _ in range(3)]
        tokens = torch.tensor([[0, 1, 2], [1, 2, 3]])

        with patch.object(
            eagle_worker_v2,
            "scatter_draft_probs_to_target_vocab",
            side_effect=AssertionError("identity map must not be scattered"),
        ):
            out = worker._build_verify_draft_probs(rows, tokens)

        torch.testing.assert_close(out, torch.stack(rows, dim=1))

    def test_chain_rows_keep_step_order(self):
        """Distinct per-step distributions must stay aligned with their tokens."""
        worker = self.make_worker(torch.arange(4, dtype=torch.int64), 4)
        rows = [torch.full((2, 4), i + 1, dtype=torch.float32) for i in range(3)]

        out = worker._build_verify_draft_probs(
            rows, torch.tensor([[0, 1, 2], [1, 2, 3]])
        )

        self.assertEqual(out[0, :, 0].tolist(), [1.0, 2.0, 3.0])


class TestVerifyDraftProbsAssert(CustomTestCase):
    """The ``q > 0`` probe in ``_build_verify_draft_probs`` under async asserts."""

    def setUp(self):
        self.worker = object.__new__(EagleDraftWorker)
        self.worker.hot_token_id = torch.tensor([1, 3, 4, 9, 15], dtype=torch.int64)
        self.worker.target_vocab_size = 16
        self.worker.draft_probs_need_vocab_scatter = True
        # Two requests, three chain slots; slot i's token is drawn from row i.
        # Distinct rows make a row permutation detectable.
        self.rows = [
            torch.full((2, 5), (i + 1) / 5, dtype=torch.float32) for i in range(3)
        ]
        self.draft_tokens = self.worker.hot_token_id[
            torch.tensor([[0, 2, 4], [1, 3, 0]])
        ]

    def test_cuda_graph_filler_rows_pass(self):
        # Capture warms up on the zeroed q buffer and replay zeroes padded rows.
        self.rows[0][1] = 0
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
            out = self.worker._build_verify_draft_probs(self.rows, self.draft_tokens)
        self.assertEqual(out.shape, (2, 3, 16))

    def test_proposed_token_without_mass_fires(self):
        # Target id 2 is not hot, so its scattered q is 0 although the row has mass.
        self.draft_tokens[0, 1] = 2
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
            with self.assertRaises(RuntimeError):
                self.worker._build_verify_draft_probs(self.rows, self.draft_tokens)


class _DequantLinear:
    def apply(self, layer, x, bias=None):
        w = layer.weight.float()
        if w.shape[0] != x.shape[-1]:
            w = w.t()
        return x.float() @ (w * layer.weight_scale.float().reshape(1, -1))


def _fp8_lm_head(hidden_size, vocab_size, *, per_channel):
    dense = torch.randn(vocab_size, hidden_size)
    scale = dense.abs().amax(dim=1, keepdim=True) / 448.0
    if not per_channel:
        scale = scale.amax().reshape(1)
    head = nn.Module()
    # Per-channel FP8 after loading: [hidden, vocab] column-major, [vocab, 1] scale.
    q = (dense / scale.reshape(-1, 1)).to(torch.float8_e4m3fn)
    head.weight = nn.Parameter(q.t() if per_channel else q, requires_grad=False)
    head.weight_scale = nn.Parameter(scale, requires_grad=False)
    head.quant_method = _DequantLinear()
    head.logical_widths = [vocab_size]
    return head


class TestPruneQuantizedLmHead(CustomTestCase):
    """--speculative-token-map with a per-channel FP8 lm_head used to slice the
    transposed weight on the hidden dim (device-side assert at startup)."""

    def setUp(self):
        torch.manual_seed(0)
        self.hidden_size, self.vocab_size = 64, 512
        self.hot_token_id = torch.tensor([0, 3, 17, 100, 255, 511])

    def test_per_channel_hot_logits_match_target(self):
        head = _fp8_lm_head(self.hidden_size, self.vocab_size, per_channel=True)
        pruned = prune_quantized_lm_head(
            head, hot_token_id=self.hot_token_id, hidden_size=self.hidden_size
        )
        x = torch.randn(5, self.hidden_size)

        torch.testing.assert_close(
            pruned.quant_method.apply(pruned, x),
            head.quant_method.apply(head, x)[:, self.hot_token_id],
        )
        # The FP8 GEMM needs the column-major weight it was loaded with.
        self.assertTrue(pruned.weight.t().is_contiguous())
        self.assertEqual(pruned.logical_widths, [self.hot_token_id.numel()])

    def test_target_head_is_not_modified(self):
        head = _fp8_lm_head(self.hidden_size, self.vocab_size, per_channel=True)
        weight, scale = head.weight, head.weight_scale
        prune_quantized_lm_head(
            head, hot_token_id=self.hot_token_id, hidden_size=self.hidden_size
        )

        self.assertIs(head.weight, weight)
        self.assertIs(head.weight_scale, scale)
        self.assertEqual(head.logical_widths, [self.vocab_size])

    def test_per_tensor_scale_slices_vocab_rows(self):
        head = _fp8_lm_head(self.hidden_size, self.vocab_size, per_channel=False)
        pruned = prune_quantized_lm_head(
            head, hot_token_id=self.hot_token_id, hidden_size=self.hidden_size
        )

        self.assertIs(pruned.weight_scale, head.weight_scale)
        torch.testing.assert_close(
            pruned.weight.float(), head.weight.float()[self.hot_token_id]
        )

    def test_block_scale_is_rejected(self):
        head = _fp8_lm_head(self.hidden_size, self.vocab_size, per_channel=True)
        head.weight_scale = nn.Parameter(torch.ones(4, 2), requires_grad=False)
        with self.assertRaises(NotImplementedError):
            prune_quantized_lm_head(
                head, hot_token_id=self.hot_token_id, hidden_size=self.hidden_size
            )


if __name__ == "__main__":
    unittest.main()
