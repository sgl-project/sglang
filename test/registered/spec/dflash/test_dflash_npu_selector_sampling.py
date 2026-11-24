"""Regression coverage for DFlash2's Ascend selector rejection sampling."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu import sample

from sglang.srt.hardware_backend.npu.dflash_sampling import (
    NpuDFlashSelectorSampler,
    filter_probs_with_row_fallback,
)
from sglang.srt.speculative import dflash_worker_v2 as worker_mod
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="extra-b-test-1-npu-a3")


class TestDFlashNpuSelectorSampling(CustomTestCase):
    def test_batched_rejection_and_commit(self):
        for width in (2, 5, 8, 16):
            with self.subTest(width=width):
                self._check_batched_rejection_and_commit(width)

    def _check_batched_rejection_and_commit(self, width):
        device = "npu"
        bs, vocab = 4, 248320
        partial = min(width // 2, width - 2)
        candidates = torch.full((bs, width), 5, dtype=torch.long, device=device)
        candidates[:, 0] = 1
        selector_ids = torch.stack(
            (candidates[:, 1:], torch.full_like(candidates[:, 1:], 11)), dim=-1
        )
        q_rows = torch.empty_like(selector_ids, dtype=torch.float32)
        q_rows[..., 0] = 0.75
        q_rows[..., 1] = 0.25
        logits = torch.full((bs, width, vocab), -torch.inf, device=device)
        logits[..., 5] = 0.0
        for row, pos, token in ((0, 0, 100), (1, partial, 101), (3, width - 1, 103)):
            logits[row, pos, 5] = -torch.inf
            logits[row, pos, token] = 0.0
        logits[2:, 0, 5] = -torch.log(torch.tensor(3.0)).item()
        logits[2:, 0, 11] = 0.0
        # For the last two requests, p(5)=.25 and q(5)=.75. A coin of .5
        # rejects and samples token 11 from the residual. A coin of .3 accepts.
        coins = torch.full((bs, width), 0.5, device=device)
        coins[3, 0] = 0.3
        final_coins = torch.full((bs,), 0.25, device=device)
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((bs, 1), device=device),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            need_min_p_sampling=False,
            is_any_greedy=False,
        )
        worker = SimpleNamespace(
            _draft_probs_buf=None,
            _npu_selector_sampler=NpuDFlashSelectorSampler(),
        )
        worker._selector_sampling_accept = MethodType(
            worker_mod.DFlashWorkerV2._selector_sampling_accept, worker
        )

        group = SimpleNamespace(rank_in_group=0)
        decision = torch.empty((2, bs), dtype=torch.long, device=device)

        def broadcast(values, src):
            if group.rank_in_group == src:
                decision.copy_(values)
            else:
                values.copy_(decision)

        group.broadcast = broadcast

        def commit():
            num_correct_drafts, bonus_tokens = (
                worker_mod.DFlashWorkerV2._accept_npu_selector_batch(
                    worker,
                    candidates=candidates,
                    next_token_logits=logits.flatten(0, 1),
                    candidate_ids=selector_ids,
                    q_rows=q_rows,
                    sampling_info=sampling_info,
                    draft_input=None,
                )
            )
            out_tokens, commit_lens = worker_mod._commit_accept(
                candidates, num_correct_drafts, bonus_tokens
            )
            return num_correct_drafts, commit_lens, bonus_tokens, out_tokens

        with patch.object(worker_mod, "get_tp_group", return_value=group):
            with patch.object(torch, "rand", side_effect=[coins, final_coins]):
                num_correct_drafts, commit_lens, bonus_tokens, out_tokens = commit()
            self.assertEqual(num_correct_drafts.tolist(), [0, partial, 0, width - 1])
            self.assertEqual(commit_lens.tolist(), [1, partial + 1, 1, width])
            self.assertEqual(bonus_tokens.tolist(), [100, 101, 11, 103])
            self.assertEqual(
                out_tokens[1, : partial + 1].tolist(), [5] * partial + [101]
            )
            self.assertEqual(out_tokens[3].tolist(), [5] * (width - 1) + [103])
            self.assertEqual(worker._draft_probs_buf.count_nonzero().item(), 0)

            for rank in range(1, 4):
                with self.subTest(tp_rank=rank):
                    group.rank_in_group = rank
                    worker._draft_probs_buf = None
                    for actual, expected in zip(
                        commit(),
                        (num_correct_drafts, commit_lens, bonus_tokens, out_tokens),
                    ):
                        torch.testing.assert_close(actual, expected)
                    self.assertIsNone(worker._draft_probs_buf)

    def test_row_fallback_preserves_mixed_target_distributions(self):
        # Rows 1--4 hit a top-k tie, a top-p tie, an exact cutoff, and k > 64.
        # Other rows use different (k, p) pairs that can stay compact.
        heads = (
            (8, 4, 2, 1),
            (5, 3, 3),
            (6, 2, 2, 1),
            (6, 2, 1),
            tuple(range(70, 0, -1)),
            (12, 7, 4, 2, 1),
            (10, 9, 8, 7, 6),
        )
        probs = torch.zeros((len(heads), 248320), dtype=torch.float32)
        for row, head in enumerate(heads):
            offset = row * 16381
            probs[row, offset : offset + len(head)] = torch.tensor(head)
        probs = (probs / probs.sum(-1, keepdim=True)).npu()
        top_ks = torch.tensor([3, 2, 3, 2, 65, 4, 4], dtype=torch.int32, device="npu")
        top_ps = torch.tensor([0.9, 0.9, 0.7, 0.75, 0.95, 0.6, 1.0], device="npu")

        def reference(values, ks, ps):
            return sample.top_p_renorm_prob(sample.top_k_renorm_prob(values, ks), ps)

        expected = reference(probs.clone(), top_ks, top_ps)
        fallback_ks = []

        def reference_with_record(values, ks, ps):
            fallback_ks.append(ks.tolist())
            return reference(values, ks, ps)

        actual = filter_probs_with_row_fallback(
            probs, top_ks, top_ps, reference_with_record
        )
        self.assertEqual(fallback_ks, [[2, 3, 2, 65]])
        self.assertTrue(torch.equal(actual > 0, expected > 0))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

        fallback_ks.clear()
        actual = filter_probs_with_row_fallback(
            probs[1:5], top_ks[1:5], top_ps[1:5], reference_with_record
        )
        self.assertEqual(fallback_ks, [[2, 3, 2, 65]])
        torch.testing.assert_close(actual, expected[1:5], rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
