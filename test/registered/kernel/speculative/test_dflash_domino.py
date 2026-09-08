import unittest

import torch
from torch import nn

from sglang.srt.speculative.dflash_utils import compute_dflash_correct_drafts_and_bonus
from sglang.srt.speculative.dflash_worker_v2 import _DominoDraftSampler
from sglang.srt.speculative.domino_utils import _domino_gru_cell, domino_greedy_rollout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestDFlashDominoRollout(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.embedding = nn.Embedding(31, 8, device="cuda", dtype=torch.bfloat16)
        self.prefix_gru = nn.GRU(8, 4, batch_first=True, bias=False).cuda().bfloat16()
        self.embed_proj = (
            nn.Sequential(
                nn.Linear(12, 5, bias=False), nn.SiLU(), nn.Linear(5, 31, bias=False)
            )
            .cuda()
            .bfloat16()
        )
        self.lm_head_weight = torch.randn(31, 8, device="cuda", dtype=torch.bfloat16)
        self.hidden = torch.randn(3, 16, 8, device="cuda", dtype=torch.bfloat16)
        self.bonus_tokens = torch.tensor([1, 4, 9], device="cuda")

    def rollout(self, hidden, bonus_tokens, pool_size=5, shift_label=True):
        return domino_greedy_rollout(
            draft_hidden=hidden,
            bonus_tokens=bonus_tokens,
            target_embedding=self.embedding,
            lm_head_weight=self.lm_head_weight,
            prefix_gru=self.prefix_gru,
            embed_proj=self.embed_proj,
            vocab_size=31,
            shift_label=shift_label,
            candidate_pool_size=pool_size,
        )

    def test_gru_feedback_matches_sequence(self):
        embeddings = self.embedding(torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda"))
        _, expected = self.prefix_gru(embeddings)
        state = torch.zeros(2, 4, device="cuda", dtype=torch.bfloat16)
        for step in embeddings.unbind(dim=1):
            state = _domino_gru_cell(self.prefix_gru, step, state)
        torch.testing.assert_close(state, expected[0], rtol=0.02, atol=0.002)

    def test_candidate_pool_boundaries(self):
        for shift_label in (True, False):
            with self.subTest(shift_label=shift_label):
                full = self.rollout(self.hidden, self.bonus_tokens, 0, shift_label)
                for pool_size in (31, 32):
                    actual = self.rollout(
                        self.hidden, self.bonus_tokens, pool_size, shift_label
                    )
                    torch.testing.assert_close(actual, full, rtol=0, atol=0)
                first_hidden = self.hidden[:, 0 if shift_label else 1]
                expected_first = (first_hidden @ self.lm_head_weight.T).argmax(dim=-1)
                for block_size in (2, 16):
                    limited = self.rollout(
                        self.hidden[:, :block_size], self.bonus_tokens, 1, shift_label
                    )
                    self.assertEqual(
                        limited.shape, (3, block_size - int(not shift_label))
                    )
                    torch.testing.assert_close(limited[:, 0], expected_first)
                    if block_size > 2:
                        torch.testing.assert_close(
                            limited[:, 1:], limited[:, 1:2].expand_as(limited[:, 1:])
                        )

    def test_batch_matches_individual_requests(self):
        for pool_size in (0, 5):
            with self.subTest(pool_size=pool_size):
                batched = self.rollout(self.hidden, self.bonus_tokens, pool_size)
                individual = torch.cat(
                    [
                        self.rollout(hidden[None], bonus[None], pool_size)
                        for hidden, bonus in zip(self.hidden, self.bonus_tokens)
                    ]
                )
                torch.testing.assert_close(batched, individual, rtol=0, atol=0)

    def test_shift_label_uses_last_hidden_position(self):
        with torch.no_grad():
            self.embed_proj[2].weight.zero_()
            self.lm_head_weight.zero_()
            self.lm_head_weight[30, 0] = 1
        self.hidden.zero_()
        self.hidden[:, -1, 0] = 1
        proposals = self.rollout(self.hidden, self.bonus_tokens, pool_size=0)
        self.assertEqual(proposals.shape, (3, 16))
        self.assertTrue(torch.all(proposals[:, :-1] == 0))
        self.assertTrue(torch.all(proposals[:, -1] == 30))
        candidates = torch.cat((self.bonus_tokens[:, None], proposals), dim=1)
        target_predict = torch.cat(
            (proposals, torch.full((3, 1), 7, device="cuda", dtype=torch.long)), dim=1
        )
        num_correct, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates, target_predict=target_predict
        )
        self.assertTrue(torch.all(num_correct + 1 == 17))
        self.assertTrue(torch.all(bonus == 7))

    def test_sampler_replays_with_new_inputs(self):
        sampler = _DominoDraftSampler(
            target_embedding=self.embedding,
            lm_head_weight=self.lm_head_weight,
            prefix_gru=self.prefix_gru,
            embed_proj=self.embed_proj,
            vocab_size=31,
            block_size=16,
            shift_label=True,
            max_bs=3,
            candidate_pool_size=5,
        )
        block_ids = torch.zeros(3, 16, device="cuda", dtype=torch.long)
        block_ids[:, 0].copy_(self.bonus_tokens)

        def sample():
            sampler(self.hidden.flatten(0, 1), block_ids.flatten())

        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            sample()
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sample()

        for _ in range(2):
            self.hidden.copy_(torch.randn_like(self.hidden))
            block_ids[:, 0].copy_(torch.randint(31, (3,), device="cuda"))
            expected = self.rollout(self.hidden, block_ids[:, 0])
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                sampler.out.view(3, 16), expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
