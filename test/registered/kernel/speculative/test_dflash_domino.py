import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

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
                    self.assertEqual(limited.shape, (3, block_size - 1))
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
                sampler.out.view(3, 15), expected, rtol=0, atol=0
            )

    def test_tp2_gathered_base_matches_full_vocab_rollout(self):
        class FakeTpGroup:
            world_size = 2

            def __init__(self, remote_logits):
                self.remote_logits = remote_logits
                self.candidate_ids = None
                self.local_top_scores = None
                self.remote_top_scores = None
                self.full_logit_gathers = 0
                self.candidate_all_reduces = 0

            def all_gather_into_tensor(self, output, local):
                output[: local.shape[0]].copy_(local)
                if local.ndim == 1:
                    remote_max, remote_pos = self.remote_logits[0, :, :15].max(-1)
                    remote = (
                        remote_pos + 16 if local.dtype == torch.long else remote_max
                    )
                elif local.shape[0] == local_vocab_size:
                    self.full_logit_gathers += 1
                    remote = self.remote_logits.reshape(-1, local_vocab_size).T
                else:
                    k = local.shape[1]
                    remote_scores, remote_pos = torch.topk(
                        self.remote_logits[1:, :, :15].amax(0),
                        k=k,
                        dim=-1,
                        sorted=False,
                    )
                    if local.dtype == torch.long:
                        remote = remote_pos + 16
                        all_scores = (
                            torch.stack(
                                (self.local_top_scores, self.remote_top_scores), dim=0
                            )
                            .permute(1, 0, 2)
                            .reshape(local.shape[0], -1)
                        )
                        all_ids = (
                            torch.stack((local, remote), dim=0)
                            .permute(1, 0, 2)
                            .reshape(local.shape[0], -1)
                        )
                        global_pos = torch.topk(
                            all_scores, k=k, dim=-1, sorted=False
                        ).indices
                        self.candidate_ids = torch.gather(all_ids, 1, global_pos)
                    else:
                        remote = remote_scores
                        self.local_top_scores = local.clone()
                        self.remote_top_scores = remote_scores
                output[local.shape[0] :].copy_(remote)

            def all_reduce(self, local):
                self.candidate_all_reduces += 1
                remote_owned = (self.candidate_ids >= 16) & (self.candidate_ids < 31)
                remote_pos = (self.candidate_ids - 16).clamp(0, 14)
                remote = torch.gather(
                    self.remote_logits[1:].transpose(0, 1),
                    2,
                    remote_pos[:, None, :].expand(
                        -1, self.remote_logits.shape[0] - 1, -1
                    ),
                )
                remote.masked_fill_(~remote_owned[:, None, :], 0)
                local.add_(remote)
                return local

        block_size = 7
        local_vocab_size = 16
        padded_weight = torch.cat(
            (
                self.lm_head_weight,
                torch.zeros(
                    1,
                    8,
                    device="cuda",
                    dtype=torch.bfloat16,
                ).fill_(1000),
            )
        )
        local_weight = padded_weight[:local_vocab_size]
        remote_weight = padded_weight[local_vocab_size:]
        cases = (
            (8, 5, None, 1 << 60, False),
            (8, 5, None, 0, True),
            (1, 5, False, 1 << 60, False),
            (8, 5, True, 1 << 60, True),
            (8, 0, True, 1 << 60, False),
        )
        for (
            batch_size,
            candidate_pool_size,
            prefer_tp_candidate_pool,
            full_base_logits_max_bytes,
            expect_compact,
        ) in cases:
            bonus_tokens = torch.arange(batch_size, device="cuda")
            for shift_label in (True, False):
                with self.subTest(
                    batch_size=batch_size,
                    shift_label=shift_label,
                    candidate_pool_size=candidate_pool_size,
                    prefer_tp_candidate_pool=prefer_tp_candidate_pool,
                    full_base_logits_max_bytes=full_base_logits_max_bytes,
                    expect_compact=expect_compact,
                ):
                    draft_hidden = torch.randn(
                        batch_size,
                        block_size,
                        8,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    start = 0 if shift_label else 1
                    z = draft_hidden[:, start : start + block_size - 1]
                    logits_input = (
                        z.transpose(0, 1)
                        .contiguous()
                        .view((block_size - 1) * batch_size, 8)
                    )
                    remote_logits = F.linear(logits_input, remote_weight).view(
                        block_size - 1, batch_size, local_vocab_size
                    )
                    tp_group = FakeTpGroup(remote_logits)
                    rollout_kwargs = dict(
                        draft_hidden=draft_hidden,
                        bonus_tokens=bonus_tokens,
                        target_embedding=self.embedding,
                        lm_head_weight=local_weight,
                        prefix_gru=self.prefix_gru,
                        embed_proj=self.embed_proj,
                        vocab_size=31,
                        shift_label=shift_label,
                        candidate_pool_size=candidate_pool_size,
                        tp_group=tp_group,
                        lm_head_num_org=local_vocab_size,
                        lm_head_num_org_padded=local_vocab_size,
                        prefer_tp_candidate_pool=prefer_tp_candidate_pool,
                    )
                    with mock.patch(
                        "sglang.srt.speculative.domino_utils."
                        "_DOMINO_TP_FULL_BASE_LOGITS_MAX_BYTES",
                        full_base_logits_max_bytes,
                    ):
                        actual = domino_greedy_rollout(**rollout_kwargs)
                    expected = self.rollout(
                        draft_hidden, bonus_tokens, candidate_pool_size, shift_label
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(
                        tp_group.candidate_all_reduces, int(expect_compact)
                    )
                    self.assertEqual(
                        tp_group.full_logit_gathers, int(not expect_compact)
                    )

    def test_capture_sampler_uses_capture_bucket_policy(self):
        from sglang.srt.speculative.dflash_worker_v2 import _DominoDraftSampler

        block_size = 7
        sampler = _DominoDraftSampler(
            target_embedding=self.embedding,
            lm_head_weight=self.lm_head_weight,
            prefix_gru=self.prefix_gru,
            embed_proj=self.embed_proj,
            vocab_size=31,
            block_size=block_size,
            shift_label=True,
            max_bs=8,
            candidate_pool_size=5,
        )
        for batch_size, expect_compact in ((1, False), (8, True)):
            with self.subTest(capture_bucket=batch_size, expect_compact=expect_compact):
                hidden_states = torch.randn(
                    batch_size * block_size,
                    8,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                input_ids = torch.zeros(
                    batch_size * block_size,
                    device="cuda",
                    dtype=torch.long,
                )
                proposals = torch.zeros(
                    batch_size,
                    block_size - 1,
                    device="cuda",
                    dtype=torch.long,
                )
                with mock.patch(
                    "sglang.srt.speculative.dflash_worker_v2.domino_greedy_rollout",
                    return_value=proposals,
                ) as rollout:
                    sampler(hidden_states, input_ids)
                self.assertEqual(
                    rollout.call_args.kwargs["prefer_tp_candidate_pool"],
                    expect_compact,
                )


if __name__ == "__main__":
    unittest.main()
