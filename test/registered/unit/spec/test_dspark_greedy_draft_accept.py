import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

BS, GAMMA, VOCAB = 3, 4, 50


class TestGreedyDraftAccept(CustomTestCase):
    def test_sampling_accept_reads_a_one_hot_draft(self):
        # The sampling accept gets a one-hot draft at the target vocab width.
        torch.manual_seed(0)
        draft_tokens = torch.randint(0, VOCAB, (BS, GAMMA))
        block = DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=None,
            greedy_mask=torch.zeros(BS, dtype=torch.bool),
            temperatures=torch.ones(BS),
            greedy_draft=True,
        )
        seen = {}

        def fake_accept(**kwargs):
            seen["draft_probs"] = kwargs["draft_probs"]
            zeros = torch.zeros(BS, dtype=torch.int64)
            return zeros, zeros, zeros

        with mock.patch.object(
            dspark_verify.AcceptSampling, "execute", side_effect=fake_accept
        ):
            dspark_verify.accept_draft_tokens(
                candidates=torch.zeros(BS, GAMMA + 1, dtype=torch.int64),
                target_logits=torch.randn(BS * (GAMMA + 1), VOCAB),
                draft_block=block,
                sampling_info=SimpleNamespace(is_all_greedy=False, is_any_greedy=False),
                draft_input=None,
                gamma=GAMMA,
                verify_num_draft_tokens=GAMMA + 1,
            )
        probs = seen["draft_probs"]
        self.assertEqual(probs.shape, (BS, GAMMA, VOCAB))
        expected = torch.nn.functional.one_hot(draft_tokens, VOCAB).float()
        self.assertTrue(torch.equal(probs, expected))


class _FakePair:
    """A TP2 group whose all-gather returns what each rank contributed."""

    world_size = 2

    def __init__(self, rank_in_group, contributions):
        self.rank_in_group = rank_in_group
        self._contributions = contributions

    def all_gather_into_tensor(self, output, local):
        self._contributions[self.rank_in_group] = local.clone()
        if len(self._contributions) == 2:
            output.copy_(torch.cat([self._contributions[0], self._contributions[1]]))


class TestAcceptTpSplit(CustomTestCase):
    def test_split_accept_matches_the_whole_batch_for_an_odd_batch(self):
        # Rank 1 holds one request fewer and pads; the merged rows must be the
        # whole-batch accept in request order.
        torch.manual_seed(0)
        rows = GAMMA + 1
        target_logits = torch.randn(BS * rows, VOCAB)
        argmax = target_logits.argmax(-1).view(BS, rows)
        candidates = torch.zeros(BS, rows, dtype=torch.int64)
        # Request i matches the target on its first i drafts.
        for i in range(BS):
            candidates[i, 1 : 1 + i] = argmax[i, :i]
            candidates[i, 1 + i :] = (argmax[i, i : rows - 1] + 1) % VOCAB
        kwargs = dict(
            candidates=candidates,
            target_logits=target_logits,
            draft_block=DraftBlockResult(
                draft_tokens=candidates[:, 1:],
                corrected_logits=None,
                greedy_mask=torch.ones(BS, dtype=torch.bool),
                temperatures=torch.ones(BS),
                greedy_draft=True,
            ),
            sampling_info=None,
            draft_input=None,
            gamma=GAMMA,
            verify_num_draft_tokens=rows,
        )
        whole = dspark_verify.accept_draft_tokens(**kwargs)
        self.assertEqual(whole[0].tolist(), list(range(BS)))
        contributions = {}
        dspark_verify.accept_draft_tokens_tp_split(
            group=_FakePair(1, contributions), **kwargs
        )
        split = dspark_verify.accept_draft_tokens_tp_split(
            group=_FakePair(0, contributions), **kwargs
        )
        for got, want in zip(split, whole):
            self.assertEqual(got.dtype, want.dtype)
            self.assertTrue(torch.equal(got, want))

    def test_shard_exchange_returns_each_ranks_rows_of_the_full_logits(self):
        # Odd batch: rank 0 pads what it sends, rank 1 reads fewer rows than it gets.
        torch.manual_seed(0)
        rows = GAMMA + 1
        full = torch.randn(BS * rows, VOCAB)
        shards = full.to(torch.bfloat16).view(-1, 2, VOCAB // 2).unbind(1)
        want = full.to(torch.bfloat16).float()
        half_rows = (BS + 1) // 2 * rows
        for rank, want_rows in ((0, want[:half_rows]), (1, want[half_rows:])):
            contributions = {}
            peer = 1 - rank
            dspark_verify._exchange_vocab_shards(
                _FakePair(peer, contributions),
                shards[peer],
                bs=BS,
                rows_per_request=rows,
            )
            got = dspark_verify._exchange_vocab_shards(
                _FakePair(rank, contributions),
                shards[rank],
                bs=BS,
                rows_per_request=rows,
            )
            self.assertTrue(torch.equal(got, want_rows))


if __name__ == "__main__":
    unittest.main()
