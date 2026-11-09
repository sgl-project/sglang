"""Greedy EAGLE chain verify under repetition_penalty must emit what plain greedy
decoding with the same penalty emits, across mixed-chunk steps and slot reuse (#49403)."""

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.penaltylib import repetition_penalty
from sglang.srt.speculative import eagle_utils, spec_repetition_penalty
from sglang.srt.speculative.spec_repetition_penalty import SpecRepetitionState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

VOCAB_SIZE = 40
NUM_DRAFT_TOKENS = 4  # root + 3 drafts
NUM_TOKENS = 40
SLOTS = (1, 3)


def _scale(logits, scaling_penalties):
    # apply_scaling_penalties without torch.compile, so CPU CI skips a cold compile.
    logits[:] = torch.where(
        logits < 0, logits * scaling_penalties, logits / scaling_penalties
    )


def _chain_verify_greedy(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
):
    # verify_tree_greedy for chain drafts, standing in for the device kernel.
    target_predict = target_predict.reshape(-1)
    for b in range(candidates.shape[0]):
        last = int(retrieve_index[b, 0])
        accept_index[b, 0] = last
        num_correct_drafts = 0
        for k in range(1, candidates.shape[1]):
            if int(candidates[b, k]) != int(target_predict[last]):
                break
            predicts[last] = target_predict[last]
            num_correct_drafts += 1
            last = int(retrieve_index[b, k])
            accept_index[b, num_correct_drafts] = last
        predicts[last] = target_predict[last]
        accept_token_num[b] = num_correct_drafts
    return predicts, accept_index, accept_token_num


def _penalized_argmax(row, seen_tokens, penalty):
    row = row.clone()
    for token in set(seen_tokens):
        row[token] = row[token] * penalty if row[token] < 0 else row[token] / penalty
    return int(torch.argmax(row))


def _reference(table, first_token, penalty):
    output = [first_token]
    while len(output) < NUM_TOKENS:
        output.append(_penalized_argmax(table[output[-1]], output, penalty))
    return output


class _Request:
    def __init__(self, rid, slot, table, penalty, req_generation):
        req_generation[slot] += 1  # what ReqToTokenPool.alloc_rows does
        self.rid = rid
        self.kv = SimpleNamespace(req_pool_idx=slot)
        self.output_ids = []
        self.table = table
        self.penalty = penalty
        self.output = [int(torch.argmax(table[0]))]


def _sampling_info(penalties, num_rows):
    return SimpleNamespace(
        acc_additive_penalties=None,
        acc_scaling_penalties=torch.ones((num_rows, VOCAB_SIZE)),
        spec_repetition_penalties=torch.tensor(penalties)[:, None],
        logit_bias=None,
        is_all_greedy=True,
        has_watermark_candidates=False,
    )


def _verify_round(reqs, candidates, state):
    bs, k = len(reqs), len(candidates[0])
    verify_input = SimpleNamespace(
        draft_token=torch.tensor(sum(candidates, []), dtype=torch.int64),
        draft_token_num=k,
        max_tree_depth=k,
        tree_topk=1,
        retrieve_index=torch.arange(bs * k, dtype=torch.int64).view(bs, k),
        retrieve_next_token=torch.tensor([list(range(1, k)) + [-1]] * bs),
        retrieve_next_sibling=torch.full((bs, k), -1, dtype=torch.int64),
        draft_probs=None,
    )
    batch = SimpleNamespace(
        device="cpu",
        seq_lens=torch.full((bs,), 16, dtype=torch.int32),
        req_pool_indices=torch.tensor([r.kv.req_pool_idx for r in reqs]),
        sampling_info=_sampling_info([r.penalty for r in reqs], bs * k),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        return_logprob=False,
    )
    logits = torch.stack(
        [r.table[c] for r, cand in zip(reqs, candidates) for c in cand]
    )
    logits_output = SimpleNamespace(
        next_token_logits=logits.clone(), next_token_logits_vocab_shard=None
    )
    state.sync_owners(reqs)
    predict, accept_lens, accept_index = eagle_utils.eagle_sample(
        verify_input, batch, logits_output, repetition_state=state
    )
    for b, req in enumerate(reqs):
        nodes = accept_index[b, : int(accept_lens[b])].long()
        req.output += predict[nodes].tolist()


def _mixed_round(reqs, state):
    # One prefill row first, then every running request as a 1-token tail.
    roots = [r.output[-1] for r in reqs]
    batch = SimpleNamespace(
        decoding_reqs=reqs,
        sampling_info=_sampling_info([1.0] + [r.penalty for r in reqs], len(reqs) + 1),
        req_pool_indices=torch.tensor([0] + [r.kv.req_pool_idx for r in reqs]),
        input_ids=torch.tensor([7, 9] + roots),
    )
    n = len(reqs)
    state.apply_mixed_tails(
        reqs=reqs,
        slots=batch.req_pool_indices[-n:],
        roots=batch.input_ids[-n:],
        penalties=batch.sampling_info.spec_repetition_penalties[-n:],
        scaling_penalties=batch.sampling_info.acc_scaling_penalties[-n:],
    )
    for i, req in enumerate(reqs):
        logits = req.table[roots[i]].clone()[None]
        _scale(logits, batch.sampling_info.acc_scaling_penalties[i + 1 :][:1])
        req.output.append(int(torch.argmax(logits[0])))


class TestSpecRepetitionPenalty(CustomTestCase):
    def setUp(self):
        spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )
        for ctx in (
            patch.object(eagle_utils, "get_spec", return_value=spec),
            patch.object(eagle_utils, "verify_tree_greedy_func", _chain_verify_greedy),
            patch.object(repetition_penalty, "apply_scaling_penalties", _scale),
            patch.object(spec_repetition_penalty, "apply_scaling_penalties", _scale),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
            get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
        ):
            ctx.__enter__()
            self.addCleanup(ctx.__exit__, None, None, None)

    def _decode(self, penalties, draft_quality, mixed_share, seed):
        rng = random.Random(seed)
        req_generation = torch.zeros(5, dtype=torch.int64)
        state = SpecRepetitionState(
            req_generation=req_generation, vocab_size=VOCAB_SIZE, device="cpu"
        )
        # A previous owner left its history in both slots.
        state.seen[list(SLOTS)] = True
        reqs = []
        for i, (slot, penalty) in enumerate(zip(SLOTS, penalties)):
            generator = torch.Generator().manual_seed(seed * 10 + i)
            table = torch.randn(VOCAB_SIZE, VOCAB_SIZE, generator=generator) * 3
            reqs.append(_Request(f"r{i}", slot, table, penalty, req_generation))

        def draft(req, j):
            # Mostly the unpenalized greedy chain, sometimes a random token.
            if rng.random() < draft_quality:
                token = req.output[-1]
                for _ in range(j + 1):
                    token = int(torch.argmax(req.table[token]))
                return token
            return rng.randrange(VOCAB_SIZE)

        while min(len(r.output) for r in reqs) < NUM_TOKENS:
            if rng.random() < mixed_share:
                _mixed_round(reqs, state)
                continue
            candidates = [
                [r.output[-1]] + [draft(r, j) for j in range(NUM_DRAFT_TOKENS - 1)]
                for r in reqs
            ]
            _verify_round(reqs, candidates, state)
        for req in reqs:
            want = _reference(req.table, req.output[0], req.penalty)
            self.assertEqual(req.output[:NUM_TOKENS], want)

    def test_matches_plain_greedy(self):
        for penalty in (1.05, 1.3, 2.0):
            for draft_quality in (0.0, 0.5, 0.95):
                for seed in range(4):
                    with self.subTest(
                        penalty=penalty, draft_quality=draft_quality, seed=seed
                    ):
                        # The second request is a bystander without a penalty.
                        self._decode((penalty, 1.0), draft_quality, 0.0, seed)
                        self._decode((penalty, 2.0), draft_quality, 0.0, seed)

    def test_matches_plain_greedy_with_mixed_steps(self):
        for penalty in (1.05, 1.3):
            for mixed_share in (0.3, 1.0):
                for seed in range(4):
                    with self.subTest(penalty=penalty, mixed=mixed_share, seed=seed):
                        self._decode((penalty, 1.3), 0.95, mixed_share, seed)

    def test_new_row_owner_resets_and_resumed_owner_rebuilds(self):
        req_generation = torch.zeros(3, dtype=torch.int64)
        state = SpecRepetitionState(
            req_generation=req_generation, vocab_size=VOCAB_SIZE, device="cpu"
        )
        req = SimpleNamespace(
            rid="a", kv=SimpleNamespace(req_pool_idx=1), output_ids=[3, 9]
        )
        req_generation[1] += 1
        state.sync_owners([req])
        self.assertEqual(state.seen[1].nonzero().flatten().tolist(), [3, 9])
        # The same allocation keeps its row.
        state.seen[1, 4] = True
        state.sync_owners([req])
        self.assertTrue(bool(state.seen[1, 4]))
        # A new allocation of the row starts over, even with the same rid.
        req_generation[1] += 1
        req.output_ids = []
        state.sync_owners([req])
        self.assertFalse(bool(state.seen[1].any()))


if __name__ == "__main__":
    unittest.main()
