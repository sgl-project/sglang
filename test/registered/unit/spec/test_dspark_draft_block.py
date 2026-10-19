import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.dspark import (
    GatedMarkovHead,
    Nemotron35VanillaMarkov,
    RNNHead,
    VanillaMarkov,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    resolve_draft_params,
    run_draft_proposal,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB, RANK, HIDDEN, BS, GAMMA = 211, 8, 16, 4, 5


def _draft(head, sampling_info, offset, greedy_draft=False):
    torch.manual_seed(0)
    base = torch.randn(BS, GAMMA, VOCAB).to(torch.bfloat16)
    hidden = torch.randn(BS, GAMMA, HIDDEN).to(torch.bfloat16)
    anchor = torch.randint(0, VOCAB, (BS,))
    model = SimpleNamespace(
        markov_head=head,
        confidence_head=None,
        sample_from_anchor=True,
        compute_base_logits=lambda h: (base.view(BS * GAMMA, -1), h),
    )
    params = resolve_draft_params(sampling_info=sampling_info, bs=BS, device="cpu")
    proposal = run_draft_proposal(
        model=model,
        hidden_states=hidden.view(BS * GAMMA, HIDDEN),
        anchor_tokens=anchor,
        bs=BS,
        gamma=GAMMA,
        step_temperatures=params.step_temperatures,
        noise=DraftNoise(key=3, offset=offset),
        tp_sync=SimpleNamespace(sync=lambda site, tokens: tokens),
        greedy_draft=greedy_draft,
    )
    return proposal, params, base, hidden, anchor


def _step_logits_along(head, base, hidden, anchor, tokens):
    state = head.init_state(hidden)
    prev = anchor
    rows = []
    for step in range(GAMMA):
        bias, state = head.step_bias(prev, hidden_states=hidden[:, step], state=state)
        rows.append(base[:, step] + bias)
        prev = tokens[:, step]
    return torch.stack(rows, dim=1)


class TestRunDraftProposal(CustomTestCase):
    def test_greedy_rows_take_the_argmax_and_keep_their_logits(self):
        for head in (
            VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK),
            RNNHead(vocab_size=VOCAB, markov_rank=RANK, hidden_size=HIDDEN),
        ):
            head = head.to(torch.bfloat16)
            # top_k <= 1 marks a greedy request; rows 0 and 2 are greedy.
            sampling_info = SimpleNamespace(
                is_all_greedy=False,
                temperatures=torch.tensor([[0.7], [1.0], [1.0], [1.3]]),
                top_ks=torch.tensor([1, 50, 1, 50]),
            )
            offset = torch.zeros(1, dtype=torch.int64)
            block, _, base, hidden, anchor = _draft(head, sampling_info, offset)
            logits = _step_logits_along(head, base, hidden, anchor, block.draft_tokens)
            self.assertEqual(block.step_logits.dtype, logits.dtype)
            self.assertTrue(torch.equal(block.step_logits, logits))
            greedy = torch.tensor([True, False, True, False])
            self.assertTrue(
                torch.equal(
                    block.draft_tokens[greedy], logits[greedy].float().argmax(-1)
                )
            )
            # The sampled rows drew noise: they leave the argmax somewhere.
            self.assertFalse(
                torch.equal(
                    block.draft_tokens[~greedy], logits[~greedy].float().argmax(-1)
                )
            )
            self.assertEqual(offset.item(), 1)

    def test_greedy_draft_takes_every_argmax_and_keeps_no_logits(self):
        head = VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK).to(torch.bfloat16)
        # Sampling requests too: a greedy draft ignores their temperatures.
        sampling_info = SimpleNamespace(
            is_all_greedy=False,
            temperatures=torch.tensor([[0.7], [1.0], [1.0], [1.3]]),
            top_ks=torch.tensor([50, 50, 1, 50]),
        )
        offset = torch.zeros(1, dtype=torch.int64)
        block, _, base, hidden, anchor = _draft(
            head, sampling_info, offset, greedy_draft=True
        )
        logits = _step_logits_along(head, base, hidden, anchor, block.draft_tokens)
        self.assertTrue(torch.equal(block.draft_tokens, logits.float().argmax(-1)))
        self.assertIsNone(block.step_logits)


class TestFusedGreedyEligibility(CustomTestCase):
    def test_only_the_plain_w2_gemv_head_fuses(self):
        # A head that changes the bias must keep the shared op.
        heads = (
            (VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK), True),
            (
                Nemotron35VanillaMarkov(
                    vocab_size=VOCAB, markov_rank=RANK, quant_config=None, prefix=""
                ),
                False,
            ),
            (
                GatedMarkovHead(vocab_size=VOCAB, markov_rank=RANK, hidden_size=HIDDEN),
                False,
            ),
            (RNNHead(vocab_size=VOCAB, markov_rank=RANK, hidden_size=HIDDEN), False),
        )
        for head, fuses in heads:
            with self.subTest(head=type(head).__name__):
                self.assertIs(head.greedy_w2_weight() is not None, fuses)


if __name__ == "__main__":
    unittest.main()
