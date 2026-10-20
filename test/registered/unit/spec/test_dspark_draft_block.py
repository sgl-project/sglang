import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models.dspark import (
    GatedMarkovHead,
    Nemotron35VanillaMarkov,
    RNNHead,
    VanillaMarkov,
)
from sglang.srt.speculative.dspark_components import dspark_draft_sampler as ds
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    sample_draft_block,
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
    block = sample_draft_block(
        base_logits=base,
        anchor_tokens=anchor,
        draft_hidden=hidden,
        sampling_info=sampling_info,
        markov_head=head,
        noise=DraftNoise(key=3, offset=offset),
        device="cpu",
        tp_sync=SimpleNamespace(sync=lambda site, tokens: tokens),
        greedy_draft=greedy_draft,
    )
    return block, base, hidden, anchor


def _step_logits_along(head, base, hidden, anchor, tokens):
    state = head.init_state(hidden)
    prev = anchor
    rows = []
    for step in range(GAMMA):
        bias, state = head.step_bias(prev, hidden_states=hidden[:, step], state=state)
        rows.append(base[:, step] + bias)
        prev = tokens[:, step]
    return torch.stack(rows, dim=1)


class TestSampleDraftBlock(CustomTestCase):
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
            block, base, hidden, anchor = _draft(head, sampling_info, offset)
            logits = _step_logits_along(head, base, hidden, anchor, block.draft_tokens)
            self.assertEqual(block.corrected_logits.dtype, logits.dtype)
            self.assertTrue(torch.equal(block.corrected_logits, logits))
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
            self.assertEqual(block.greedy_mask.tolist(), greedy.tolist())
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
        block, base, hidden, anchor = _draft(
            head, sampling_info, offset, greedy_draft=True
        )
        logits = _step_logits_along(head, base, hidden, anchor, block.draft_tokens)
        self.assertTrue(torch.equal(block.draft_tokens, logits.float().argmax(-1)))
        self.assertIsNone(block.corrected_logits)
        self.assertTrue(block.greedy_draft)


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


class TestRepetitionDraftFallback(CustomTestCase):
    def _req(self, rid="r", *, period=(11, 22, 33), verify=40, correct=0):
        return SimpleNamespace(
            rid=rid,
            origin_input_ids=[1] * 100,
            output_ids=list(period) * max(16, 32 // len(period)),
            spec_verify_ct=verify,
            spec_num_correct_drafts=correct,
        )

    def test_overlap_phase_and_reordered_rows(self):
        """Continue the live GPU prefix when host history lags or rows reorder."""
        from sglang.srt.speculative.dspark_components.dspark_repetition import (
            RepetitionDraftFallback,
        )

        fallback = RepetitionDraftFallback()
        a, b = self._req(), self._req("b", period=(55,))
        tokens = torch.zeros((2, 7), dtype=torch.int64)
        fallback.apply(
            reqs=[a, b], prefix_lens=torch.tensor([151, 120]), draft_tokens=tokens
        )
        self.assertEqual(tokens[0].tolist(), [22, 33, 11, 22, 33, 11, 22])
        self.assertEqual(tokens[1].tolist(), [55] * 7)
        fallback.apply(
            reqs=[b, a], prefix_lens=torch.tensor([127, 154]), draft_tokens=tokens
        )
        self.assertEqual(tokens[0].tolist(), [55] * 7)
        self.assertEqual(tokens[1].tolist(), [22, 33, 11, 22, 33, 11, 22])

    def test_recovery_does_not_oscillate_and_broken_cycle_exits(self):
        """Keep recovery through success, but clear broken and finished cycles."""
        from sglang.srt.speculative.dspark_components.dspark_repetition import (
            RepetitionDraftFallback,
        )

        fallback = RepetitionDraftFallback()
        req = self._req()
        self.assertIsNotNone(fallback.select_patterns([req]))
        # Successful fallback raises the cumulative acceptance: keep using it
        # until the verified output stops repeating, rather than oscillating.
        req.spec_num_correct_drafts = 200
        self.assertIsNotNone(fallback.select_patterns([req]))
        req.output_ids.append(99)
        self.assertIsNone(fallback.select_patterns([req]))
        # A finished request's state cannot authorize a new request.
        fallback.select_patterns([])
        replacement = self._req(verify=0)
        self.assertIsNone(fallback.select_patterns([replacement]))

    def test_reused_request_id_does_not_inherit_recovery(self):
        """A replacement request must establish its own low-acceptance history."""
        from sglang.srt.speculative.dspark_components.dspark_repetition import (
            RepetitionDraftFallback,
        )

        fallback = RepetitionDraftFallback()
        self.assertIsNotNone(fallback.select_patterns([self._req()]))
        replacement = self._req(correct=100)
        self.assertIsNone(fallback.select_patterns([replacement]))

    def test_healthy_and_insufficient_history_keep_learned_candidates(self):
        """Leave learned proposals intact until repeated rejection is established."""
        from sglang.srt.speculative.dspark_components.dspark_repetition import (
            RepetitionDraftFallback,
        )

        fallback = RepetitionDraftFallback()
        healthy = self._req(correct=100)
        young = self._req("young", verify=2)
        short = self._req("short")
        short.output_ids = [7] * 8
        tokens = torch.arange(21).view(3, 7)
        original = tokens.clone()
        fallback.apply(
            reqs=[healthy, young, short],
            prefix_lens=torch.tensor([151, 151, 108]),
            draft_tokens=tokens,
        )
        self.assertTrue(torch.equal(tokens, original))

    def test_target_still_rejects_a_broken_repetition(self):
        """A periodic proposal cannot commit tokens past a target mismatch."""
        from sglang.srt.speculative.dflash_utils import (
            compute_dflash_correct_drafts_and_bonus,
        )
        from sglang.srt.speculative.dspark_components.dspark_repetition import (
            RepetitionDraftFallback,
        )

        req = self._req(period=(11, 22))
        drafts = torch.zeros((1, 7), dtype=torch.int64)
        RepetitionDraftFallback().apply(
            reqs=[req], prefix_lens=torch.tensor([131]), draft_tokens=drafts
        )
        candidates = torch.cat([torch.tensor([[22]]), drafts], dim=1)
        target = torch.tensor([[11, 22, 99, 11, 22, 11, 22, 11]])
        accepted, bonus = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates, target_predict=target
        )
        self.assertEqual(accepted.tolist(), [2])
        self.assertEqual(bonus.tolist(), [99])


def _folded_draft_model():
    """A draft model for the folded proposal."""
    model = SimpleNamespace(
        markov_head=VanillaMarkov(vocab_size=VOCAB, markov_rank=RANK),
        sample_from_anchor=True,
        confidence_head=SimpleNamespace(proj=SimpleNamespace(weight=torch.zeros(1))),
        lm_head=SimpleNamespace(org_vocab_size=VOCAB),
        compute_base_logits=Mock(
            side_effect=lambda h: (torch.zeros(h.shape[0], VOCAB), h + 1)
        ),
        compute_confidence=Mock(
            side_effect=lambda **kw: (torch.zeros(BS, GAMMA), torch.zeros(BS, GAMMA))
        ),
    )
    return model


def _build(model, *, greedy_draft, factory=None, agree=None):
    """maybe_build_draft_sampler, with factory as the megakernel's draft sampler."""
    with (
        patch.object(ds, "megakernel_draft_sampler", return_value=factory),
        patch.object(ds, "megakernel_agree", agree or Mock()),
        patch.object(ds, "get_dp_tp_group", return_value="draft tp group"),
    ):
        return ds.maybe_build_draft_sampler(
            draft_model=model,
            gamma=GAMMA,
            max_bs=2 * BS,
            device="cpu",
            tp_rank=1,
            noise=DraftNoise(key=3, offset=torch.zeros(1, dtype=torch.int64)),
            greedy_draft=greedy_draft,
            tp_sync="sync",
        )


class TestMegakernelDraftSampler(CustomTestCase):
    def test_the_megakernel_step_runs_only_on_the_batch_sizes_every_rank_serves(self):
        """A served batch size runs the megakernel's step in place of the base
        logits and the Markov block, writing SGLang's buffers; every other size runs
        SGLang's block. The served sizes are agreed over the token-sync group, since
        a rank on SGLang's block waits in a sync its peers never enter."""
        hidden, input_ids = torch.randn(BS * GAMMA, HIDDEN), torch.arange(BS * GAMMA)
        step = Mock(side_effect=lambda h, anchor: ("tokens", h * 2))
        factory = Mock(return_value=lambda bs: step if bs == BS else None)
        agree, model = Mock(), _folded_draft_model()
        with patch.object(ds, "draft_markov_block", side_effect=AssertionError):
            sampler = _build(model, greedy_draft=False, factory=factory, agree=agree)
            sampler(hidden, input_ids)
        agree.assert_called_once_with(
            (BS,), what="draft sampler batch sizes", group="draft tp group"
        )
        model.compute_base_logits.assert_not_called()
        self.assertIs(factory.call_args.kwargs["draft_tokens"], sampler.out)
        self.assertIs(factory.call_args.kwargs["step_logits"], sampler.step_logits)
        self.assertTrue(torch.equal(step.call_args.args[1], input_ids[::GAMMA]))

        declined = Mock(return_value=lambda bs: None)
        tokens = torch.full((BS, GAMMA), 7)
        with (
            patch.object(
                ds, "draft_markov_block", return_value=(tokens, None)
            ) as stock,
            patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
        ):
            sampler = _build(
                _folded_draft_model(), greedy_draft=False, factory=declined
            )
            sampler(torch.ones(BS * GAMMA, HIDDEN), input_ids)
        stock.assert_called_once()
        self.assertTrue(sampler.out[: BS * GAMMA].eq(7).all())


if __name__ == "__main__":
    unittest.main()
