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
    def test_a_plugin_step_replaces_only_the_base_logits_and_markov_block(self):
        hidden = torch.randn(BS * GAMMA, HIDDEN)
        input_ids = torch.arange(BS * GAMMA)
        for greedy_draft in (False, True):
            factory = Mock()
            step = Mock(side_effect=lambda h, anchor: ("tokens", h * 2))
            factory.return_value = lambda bs: step if bs == BS else None
            model = _folded_draft_model()
            with (
                self.subTest(greedy_draft=greedy_draft),
                patch.object(ds, "draft_markov_block", side_effect=AssertionError),
            ):
                sampler = _build(model, greedy_draft=greedy_draft, factory=factory)
                sampler(hidden, input_ids)
            model.compute_base_logits.assert_not_called()
            # the plugin writes into SGLang's buffers; a temperature-0 row it skips
            # must read as finite logits in a mixed accept
            kwargs = factory.call_args.kwargs
            self.assertIs(kwargs["draft_tokens"], sampler.out)
            self.assertIs(kwargs["noise"], sampler._noise)
            self.assertIs(kwargs["step_temperatures"], sampler.params.step_temperatures)
            self.assertEqual(kwargs["greedy_draft"], greedy_draft)
            if greedy_draft:
                self.assertIsNone(kwargs["step_logits"])
            else:
                self.assertIs(kwargs["step_logits"], sampler.step_logits)
                self.assertEqual(sampler.step_logits.shape, (2 * BS, GAMMA, VOCAB))
                self.assertFalse(sampler.step_logits.any())
            self.assertTrue(torch.equal(step.call_args.args[1], input_ids[::GAMMA]))
            confidence = model.compute_confidence.call_args.kwargs
            self.assertEqual(confidence["draft_tokens"], "tokens")
            self.assertTrue(torch.equal(confidence["confidence_features"], hidden * 2))

    def test_without_a_plugin_step_the_stock_proposal_runs(self):
        """Unset (no megakernel draft sampler), and a batch size the megakernel declines:
        the stock base logits and Markov block run with today's arguments."""
        declined = lambda **kw: lambda bs: None  # noqa: E731
        for factory in (None, declined):
            model = _folded_draft_model()
            tokens = torch.full((BS, GAMMA), 7)
            with (
                patch.object(
                    ds, "draft_markov_block", return_value=(tokens, None)
                ) as stock,
                patch.object(torch.cuda, "is_current_stream_capturing", lambda: False),
            ):
                sampler = _build(model, greedy_draft=False, factory=factory)
                if factory is None:
                    self.assertIsNone(sampler.step_logits)  # sized on the first call
                sampler(torch.ones(BS * GAMMA, HIDDEN), torch.arange(BS * GAMMA))
            kwargs = stock.call_args.kwargs
            self.assertEqual(
                kwargs.keys(),
                {
                    "markov_head",
                    "base_logits",
                    "anchor_tokens",
                    "draft_hidden",
                    "step_temperatures",
                    "noise",
                    "tp_sync",
                    "logits_out",
                    "greedy_draft",
                },
            )
            self.assertEqual(kwargs["base_logits"].shape, (BS, GAMMA, VOCAB))
            self.assertTrue(
                torch.equal(kwargs["anchor_tokens"], torch.arange(0, 20, 5))
            )
            self.assertEqual(
                kwargs["logits_out"].data_ptr(), sampler.step_logits.data_ptr()
            )
            self.assertTrue(sampler.out[: BS * GAMMA].eq(7).all())
            features = model.compute_confidence.call_args.kwargs["confidence_features"]
            self.assertTrue(features.eq(2).all())

    def test_every_rank_serves_the_same_draft_batch_sizes(self):
        """A rank whose megakernel declines a size runs SGLang's block, whose per-step
        token sync its peers on the megakernel step never enter: the sizes must agree,
        over the group that sync uses."""
        agree = Mock()
        factory = Mock(return_value=lambda bs: "step" if bs in (1, BS) else None)
        _build(_folded_draft_model(), greedy_draft=True, factory=factory, agree=agree)
        agree.assert_called_once_with(
            (1, BS), what="draft sampler batch sizes", group="draft tp group"
        )

    def test_ranks_serving_different_draft_batch_sizes_are_refused(self):
        from sglang.srt.megakernels import runtime

        factory = Mock(return_value=lambda bs: "step" if bs == 1 else None)
        group = SimpleNamespace(world_size=2, cpu_group=None)

        def gather(values, value, group):
            values[:] = [value, (1, 2)]  # the other rank also serves batch 2

        with (
            patch.object(runtime.dist, "all_gather_object", gather),
            patch.object(ds, "megakernel_draft_sampler", return_value=factory),
            patch.object(ds, "get_dp_tp_group", return_value=group),
            self.assertRaisesRegex(RuntimeError, r"rank 1: \(1, 2\)"),
        ):
            ds.maybe_build_draft_sampler(
                draft_model=_folded_draft_model(),
                gamma=GAMMA,
                max_bs=2 * BS,
                device="cpu",
                tp_rank=0,
                noise=DraftNoise(key=3, offset=torch.zeros(1, dtype=torch.int64)),
                greedy_draft=True,
                tp_sync="sync",
            )


if __name__ == "__main__":
    unittest.main()
