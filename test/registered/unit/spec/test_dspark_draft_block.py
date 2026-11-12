import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.kernel_plugins.nvidia import dspark_binding
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _model_load_weights_direct,
)
from sglang.srt.models.dspark import (
    GatedMarkovHead,
    Nemotron35VanillaMarkov,
    Qwen3DSparkModel,
    RNNHead,
    VanillaMarkov,
)
from sglang.srt.speculative.dspark_components import dspark_draft_sampler as ds
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftNoise,
    draft_markov_block,
    sample_draft_block,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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
    def test_scaled_reduced_vocab_keeps_target_space_logits(self):
        torch.manual_seed(7)
        draft_vocab = 17
        mapped_ids = torch.randperm(VOCAB)[:draft_vocab]
        offsets = mapped_ids - torch.arange(draft_vocab)
        for head_type in (VanillaMarkov, GatedMarkovHead, RNNHead):
            for greedy_draft in (False, True):
                with self.subTest(head=head_type.__name__, greedy=greedy_draft):
                    kwargs = (
                        {} if head_type is VanillaMarkov else {"hidden_size": HIDDEN}
                    )
                    head = head_type(
                        vocab_size=VOCAB,
                        draft_vocab_size=draft_vocab,
                        markov_rank=RANK,
                        logit_scale=0.5,
                        **kwargs,
                    )
                    head.configure_target_vocab(VOCAB, offsets)
                    base = torch.randn(1, GAMMA, draft_vocab)
                    hidden = torch.randn(1, GAMMA, HIDDEN)
                    anchor = torch.tensor([VOCAB - 1])
                    block = sample_draft_block(
                        base_logits=base,
                        anchor_tokens=anchor,
                        draft_hidden=hidden,
                        sampling_info=None,
                        markov_head=head,
                        noise=DraftNoise(
                            key=3, offset=torch.zeros(1, dtype=torch.int64)
                        ),
                        device="cpu",
                        tp_sync=SimpleNamespace(sync=lambda site, tokens: tokens),
                        greedy_draft=greedy_draft,
                    )
                    state = head.init_state(hidden)
                    previous = anchor
                    for step in range(GAMMA):
                        bias, state = head.step_bias(
                            previous, hidden_states=hidden[:, step], state=state
                        )
                        expected = torch.full((1, VOCAB), -torch.inf)
                        expected[:, mapped_ids] = base[:, step] + 0.5 * bias
                        previous = expected.argmax(-1)
                        torch.testing.assert_close(
                            block.draft_tokens[:, step], previous
                        )
                        if not greedy_draft:
                            torch.testing.assert_close(
                                block.corrected_logits[:, step], expected
                            )
                    if greedy_draft:
                        self.assertIsNone(block.corrected_logits)

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


class TestMarkovCandidates(CustomTestCase):
    def test_block_preserves_candidate_logits_with_padded_rows(self):
        torch.manual_seed(13)
        head = VanillaMarkov(vocab_size=11, draft_vocab_size=7, markov_rank=5)
        head.logit_scale = -0.5
        mapping = torch.tensor([8, 5, 9, 1, 7, 3, 6])
        head.configure_target_vocab(11, mapping - torch.arange(7))
        head.configure_candidates(3)
        base = torch.randn(3, 4, 7)
        for greedy in (False, True):
            tokens, logits = draft_markov_block(
                markov_head=head,
                base_logits=base,
                anchor_tokens=torch.tensor([2, 4, 99999]),
                draft_hidden=None,
                step_temperatures=torch.tensor([0.7, 1.2, 0.0]),
                noise=DraftNoise(key=3, offset=torch.zeros(1, dtype=torch.int64)),
                tp_sync=SimpleNamespace(sync=lambda site, tokens: tokens),
                greedy_draft=greedy,
                num_valid=torch.tensor(2),
            )
            previous = torch.tensor([2, 4, 0])
            for step in range(4):
                expected = head.candidate_logits(previous, base[:, step])
                if greedy:
                    torch.testing.assert_close(tokens[:, step], expected.argmax(-1))
                else:
                    torch.testing.assert_close(logits[:, step], expected)
                    self.assertTrue(
                        expected.gather(1, tokens[:, step, None]).isfinite().all()
                    )
                previous = tokens[:, step]

    def test_union_preserves_scores_and_target_ids(self):
        """Overlapping sets must not double bias or double the proposal mass."""
        head = VanillaMarkov(vocab_size=8, draft_vocab_size=4, markov_rank=1).to(
            torch.bfloat16
        )
        with torch.no_grad():
            head.markov_w1.weight.fill_(1)
            head.markov_w2.weight.copy_(torch.tensor([[1], [4], [3], [-2]]))
        target_ids = torch.tensor([3, 6, 1, 7])
        head.configure_target_vocab(8, target_ids - torch.arange(4))
        prev = torch.tensor([2])
        base = torch.tensor([[6, 5, 0, 1]], dtype=torch.bfloat16)
        for scale, scores in (
            (1.0, [7, 9, 3, -torch.inf]),
            (-1.0, [5, 1, -torch.inf, 3]),
        ):
            with self.subTest(scale=scale):
                head.logit_scale = scale
                head.configure_candidates(2)
                logits = head.candidate_logits(prev, base)
                expected = torch.full((1, 8), -torch.inf)
                expected[0, target_ids] = torch.tensor(scores)
                self.assertEqual(logits.dtype, torch.float32)
                torch.testing.assert_close(logits, expected)
                self.assertEqual(head.candidate_biases.dtype, torch.float32)
                torch.testing.assert_close(
                    head.candidate_logits(prev, base, base.topk(2, sorted=False)),
                    expected,
                )

        # Cached scores are authoritative for static IDs, including overlap.
        head.candidate_biases[2] = torch.where(head.candidate_ids[2] == 0, 20.0, 10.0)
        expected[0, target_ids] = torch.tensor([26, 1, -torch.inf, 11])
        torch.testing.assert_close(head.candidate_logits(prev, base), expected)

        head.logit_scale = 0
        head.refresh_candidates()
        logits = head.candidate_logits(prev, base)
        finite = torch.isfinite(logits)
        torch.testing.assert_close(
            logits[finite], head.to_target_logits(base)[finite].float()
        )
        torch.testing.assert_close(logits[0, target_ids[:2]], base[0, :2].float())

        head.logit_scale = 1
        head.refresh_candidates()
        logits = head.candidate_logits(prev, torch.full_like(base, -torch.inf))
        expected = torch.full((1, 8), -torch.inf)
        expected[0, 1] = 0  # Lowest target ID in the union, not target token zero.
        torch.testing.assert_close(logits, expected)

    def test_disabled_and_full_vocabulary_candidates_use_dense_path(self):
        head = VanillaMarkov(vocab_size=4, markov_rank=1)
        head.configure_candidates(2)
        for topk in (0, 4, 5):
            with self.subTest(topk=topk):
                head.configure_candidates(topk)
                self.assertIsNone(head.candidate_ids)
                self.assertIsNone(head.candidate_biases)
                self.assertIsNone(
                    head.candidate_logits(torch.tensor([0]), torch.zeros(1, 4))
                )


class TestDraftWeightUpdates(CustomTestCase):
    def test_checkpoint_can_be_updated_in_parameter_buckets(self):
        """Online updates may omit other weights and the unchanged vocab mapping."""
        model = Qwen3DSparkModel.__new__(Qwen3DSparkModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            draft_vocab_size=4,
            hidden_size=3,
            _sglang_speculators_dspark_normalized=True,
        )
        model.markov_head = VanillaMarkov(vocab_size=4, markov_rank=2)
        model.embed_tokens = torch.nn.Embedding(4, 3)
        model.lm_head = torch.nn.Linear(3, 4, bias=False)
        model.norm = torch.nn.Linear(3, 3, bias=False)
        model.layers = torch.nn.ModuleList()
        model.confidence_head = torch.nn.Linear(3, 1)
        model.input_vocab_size = model.draft_vocab_size = 4
        model.has_own_embed_tokens = model.has_own_lm_head = True
        model.is_nemotron_35_draft = False
        model.projector_type = None
        model.logit_scale = 1.0
        model._vocab_attached = False
        weights = {
            name: torch.zeros_like(param) for name, param in model.named_parameters()
        }
        mapping = torch.zeros(4, dtype=torch.int64)
        model.load_weights([*weights.items(), ("d2t", mapping)])
        model._vocab_attached = True

        model.markov_head.configure_candidates(2)
        candidate_ptr = model.markov_head.candidate_ids.data_ptr()
        bias_ptr = model.markov_head.candidate_biases.data_ptr()
        for index, (name, param) in enumerate(model.named_parameters(), start=1):
            with self.subTest(weight=name):
                weights[name] = torch.full_like(param, index)
                if name == "markov_head.markov_w2.weight":
                    weights[name] *= torch.arange(4)[:, None]
                model.load_weights([(name, weights[name])])
                for key, actual in model.named_parameters():
                    torch.testing.assert_close(actual, weights[key])
                torch.testing.assert_close(model.draft_id_to_target_id, mapping)
                self.assertEqual(
                    model.markov_head.candidate_ids.data_ptr(), candidate_ptr
                )
                self.assertEqual(
                    model.markov_head.candidate_biases.data_ptr(), bias_ptr
                )
                expected_bias = (
                    model.markov_head.markov_w1.weight.float()
                    @ model.markov_head.markov_w2.weight.float().T
                ).gather(1, model.markov_head.candidate_ids)
                torch.testing.assert_close(
                    model.markov_head.candidate_biases, expected_bias
                )
        torch.testing.assert_close(
            model.markov_head.candidate_ids, torch.tensor([[3, 2]]).expand(4, -1)
        )

        # Direct tensor copies bypass load_weights; session finalization refreshes.
        name = "markov_head.markov_w1.weight"
        updated = torch.full_like(weights[name], -11)
        _model_load_weights_direct(model, [(name, updated)])
        with patch(
            "sglang.srt.model_executor.model_runner_components.weight_updater."
            "DefaultModelLoader.postprocess_weights"
        ):
            WeightUpdater.end_weight_update(
                SimpleNamespace(get_model=lambda: model, device="cpu"),
                run_post_load=False,
            )
        torch.testing.assert_close(
            model.markov_head.candidate_ids, torch.tensor([[0, 1]]).expand(4, -1)
        )
        self.assertEqual(model.markov_head.candidate_ids.data_ptr(), candidate_ptr)
        self.assertEqual(model.markov_head.candidate_biases.data_ptr(), bias_ptr)
        expected_bias = (updated @ weights["markov_head.markov_w2.weight"].T).gather(
            1, model.markov_head.candidate_ids
        )
        torch.testing.assert_close(model.markov_head.candidate_biases, expected_bias)

        with self.assertRaisesRegex(ValueError, "Changing DSpark d2t"):
            model.load_weights([("d2t", mapping + 1)])
        model._vocab_attached = False
        with self.assertRaisesRegex(ValueError, "missing required"):
            name = "markov_head.markov_w1.weight"
            model.load_weights([(name, weights[name])])


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
        patch.object(dspark_binding, "megakernel_draft_sampler", return_value=factory),
        patch.object(dspark_binding, "megakernel_agree", agree or Mock()),
        patch.object(dspark_binding, "get_dp_tp_group", return_value="draft tp group"),
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

    def test_absent_provider_uses_the_always_present_stock_strategy(self):
        tokens = torch.full((BS, GAMMA), 11)
        input_ids = torch.arange(BS * GAMMA)
        with patch.object(
            ds, "draft_markov_block", return_value=(tokens, None)
        ) as stock:
            sampler = _build(_folded_draft_model(), greedy_draft=True)
            sampler(torch.ones(BS * GAMMA, HIDDEN), input_ids)
        stock.assert_called_once()
        self.assertTrue(sampler.out[: BS * GAMMA].eq(11).all())


class TestDraftParamsRestage(CustomTestCase):
    def test_the_draft_params_are_restaged_only_when_their_inputs_change(self):
        """The draft graph's sampling params are copied from the batch's temperature /
        top-k tensors; a step whose sampler, bs and tensors are the ones last staged
        skips the copy, and any change (a rebound tensor after filter / merge, a new
        bs, another sampler) restages."""

        proposer = SimpleNamespace(
            _staged_params_key=None,
            num_staged_params_skips=0,
            num_staged_params_calls=0,
        )
        staged = []
        sampler = SimpleNamespace(
            stage_sampling_params=lambda **kw: staged.append(
                (kw["bs"], kw["sampling_info"])
            )
        )
        info = SimpleNamespace(temperatures=torch.ones(4), top_ks=torch.ones(4))

        def stage(sampler, bs, sampling_info):
            DraftBlockProposer._stage_sampling_params(
                proposer, sampler, bs=bs, sampling_info=sampling_info
            )

        stage(sampler, 4, info)
        stage(sampler, 4, info)  # unchanged: skipped
        info.temperatures = info.temperatures[:3]  # rebound, as filter_batch does
        stage(sampler, 3, info)
        stage(sampler, 2, info)  # another bs
        other = SimpleNamespace(stage_sampling_params=sampler.stage_sampling_params)
        stage(other, 2, info)  # another sampler
        stage(other, 2, None)
        stage(other, 2, None)  # no sampling info twice: skipped
        self.assertEqual([bs for bs, _ in staged], [4, 3, 2, 2, 2])
        self.assertEqual(proposer.num_staged_params_skips, 2)
        self.assertEqual(proposer.num_staged_params_calls, 7)


if __name__ == "__main__":
    unittest.main()
