import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

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


class TestMarkovCandidates(CustomTestCase):
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
        for topk in (0, 4, 5):
            with self.subTest(topk=topk):
                head.configure_candidates(topk)
                self.assertIsNone(head.candidate_ids)
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

        with self.assertRaisesRegex(ValueError, "Changing DSpark d2t"):
            model.load_weights([("d2t", mapping + 1)])
        model._vocab_attached = False
        with self.assertRaisesRegex(ValueError, "missing required"):
            name = "markov_head.markov_w1.weight"
            model.load_weights([(name, weights[name])])


if __name__ == "__main__":
    unittest.main()
