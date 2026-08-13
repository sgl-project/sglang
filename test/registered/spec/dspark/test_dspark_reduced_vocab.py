"""Reduced-vocabulary (independent-head) DSpark draft support.

Speculators-trained DSpark checkpoints (e.g. RedHatAI/*-speculator.dspark) ship
an independent, reduced-vocabulary lm_head plus a d2t draft->target map, instead
of sharing the target head. These tests pin the invariants that make that work
and that keep the ordinary full-vocab/shared-head path byte-for-byte unchanged:

  - markov_w1 stays target-vocab-wide (its input is the previously sampled
    target id) while markov_w2 / lm_head shrink to the draft vocab;
  - a draft-space sampled id is mapped to a target id before it is stored AND
    before it conditions the next Markov step (markov_w1);
  - the markov-corrected block is lifted into target-vocab columns (unmapped
    columns -inf) for target-space rejection sampling;
  - inconsistent / missing reduced-vocab state fails loudly at load.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.models.dspark import (
    GatedMarkovHead,
    RNNHead,
    _is_dspark_d2t_weight,
    _is_dspark_t2d_weight,
    build_independent_lm_head,
    build_markov_head,
    scatter_draft_logits_to_target,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
    resolve_draft_vocab_size,
)
from sglang.srt.speculative.dspark_components.dspark_draft import sample_draft_block
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.utils.hf_transformers.config import (
    _normalize_speculators_draft_config_dict,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TARGET_VOCAB = 100
DRAFT_VOCAB = 8
MARKOV_RANK = 4
HIDDEN = 16


def _markov_config(*, head_type="vanilla"):
    return SimpleNamespace(
        markov_rank=MARKOV_RANK,
        markov_head_type=head_type,
        vocab_size=TARGET_VOCAB,
        hidden_size=HIDDEN,
    )


def _reduced_vanilla_head(*, d2t_delta=None):
    """A reduced VanillaMarkov with markov weights zeroed, so apply_step_logits
    is the identity on base_logits (bias == 0). Lets a test drive the argmax
    purely from crafted base_logits and read back the exact mapped ids."""
    head = build_markov_head(_markov_config(), draft_vocab_size=DRAFT_VOCAB)
    with torch.no_grad():
        head.markov_w1.weight.zero_()
        head.markov_w2.weight.zero_()
    if d2t_delta is None:
        # target = draft + 2  -> a non-identity, easily checked mapping.
        d2t_delta = torch.full((DRAFT_VOCAB,), 2, dtype=torch.long)
    head.load_draft_to_target(d2t_delta)
    return head


class TestMarkovHeadVocabGeometry(CustomTestCase):
    """markov_w1 input == target vocab, markov_w2 output == draft vocab."""

    def test_vanilla_reduced_is_asymmetric(self):
        head = build_markov_head(_markov_config(), draft_vocab_size=DRAFT_VOCAB)
        self.assertEqual(
            tuple(head.markov_w1.weight.shape), (TARGET_VOCAB, MARKOV_RANK)
        )
        self.assertEqual(tuple(head.markov_w2.weight.shape), (DRAFT_VOCAB, MARKOV_RANK))
        self.assertTrue(head.reduced_vocab)
        self.assertEqual(head.draft_vocab_size, DRAFT_VOCAB)

    def test_full_vocab_is_symmetric(self):
        head = build_markov_head(_markov_config(), draft_vocab_size=None)
        self.assertEqual(
            tuple(head.markov_w1.weight.shape), (TARGET_VOCAB, MARKOV_RANK)
        )
        self.assertEqual(
            tuple(head.markov_w2.weight.shape), (TARGET_VOCAB, MARKOV_RANK)
        )
        self.assertFalse(head.reduced_vocab)
        self.assertIsNone(head.draft_to_target)

    def test_draft_vocab_equal_to_target_is_not_reduced(self):
        # A checkpoint that sets draft_vocab_size == vocab_size shares the full
        # vocab (no reduction, no independent head, no mapping).
        head = build_markov_head(_markov_config(), draft_vocab_size=TARGET_VOCAB)
        self.assertFalse(head.reduced_vocab)
        self.assertIsNone(head.draft_to_target)

    def test_gated_and_rnn_reduced_shrink_only_w2(self):
        for head_type, cls in (("gated", GatedMarkovHead), ("rnn", RNNHead)):
            with self.subTest(head_type=head_type):
                head = build_markov_head(
                    _markov_config(head_type=head_type), draft_vocab_size=DRAFT_VOCAB
                )
                self.assertIsInstance(head, cls)
                self.assertEqual(head.markov_w1.weight.shape[0], TARGET_VOCAB)
                self.assertEqual(head.markov_w2.weight.shape[0], DRAFT_VOCAB)


class TestDraftToTargetMapping(CustomTestCase):
    def test_d2t_delta_becomes_absolute_map(self):
        head = build_markov_head(_markov_config(), draft_vocab_size=DRAFT_VOCAB)
        delta = torch.arange(DRAFT_VOCAB, dtype=torch.long)  # target = draft + draft
        head.load_draft_to_target(delta)
        expected = torch.arange(DRAFT_VOCAB) + delta
        self.assertTrue(torch.equal(head.draft_to_target, expected))
        self.assertTrue(head.draft_to_target_loaded)

    def test_map_sampled_to_target_applies_map(self):
        head = _reduced_vanilla_head()  # target = draft + 2
        sampled = torch.tensor([0, 1, 7])
        self.assertTrue(
            torch.equal(head.map_sampled_to_target(sampled), torch.tensor([2, 3, 9]))
        )

    def test_full_vocab_map_is_identity(self):
        head = build_markov_head(_markov_config(), draft_vocab_size=None)
        sampled = torch.tensor([0, 5, 99])
        self.assertTrue(torch.equal(head.map_sampled_to_target(sampled), sampled))

    def test_wrong_length_d2t_raises(self):
        head = build_markov_head(_markov_config(), draft_vocab_size=DRAFT_VOCAB)
        with self.assertRaises(ValueError):
            head.load_draft_to_target(torch.zeros(DRAFT_VOCAB + 1, dtype=torch.long))

    def test_out_of_range_d2t_raises(self):
        head = build_markov_head(_markov_config(), draft_vocab_size=DRAFT_VOCAB)
        # Maps draft id 0 to target id TARGET_VOCAB (>= vocab) -> must reject.
        delta = torch.zeros(DRAFT_VOCAB, dtype=torch.long)
        delta[0] = TARGET_VOCAB
        with self.assertRaises(ValueError):
            head.load_draft_to_target(delta)

    def test_duplicate_d2t_target_ids_raise(self):
        head = build_markov_head(_markov_config(), draft_vocab_size=DRAFT_VOCAB)
        delta = torch.zeros(DRAFT_VOCAB, dtype=torch.long)
        delta[1] = -1  # draft ids 0 and 1 would both map to target id 0
        with self.assertRaises(ValueError):
            head.load_draft_to_target(delta)

    def test_d2t_on_full_vocab_head_raises(self):
        head = build_markov_head(_markov_config(), draft_vocab_size=None)
        with self.assertRaises(ValueError):
            head.load_draft_to_target(torch.zeros(TARGET_VOCAB, dtype=torch.long))

    def test_weight_name_classification(self):
        # d2t is loaded into the map; t2d (inverse, training-only) is dropped.
        self.assertTrue(_is_dspark_d2t_weight("d2t"))
        self.assertTrue(_is_dspark_d2t_weight("markov_head.d2t"))
        self.assertTrue(_is_dspark_t2d_weight("t2d"))
        self.assertTrue(_is_dspark_t2d_weight("model.t2d"))
        # Guard the negative branch: a normal weight is neither.
        self.assertFalse(_is_dspark_d2t_weight("lm_head.weight"))
        self.assertFalse(_is_dspark_t2d_weight("lm_head.weight"))
        self.assertFalse(_is_dspark_d2t_weight("t2d"))


class TestSequentialGreedySampling(CustomTestCase):
    """Greedy sample_block returns TARGET ids and feeds TARGET ids to the next
    Markov step (markov_w1's input vocab is the target vocab)."""

    def test_greedy_returns_target_ids_and_feeds_target_ids(self):
        head = _reduced_vanilla_head()  # target = draft + 2, bias == 0
        recorded = []
        orig_prev_emb = head.get_prev_embeddings

        def spy(token_ids):
            recorded.append(token_ids.clone())
            return orig_prev_emb(token_ids)

        head.get_prev_embeddings = spy

        # step 0 argmax -> draft id 1 (-> target 3); step 1 argmax -> draft id 0
        # (-> target 2). bias is 0, so argmax is purely from base_logits.
        base_logits = torch.zeros(1, 2, DRAFT_VOCAB)
        base_logits[0, 0, 1] = 9.0
        base_logits[0, 1, 0] = 9.0
        anchor = torch.tensor([7])  # a target id

        sampled, corrected = head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=None,
            sampler=lambda logits, i: torch.argmax(logits, dim=-1),
        )

        # Stored ids are target-space (mapped), not the raw draft ids 1/0.
        self.assertTrue(torch.equal(sampled, torch.tensor([[3, 2]])))
        # markov_w1 saw the anchor, then the mapped target id from step 0.
        self.assertTrue(torch.equal(recorded[0], anchor))
        self.assertTrue(torch.equal(recorded[1], torch.tensor([3])))
        # corrected logits stay draft-space (they get lifted to target later).
        self.assertEqual(corrected.shape, (1, 2, DRAFT_VOCAB))

    def test_eager_sample_draft_block_maps_to_target(self):
        head = _reduced_vanilla_head()
        base_logits = torch.zeros(2, 3, DRAFT_VOCAB)
        base_logits[:, :, 4] = 9.0  # every step argmax -> draft id 4 -> target 6
        result = sample_draft_block(
            base_logits=base_logits,
            anchor_tokens=torch.tensor([7, 8]),
            draft_hidden=torch.zeros(2, 3, HIDDEN),
            sampling_info=None,
            markov_head=head,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.all(result.draft_tokens == 6))
        self.assertEqual(result.corrected_logits.shape, (2, 3, DRAFT_VOCAB))


class TestCorrectedScatterToTarget(CustomTestCase):
    """Reduced-vocab probabilistic verify needs corrected logits in TARGET
    columns; unmapped columns must be -inf (zero softmax mass)."""

    def test_scatter_places_draft_columns_and_fills_neg_inf(self):
        draft_to_target = torch.tensor([2, 5, 7])  # draft vocab 3 -> target vocab 10
        draft_logits = torch.tensor([[[1.0, 2.0, 3.0]]])  # [1, 1, 3]
        out = torch.empty(1, 1, 10)
        scattered = scatter_draft_logits_to_target(
            draft_logits, draft_to_target=draft_to_target, out=out
        )
        self.assertEqual(scattered.shape, (1, 1, 10))
        # Mapped target columns carry the draft logits.
        self.assertEqual(scattered[0, 0, 2].item(), 1.0)
        self.assertEqual(scattered[0, 0, 5].item(), 2.0)
        self.assertEqual(scattered[0, 0, 7].item(), 3.0)
        # Every other (unmapped) target column is -inf.
        mapped = {2, 5, 7}
        for col in range(10):
            if col not in mapped:
                self.assertEqual(scattered[0, 0, col].item(), float("-inf"))

    def test_softmax_of_scattered_matches_draft_softmax_on_mapped_ids(self):
        # The derived property that makes target-space rejection sampling and the
        # block-accept estimator correct: softmax over the scattered target row,
        # read at the mapped id, equals the draft-space softmax at the draft id.
        head = _reduced_vanilla_head()  # target = draft + 2
        draft_logits = torch.randn(1, 1, DRAFT_VOCAB)
        out = torch.empty(1, 1, TARGET_VOCAB)
        scattered = scatter_draft_logits_to_target(
            draft_logits, draft_to_target=head.draft_to_target, out=out
        )
        draft_probs = torch.softmax(draft_logits.float(), dim=-1)
        target_probs = torch.softmax(scattered.float(), dim=-1)
        for draft_id in range(DRAFT_VOCAB):
            target_id = int(head.draft_to_target[draft_id])
            self.assertAlmostEqual(
                target_probs[0, 0, target_id].item(),
                draft_probs[0, 0, draft_id].item(),
                places=5,
            )


class TestReducedVocabConfig(CustomTestCase):
    def test_normalize_native_speculators_config(self):
        normalized = _normalize_speculators_draft_config_dict(
            {
                "architectures": ["Qwen3DSparkModel"],
                "auto_map": {"": "config.DSparkSpeculatorConfig"},
                "speculators_model_type": "dspark",
                "draft_vocab_size": 32000,
                "block_size": 8,
                "aux_hidden_state_layer_ids": [2, 10, 20, 30, 37],
                "transformer_layer_config": {
                    "model_type": "qwen3",
                    "hidden_size": 2048,
                    "vocab_size": 248320,
                },
            }
        )
        self.assertEqual(normalized["model_type"], "qwen3")
        self.assertEqual(normalized["architectures"], ["Qwen3DSparkModel"])
        self.assertEqual(normalized["hidden_size"], 2048)
        self.assertEqual(normalized["draft_vocab_size"], 32000)
        self.assertEqual(normalized["target_layer_ids"], [2, 10, 20, 30, 37])
        self.assertNotIn("auto_map", normalized)
        self.assertNotIn("transformer_layer_config", normalized)

    def test_does_not_normalize_other_speculators_formats(self):
        self.assertIsNone(
            _normalize_speculators_draft_config_dict(
                {
                    "speculators_model_type": "eagle",
                    "transformer_layer_config": {"model_type": "qwen3"},
                }
            )
        )

    def test_resolve_from_top_level(self):
        cfg = SimpleNamespace(draft_vocab_size=32000, vocab_size=248320)
        self.assertEqual(resolve_draft_vocab_size(cfg), 32000)

    def test_resolve_absent_is_none(self):
        self.assertIsNone(resolve_draft_vocab_size(SimpleNamespace(vocab_size=100)))

    def test_resolve_larger_than_target_raises(self):
        cfg = SimpleNamespace(draft_vocab_size=300, vocab_size=100)
        with self.assertRaises(ValueError):
            resolve_draft_vocab_size(cfg)

    def test_parse_surfaces_draft_vocab_and_reduced_flag(self):
        cfg = SimpleNamespace(
            architectures=["Qwen3DSparkModel"],
            block_size=8,
            markov_rank=256,
            markov_head_type="vanilla",
            mask_token_id=128799,
            vocab_size=248320,
            draft_vocab_size=32000,
            sample_from_anchor=False,
        )
        parsed = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertEqual(parsed.draft_vocab_size, 32000)
        self.assertTrue(parsed.uses_reduced_draft_vocab(target_vocab_size=248320))

    def test_parse_full_vocab_is_not_reduced(self):
        cfg = SimpleNamespace(
            architectures=["Qwen3DSparkModel"],
            block_size=8,
            markov_rank=256,
            markov_head_type="vanilla",
            mask_token_id=128799,
            vocab_size=248320,
        )
        parsed = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertIsNone(parsed.draft_vocab_size)
        self.assertFalse(parsed.uses_reduced_draft_vocab(target_vocab_size=248320))


class _StubReducedModel:
    """Minimal draft-model stand-in for the graph-folded sampler: a real reduced
    Markov head plus a fake draft-vocab lm_head and a compute_base_logits that
    argmaxes to draft id 4."""

    def __init__(self, head):
        self.markov_head = head
        self.lm_head = SimpleNamespace(
            org_vocab_size=DRAFT_VOCAB,
            weight=SimpleNamespace(dtype=torch.float32),
        )

    def compute_base_logits(self, hidden):
        base = torch.zeros(hidden.shape[0], DRAFT_VOCAB)
        base[:, 4] = 9.0
        return base, None


class TestFoldedSamplerReducedVocab(CustomTestCase):
    def test_folded_buffer_stays_draft_space_and_maps_to_target(self):
        head = _reduced_vanilla_head()  # target = draft + 2
        sampler = DsparkDraftSampler(
            model=_StubReducedModel(head),
            gamma=2,
            max_bs=2,
            device="cpu",
            folded_sampling=True,
            bonus_anchor=False,
        )
        # The in-graph corrected-logits buffer must NOT balloon to the target
        # vocab: it stays draft-space (org_vocab_size of the reduced head).
        self.assertEqual(sampler.corrected_out.shape[1], DRAFT_VOCAB)

        sampler.stage_sampling_params(bs=1, sampling_info=None)  # greedy
        sampler(
            hidden_states=torch.zeros(2, HIDDEN),  # bs=1, draft_width=gamma=2
            input_ids=torch.tensor([7, 0]),
        )
        # Folded sampling also maps draft id 4 -> target id 6 before storing.
        self.assertTrue(torch.all(sampler.out[:2] == 6))


class TestIndependentLmHead(CustomTestCase):
    """The reduced draft head is a plain ParallelLMHead sized to the draft vocab
    and loaded with the standard vocab-parallel weight loader."""

    def _build_head(self, draft_vocab, hidden):
        # tp=1 CPU stand-in for the parallel context ParallelLMHead reads.
        with patch(
            "sglang.srt.layers.vocab_parallel_embedding.get_parallel",
            return_value=SimpleNamespace(tp_rank=0, tp_size=1),
        ):
            return build_independent_lm_head(
                draft_vocab_size=draft_vocab,
                hidden_size=hidden,
                quant_config=None,
                prefix="lm_head",
            )

    def test_org_vocab_is_draft_vocab(self):
        head = self._build_head(DRAFT_VOCAB, HIDDEN)
        # org_vocab_size drives gather_and_crop_vocab -> logits stay draft-space.
        self.assertEqual(head.org_vocab_size, DRAFT_VOCAB)
        self.assertEqual(head.embedding_dim, HIDDEN)
        self.assertGreaterEqual(head.weight.shape[0], DRAFT_VOCAB)

    def test_weight_loader_populates_org_rows(self):
        head = self._build_head(DRAFT_VOCAB, HIDDEN)
        loaded = torch.randn(DRAFT_VOCAB, HIDDEN)
        head.weight_loader(head.weight, loaded)
        self.assertTrue(torch.equal(head.weight[:DRAFT_VOCAB], loaded))
        # Padding rows past the org vocab are zero-filled.
        if head.weight.shape[0] > DRAFT_VOCAB:
            self.assertTrue(torch.all(head.weight[DRAFT_VOCAB:] == 0))


if __name__ == "__main__":
    unittest.main()
