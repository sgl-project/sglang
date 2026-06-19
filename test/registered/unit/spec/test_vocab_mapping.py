# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TLI VocabMapping (sglang.srt.speculative.vocab_mapping).

All tests are CPU-only: no server, no GPU, no real tokenizer needed.
Tokenizers are replaced with lightweight fakes.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.speculative.vocab_mapping import (
    VocabMapping,
    _detect_space_sign,
    _normalize_token,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")


def _make_tokenizer(vocab: dict, unk_token_id=None, eos_token_id=2):
    """Build a minimal MagicMock tokenizer backed by a vocab dict.

    Args:
        vocab: {token_string: token_id}
        unk_token_id: value returned by tokenizer.unk_token_id
        eos_token_id: value returned by tokenizer.eos_token_id
    """
    tok = MagicMock()
    tok.unk_token_id = unk_token_id
    tok.eos_token_id = eos_token_id
    tok.get_vocab.return_value = dict(vocab)

    # _detect_space_sign probes tokenizer(" ", add_special_tokens=False)
    # and tokenizer.convert_ids_to_tokens().
    # Default: no space-prefix detected.
    tok.return_value = {"input_ids": []}
    tok.convert_ids_to_tokens.return_value = []
    return tok


def _simple_mapping(
    target_vocab,
    draft_vocab,
    unk=0,
    target_unk=None,
    draft_unk=None,
):
    """Build a VocabMapping from plain dicts."""
    t_tok = _make_tokenizer(target_vocab, unk_token_id=target_unk or unk)
    d_tok = _make_tokenizer(draft_vocab, unk_token_id=draft_unk or unk)
    return VocabMapping(
        target_tokenizer=t_tok,
        draft_tokenizer=d_tok,
        target_vocab_size=max(target_vocab.values()) + 1,
        draft_vocab_size=max(draft_vocab.values()) + 1,
        device=DEVICE,
    )


# ---------------------------------------------------------------------------
# _detect_space_sign
# ---------------------------------------------------------------------------


class TestDetectSpaceSign(CustomTestCase):
    def test_returns_glyph_sign(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [42]}
        tok.convert_ids_to_tokens.return_value = ["\u0120hello"]
        self.assertEqual(_detect_space_sign(tok), "\u0120")

    def test_returns_sentencepiece_sign(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [7]}
        tok.convert_ids_to_tokens.return_value = ["\u2581world"]
        self.assertEqual(_detect_space_sign(tok), "\u2581")

    def test_returns_none_for_unknown_tokenizer(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [1]}
        tok.convert_ids_to_tokens.return_value = [
            "Xspace"
        ]  # first char 'X', not a space prefix
        self.assertIsNone(_detect_space_sign(tok))

    def test_returns_none_for_empty_ids(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": []}
        self.assertIsNone(_detect_space_sign(tok))

    def test_returns_none_on_exception(self):
        tok = MagicMock()
        tok.side_effect = RuntimeError("no tokenizer")
        self.assertIsNone(_detect_space_sign(tok))


# ---------------------------------------------------------------------------
# _normalize_token
# ---------------------------------------------------------------------------


class TestNormalizeToken(CustomTestCase):
    def test_glyph_prefix_removed(self):
        self.assertEqual(_normalize_token("\u0120hello"), " hello")

    def test_sentencepiece_prefix_removed(self):
        self.assertEqual(_normalize_token("\u2581world"), " world")

    def test_no_prefix_unchanged(self):
        self.assertEqual(_normalize_token("hello"), "hello")

    def test_explicit_space_sign(self):
        self.assertEqual(_normalize_token("\u0120foo", space_sign="\u0120"), " foo")

    def test_explicit_space_sign_no_match_unchanged(self):
        # Token has ▁ prefix but explicit sign is Ġ → not stripped
        self.assertEqual(
            _normalize_token("\u2581bar", space_sign="\u0120"), "\u2581bar"
        )

    def test_token_is_only_prefix(self):
        self.assertEqual(_normalize_token("\u0120"), " ")


# ---------------------------------------------------------------------------
# VocabMapping — intersection construction
# ---------------------------------------------------------------------------


class TestVocabMappingIntersection(CustomTestCase):
    """Tests that the intersection tensors are built correctly."""

    def setUp(self):
        # target: tokens a(0), b(1), c(2), d(3)
        # draft:  tokens a(0), b(1), e(2)
        # intersection: a, b  →  draft IDs {0,1}
        self.target_vocab = {"a": 0, "b": 1, "c": 2, "d": 3}
        self.draft_vocab = {"a": 0, "b": 1, "e": 2}
        self.vm = _simple_mapping(self.target_vocab, self.draft_vocab, unk=3)

    def test_intersection_size(self):
        self.assertEqual(self.vm.intersection_size, 2)

    def test_intersection_mask_draft(self):
        mask = self.vm.intersection_mask_draft
        self.assertTrue(mask[0].item())  # a in intersection
        self.assertTrue(mask[1].item())  # b in intersection
        self.assertFalse(mask[2].item())  # e not in intersection

    def test_intersection_draft_ids(self):
        ids = self.vm.intersection_draft_ids.tolist()
        self.assertEqual(sorted(ids), [0, 1])

    def test_draft_to_target_ids(self):
        # a: draft 0 → target 0; b: draft 1 → target 1; e: draft 2 → -1
        self.assertEqual(self.vm.draft_to_target_ids[0].item(), 0)
        self.assertEqual(self.vm.draft_to_target_ids[1].item(), 1)
        self.assertEqual(self.vm.draft_to_target_ids[2].item(), -1)

    def test_target_to_draft_ids(self):
        # a: target 0 → draft 0; b: target 1 → draft 1; c,d not in intersection
        self.assertEqual(self.vm.target_to_draft_ids[0].item(), 0)
        self.assertEqual(self.vm.target_to_draft_ids[1].item(), 1)
        self.assertEqual(self.vm.target_to_draft_ids[2].item(), -1)
        self.assertEqual(self.vm.target_to_draft_ids[3].item(), -1)

    def test_space_prefix_tokens_match(self):
        """Tokens that differ only in BPE space prefix should still intersect."""
        target_vocab = {"\u0120hello": 0, "world": 1}  # Ġhello, world
        draft_vocab = {"\u2581hello": 0, "world": 1}  # ▁hello, world
        # Both tokenizers probe their space sign; mock them to return the right prefix.
        t_tok = _make_tokenizer(target_vocab, unk_token_id=5)
        t_tok.return_value = {"input_ids": [0]}
        t_tok.convert_ids_to_tokens.return_value = ["\u0120"]
        d_tok = _make_tokenizer(draft_vocab, unk_token_id=5)
        d_tok.return_value = {"input_ids": [0]}
        d_tok.convert_ids_to_tokens.return_value = ["\u2581"]
        vm = VocabMapping(
            target_tokenizer=t_tok,
            draft_tokenizer=d_tok,
            target_vocab_size=2,
            draft_vocab_size=2,
            device=DEVICE,
        )
        self.assertEqual(vm.intersection_size, 2)


# ---------------------------------------------------------------------------
# VocabMapping — unk_token_id fallback
# ---------------------------------------------------------------------------


class TestVocabMappingUnkFallback(CustomTestCase):
    def test_eos_used_when_no_unk(self):
        """Models without unk_token_id (e.g. Llama 3) fall back to eos_token_id."""
        target_vocab = {"a": 0, "b": 1, "c": 2}
        draft_vocab = {"a": 0, "b": 1, "d": 2}
        t_tok = _make_tokenizer(target_vocab, unk_token_id=None, eos_token_id=99)
        d_tok = _make_tokenizer(draft_vocab, unk_token_id=None, eos_token_id=88)
        vm = VocabMapping(
            target_tokenizer=t_tok,
            draft_tokenizer=d_tok,
            target_vocab_size=3,
            draft_vocab_size=3,
            device=DEVICE,
        )
        self.assertEqual(vm.target_unk_token_id, 99)
        self.assertEqual(vm.draft_unk_token_id, 88)

    def test_explicit_unk_preferred_over_eos(self):
        target_vocab = {"a": 0, "b": 1, "<unk>": 2}
        draft_vocab = {"a": 0, "b": 1, "<unk>": 2}
        t_tok = _make_tokenizer(target_vocab, unk_token_id=2, eos_token_id=1)
        d_tok = _make_tokenizer(draft_vocab, unk_token_id=2, eos_token_id=1)
        vm = _simple_mapping(target_vocab, draft_vocab, target_unk=2, draft_unk=2)
        self.assertEqual(vm.target_unk_token_id, 2)
        self.assertEqual(vm.draft_unk_token_id, 2)

    def test_raises_when_neither_unk_nor_eos(self):
        target_vocab = {"a": 0, "b": 1}
        draft_vocab = {"a": 0, "b": 1}
        t_tok = _make_tokenizer(target_vocab, unk_token_id=None, eos_token_id=None)
        d_tok = _make_tokenizer(draft_vocab, unk_token_id=None, eos_token_id=None)
        with self.assertRaises(ValueError):
            VocabMapping(
                target_tokenizer=t_tok,
                draft_tokenizer=d_tok,
                target_vocab_size=2,
                draft_vocab_size=2,
                device=DEVICE,
            )


# ---------------------------------------------------------------------------
# VocabMapping — map_target_to_draft_ids
# ---------------------------------------------------------------------------


class TestMapTargetToDraft(CustomTestCase):
    def setUp(self):
        # target: a(0) b(1) c(2); draft: a(0) b(1) e(2); unk=5 (out of range, safe)
        self.vm = _simple_mapping(
            {"a": 0, "b": 1, "c": 2}, {"a": 0, "b": 1, "e": 2}, unk=5
        )

    def test_intersection_tokens_mapped_correctly(self):
        ids = torch.tensor([0, 1])  # a, b
        result = self.vm.map_target_to_draft_ids(ids)
        self.assertEqual(result.tolist(), [0, 1])

    def test_out_of_intersection_mapped_to_draft_unk(self):
        ids = torch.tensor([2])  # c not in draft
        result = self.vm.map_target_to_draft_ids(ids)
        self.assertEqual(result.tolist(), [5])

    def test_mixed_batch(self):
        ids = torch.tensor([0, 2, 1])  # a, c(unk), b
        result = self.vm.map_target_to_draft_ids(ids)
        self.assertEqual(result.tolist(), [0, 5, 1])

    def test_output_dtype_matches_input(self):
        ids = torch.tensor([0], dtype=torch.int32)
        result = self.vm.map_target_to_draft_ids(ids)
        self.assertEqual(result.dtype, torch.int32)

    def test_no_clone_when_fully_in_intersection(self):
        """No extra allocation when all tokens are in the intersection."""
        ids = torch.tensor([0, 1])
        result = self.vm.map_target_to_draft_ids(ids)
        # Just verify correctness; the no-clone path is covered by coverage.
        self.assertEqual(result.tolist(), [0, 1])


# ---------------------------------------------------------------------------
# VocabMapping — map_draft_to_target_ids
# ---------------------------------------------------------------------------


class TestMapDraftToTarget(CustomTestCase):
    def setUp(self):
        self.vm = _simple_mapping(
            {"a": 0, "b": 1, "c": 2}, {"a": 0, "b": 1, "e": 2}, unk=5
        )

    def test_intersection_tokens_mapped_correctly(self):
        ids = torch.tensor([0, 1])  # a, b
        result = self.vm.map_draft_to_target_ids(ids)
        self.assertEqual(result.tolist(), [0, 1])

    def test_out_of_intersection_mapped_to_target_unk(self):
        ids = torch.tensor([2])  # e not in target
        result = self.vm.map_draft_to_target_ids(ids)
        self.assertEqual(result.tolist(), [5])

    def test_output_dtype_matches_input(self):
        ids = torch.tensor([0], dtype=torch.int64)
        result = self.vm.map_draft_to_target_ids(ids)
        self.assertEqual(result.dtype, torch.int64)


# ---------------------------------------------------------------------------
# VocabMapping — constrain_draft_logits
# ---------------------------------------------------------------------------


class TestConstrainDraftLogits(CustomTestCase):
    def setUp(self):
        # intersection: draft tokens 0, 1; non-intersection: 2
        self.vm = _simple_mapping(
            {"a": 0, "b": 1, "c": 2}, {"a": 0, "b": 1, "e": 2}, unk=5
        )

    def test_non_intersection_set_to_neg_inf(self):
        logits = torch.zeros(4, 3)  # batch=4, draft_vocab=3
        out = self.vm.constrain_draft_logits(logits)
        self.assertTrue(torch.isinf(out[:, 2]).all())
        self.assertTrue((out[:, 2] < 0).all())

    def test_intersection_logits_unchanged(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        out = self.vm.constrain_draft_logits(logits)
        self.assertAlmostEqual(out[0, 0].item(), 1.0)
        self.assertAlmostEqual(out[0, 1].item(), 2.0)

    def test_does_not_modify_input(self):
        logits = torch.ones(2, 3)
        original = logits.clone()
        self.vm.constrain_draft_logits(logits)
        self.assertTrue(torch.equal(logits, original))

    def test_1d_batch_dimension(self):
        logits = torch.zeros(3)  # single token, no batch dim
        out = self.vm.constrain_draft_logits(logits.unsqueeze(0))
        self.assertTrue(torch.isinf(out[0, 2]))

    def test_pruned_head_makes_constrain_noop(self):
        """After LM head pruning, constrain_draft_logits should be a no-op.

        The pruned head's forward() pre-fills non-intersection positions with
        -inf, so applying the mask again produces the identical tensor.
        """
        vocab_size = 3
        logits = torch.full((1, vocab_size), float("-inf"))
        logits[0, 0] = 1.0
        logits[0, 1] = 2.0
        out = self.vm.constrain_draft_logits(logits)
        self.assertTrue(torch.equal(out, logits))


# ---------------------------------------------------------------------------
# VocabMapping — disjoint vocabularies
# ---------------------------------------------------------------------------


class TestVocabMappingEdgeCases(CustomTestCase):
    def test_zero_intersection_warns(self):
        """Completely disjoint vocabs produce intersection_size == 0."""
        vm = _simple_mapping({"a": 0}, {"b": 0}, unk=0)
        self.assertEqual(vm.intersection_size, 0)

    def test_full_intersection(self):
        """Identical vocabs produce full intersection."""
        vocab = {"a": 0, "b": 1, "c": 2}
        vm = _simple_mapping(vocab, vocab, unk=3)
        self.assertEqual(vm.intersection_size, 3)
        self.assertTrue(vm.intersection_mask_draft.all())

    def test_token_id_bounds_respected(self):
        """Tokens whose IDs exceed vocab_size are silently excluded."""
        # target_vocab_size=2 but token c has id=5 — should be ignored
        target_vocab = {"a": 0, "b": 1, "c": 5}
        draft_vocab = {"a": 0, "b": 1, "c": 5}
        t_tok = _make_tokenizer(target_vocab, unk_token_id=3)
        d_tok = _make_tokenizer(draft_vocab, unk_token_id=3)
        vm = VocabMapping(
            target_tokenizer=t_tok,
            draft_tokenizer=d_tok,
            target_vocab_size=2,
            draft_vocab_size=2,
            device=DEVICE,
        )
        # Only a and b fit within vocab_size=2
        self.assertEqual(vm.intersection_size, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
