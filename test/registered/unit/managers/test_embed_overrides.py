"""Unit tests for token embedding override support.

Covers:
- PositionalEmbeds dataclass (embed_types.py)
- convert_embeds_to_tensors (utils.py)
- TokenizerManager._resolve_embed_overrides (tokenizer_manager.py)
- positional_embed_overrides on GenerateReqInput/EmbeddingReqInput (io_struct.py)
- Score mixin override resolution (tokenizer_manager_score_mixin.py)
"""

import unittest

import torch

from sglang.srt.entrypoints.openai.utils import convert_embeds_to_tensors
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.tokenizer_manager_score_mixin import (
    TokenizerManagerScoreMixin,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")

HIDDEN_DIM = 4


def _vec(val: float = 1.0) -> torch.Tensor:
    """Create a 1-D tensor of size HIDDEN_DIM."""
    return torch.full((HIDDEN_DIM,), val, dtype=torch.float32)


def _vec2d(val: float = 1.0) -> torch.Tensor:
    """Create a [1, HIDDEN_DIM] tensor."""
    return torch.full((1, HIDDEN_DIM), val, dtype=torch.float32)


# ========================================================================
# PositionalEmbeds
# ========================================================================


class TestPositionalEmbeds(CustomTestCase):
    def test_from_list_of_1d_tensors(self):
        pe = PositionalEmbeds(embeds=[_vec(1), _vec(2)], positions=[0, 5])
        self.assertEqual(pe.embeds.shape, (2, HIDDEN_DIM))
        self.assertAlmostEqual(pe.embeds[0, 0].item(), 1.0)
        self.assertAlmostEqual(pe.embeds[1, 0].item(), 2.0)

    def test_from_list_of_2d_tensors(self):
        pe = PositionalEmbeds(embeds=[_vec2d(3), _vec2d(4)], positions=[1, 2])
        self.assertEqual(pe.embeds.shape, (2, HIDDEN_DIM))

    def test_from_pre_stacked_tensor(self):
        stacked = torch.zeros(3, HIDDEN_DIM)
        pe = PositionalEmbeds(embeds=stacked, positions=[0, 1, 2])
        self.assertIs(pe.embeds, stacked)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            PositionalEmbeds(embeds=[_vec()], positions=[0, 1])

    def test_empty(self):
        pe = PositionalEmbeds(embeds=torch.zeros(0, HIDDEN_DIM), positions=[])
        self.assertEqual(pe.embeds.shape[0], 0)


# ========================================================================
# convert_embeds_to_tensors
# ========================================================================


class TestConvertEmbedsToTensors(CustomTestCase):
    def test_none_returns_none(self):
        self.assertIsNone(convert_embeds_to_tensors(None))

    def test_empty_list(self):
        self.assertEqual(convert_embeds_to_tensors([]), [])

    def test_single_input(self):
        """[num_replacements][hidden_size] -> [[tensor, ...]]"""
        result = convert_embeds_to_tensors([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(len(result), 1)  # wrapped in outer list
        self.assertEqual(len(result[0]), 2)  # two replacement vectors
        self.assertTrue(torch.is_tensor(result[0][0]))
        self.assertEqual(result[0][0].tolist(), [1.0, 2.0])

    def test_batch_input(self):
        """[num_inputs][num_replacements][hidden_size] -> [[tensor, ...], ...]"""
        result = convert_embeds_to_tensors(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0], [5.0, 6.0]],
            ]
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 2)


# ========================================================================
# TokenizerManager._resolve_embed_overrides
# ========================================================================


class TestResolveEmbedOverrides(CustomTestCase):
    def test_basic_resolution(self):
        embeds = [_vec(1), _vec(2)]
        pe = TokenizerManager._resolve_embed_overrides(
            input_ids=[10, 50, 20, 50, 30],
            token_id=50,
            embeds=embeds,
        )
        self.assertIsInstance(pe, PositionalEmbeds)
        self.assertEqual(pe.positions, [1, 3])
        self.assertEqual(pe.embeds.shape, (2, HIDDEN_DIM))

    def test_no_placeholders_raises(self):
        with self.assertRaises(ValueError):
            TokenizerManager._resolve_embed_overrides(
                input_ids=[10, 20, 30],
                token_id=50,
                embeds=[_vec()],
            )

    def test_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            TokenizerManager._resolve_embed_overrides(
                input_ids=[10, 50, 20],
                token_id=50,
                embeds=[_vec(1), _vec(2)],
            )


# ========================================================================
# io_struct: positional_embed_overrides on GenerateReqInput
# ========================================================================


class TestGenerateReqInputEmbedOverride(CustomTestCase):
    def test_single_override_in_getitem(self):
        """Single PositionalEmbeds is shared across all items in __getitem__."""
        pe = PositionalEmbeds(embeds=[_vec()], positions=[0])
        req = GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            sampling_params=[{}, {}],
            positional_embed_overrides=pe,
        )
        req.normalize_batch_and_arguments()
        item = req[0]
        self.assertIs(item.positional_embed_overrides, pe)

    def test_batch_override_in_getitem(self):
        """List[Optional[PositionalEmbeds]] is indexed per-item."""
        pe0 = PositionalEmbeds(embeds=[_vec(1)], positions=[0])
        pe1 = None
        req = GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            sampling_params=[{}, {}],
            positional_embed_overrides=[pe0, pe1],
        )
        req.normalize_batch_and_arguments()
        self.assertEqual(req[0].positional_embed_overrides, pe0)
        self.assertIsNone(req[1].positional_embed_overrides)


# ========================================================================
# io_struct: embed override fields on EmbeddingReqInput
# ========================================================================


class TestEmbeddingReqInputEmbedOverride(CustomTestCase):
    def test_override_fields_in_getitem(self):
        """embed_override_token_id, embed_overrides, and positional_embed_overrides
        are correctly sliced in __getitem__."""
        pe0 = PositionalEmbeds(embeds=[_vec(1)], positions=[0])
        pe1 = PositionalEmbeds(embeds=[_vec(2)], positions=[1])
        req = EmbeddingReqInput(
            input_ids=[[50, 10], [20, 50]],
            sampling_params=[{}, {}],
            embed_override_token_id=50,
            embed_overrides=[[_vec(1)], [_vec(2)]],
            positional_embed_overrides=[pe0, pe1],
        )
        req.normalize_batch_and_arguments()
        item0 = req[0]
        item1 = req[1]
        self.assertEqual(item0.embed_override_token_id, 50)
        self.assertEqual(len(item0.embed_overrides), 1)
        self.assertEqual(item0.positional_embed_overrides, pe0)
        self.assertEqual(item1.positional_embed_overrides, pe1)


# ========================================================================
# Score mixin: _resolve_overrides_for_sequence
# ========================================================================


class _FakeMixin(TokenizerManagerScoreMixin):
    """Minimal stub to call mixin methods without a full TokenizerManager."""

    pass


class TestResolveOverridesForSequence(CustomTestCase):
    def setUp(self):
        self.mixin = _FakeMixin()

    def test_none_embeds_returns_empty(self):
        embeds, positions = self.mixin._resolve_overrides_for_sequence(
            token_ids=[10, 50, 20],
            embeds=None,
            embed_override_token_id=50,
        )
        self.assertEqual(embeds, [])
        self.assertEqual(positions, [])

    def test_basic_resolution(self):
        e1, e2 = _vec(1), _vec(2)
        embeds, positions = self.mixin._resolve_overrides_for_sequence(
            token_ids=[50, 10, 50],
            embeds=[e1, e2],
            embed_override_token_id=50,
        )
        self.assertEqual(len(embeds), 2)
        self.assertEqual(positions, [0, 2])

    def test_with_offset(self):
        embeds, positions = self.mixin._resolve_overrides_for_sequence(
            token_ids=[10, 50],
            embeds=[_vec()],
            embed_override_token_id=50,
            position_offset=100,
        )
        self.assertEqual(positions, [101])

    def test_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.mixin._resolve_overrides_for_sequence(
                token_ids=[50, 50],
                embeds=[_vec()],
                embed_override_token_id=50,
            )


# ========================================================================
# Score mixin: _resolve_embed_overrides_for_request
# ========================================================================


class TestResolveEmbedOverridesForRequest(CustomTestCase):
    def setUp(self):
        self.mixin = _FakeMixin()

    def test_no_overrides_returns_none(self):
        result = self.mixin._resolve_embed_overrides_for_request(
            query=[10, 20],
            item=[30, 40],
            embed_override_token_id=50,
            query_embed_overrides=None,
            item_embeds=None,
            item_position_offset=2,
            item_label="items[0]",
        )
        self.assertIsNone(result)

    def test_query_only_overrides(self):
        pe = self.mixin._resolve_embed_overrides_for_request(
            query=[50, 20],
            item=[30, 40],
            embed_override_token_id=50,
            query_embed_overrides=[_vec(1)],
            item_embeds=None,
            item_position_offset=2,
            item_label="items[0]",
        )
        self.assertIsInstance(pe, PositionalEmbeds)
        self.assertEqual(pe.positions, [0])
        self.assertEqual(pe.embeds.shape, (1, HIDDEN_DIM))

    def test_item_only_overrides(self):
        pe = self.mixin._resolve_embed_overrides_for_request(
            query=[10, 20],
            item=[50, 40],
            embed_override_token_id=50,
            query_embed_overrides=None,
            item_embeds=[_vec(2)],
            item_position_offset=2,
            item_label="items[0]",
        )
        self.assertEqual(pe.positions, [2])  # offset applied

    def test_query_and_item_overrides(self):
        pe = self.mixin._resolve_embed_overrides_for_request(
            query=[50, 20],
            item=[30, 50],
            embed_override_token_id=50,
            query_embed_overrides=[_vec(1)],
            item_embeds=[_vec(2)],
            item_position_offset=2,
            item_label="items[0]",
        )
        self.assertEqual(pe.positions, [0, 3])  # query pos 0, item pos 1+offset 2
        self.assertEqual(pe.embeds.shape, (2, HIDDEN_DIM))


if __name__ == "__main__":
    unittest.main()
