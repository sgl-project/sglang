"""Unit tests for token embedding override support.

Covers:
- PositionalEmbeds dataclass (embed_types.py)
- convert_embeds_to_tensors (utils.py)
- TokenizerManager._resolve_embed_overrides (tokenizer_manager.py)
- positional_embed_overrides on GenerateReqInput/EmbeddingReqInput (io_struct.py)
- Score mixin override resolution (tokenizer_manager_score_mixin.py)
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

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
        self.assertEqual(result[0][0].dtype, torch.float32)
        self.assertEqual(result[0][0].dim(), 1)  # each vector is 1-D

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


class _FakeServerArgs:
    """Minimal stub for server_args."""

    def __init__(self, multi_item_scoring_delimiter=None):
        self.multi_item_scoring_delimiter = multi_item_scoring_delimiter


class _FakeMixin(TokenizerManagerScoreMixin):
    """Minimal stub to call mixin methods without a full TokenizerManager."""

    def __init__(self, delimiter=None):
        self.server_args = _FakeServerArgs(delimiter)
        self.multi_item_delimiter_text = None
        self.tokenizer = None
        self.is_generation = True


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

    def test_empty_embeds_list(self):
        """Empty embeds list with no placeholders succeeds."""
        embeds, positions = self.mixin._resolve_overrides_for_sequence(
            token_ids=[10, 20],
            embeds=[],
            embed_override_token_id=50,
        )
        self.assertEqual(embeds, [])
        self.assertEqual(positions, [])

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


# ========================================================================
# Score mixin: _build_token_id_inputs
# ========================================================================

DELIM_TOKEN = 99


class TestBuildTokenIdInputs(CustomTestCase):
    def setUp(self):
        self.mixin = _FakeMixin(delimiter=DELIM_TOKEN)

    # --- single-item mode, no embeds ---

    def test_single_item_no_embeds(self):
        _, input_ids, injection = self.mixin._build_token_id_inputs(
            query=[1, 2],
            items=[[3, 4], [5, 6]],
            item_first=False,
            use_multi_item_scoring=False,
            embed_override_token_id=None,
            query_embed_overrides=None,
            item_embed_overrides=None,
        )
        self.assertEqual(input_ids, [[1, 2, 3, 4], [1, 2, 5, 6]])
        self.assertIsNone(injection)

    def test_single_item_no_embeds_item_first(self):
        _, input_ids, injection = self.mixin._build_token_id_inputs(
            query=[1, 2],
            items=[[3, 4]],
            item_first=True,
            use_multi_item_scoring=False,
            embed_override_token_id=None,
            query_embed_overrides=None,
            item_embed_overrides=None,
        )
        self.assertEqual(input_ids, [[3, 4, 1, 2]])
        self.assertIsNone(injection)

    # --- multi-item mode, no embeds ---

    def test_multi_item_no_embeds(self):
        _, input_ids, injection = self.mixin._build_token_id_inputs(
            query=[1, 2],
            items=[[3, 4], [5, 6]],
            item_first=False,
            use_multi_item_scoring=True,
            embed_override_token_id=None,
            query_embed_overrides=None,
            item_embed_overrides=None,
        )
        # query<D>item1<D>item2<D>
        self.assertEqual(
            input_ids, [[1, 2, DELIM_TOKEN, 3, 4, DELIM_TOKEN, 5, 6, DELIM_TOKEN]]
        )
        self.assertIsNone(injection)

    # --- single-item mode, with embeds ---

    def test_single_item_query_embeds(self):
        """Query placeholder overrides are resolved per item."""
        _, input_ids, injection = self.mixin._build_token_id_inputs(
            query=[50, 10],
            items=[[20, 30], [40, 50]],
            item_first=False,
            use_multi_item_scoring=False,
            embed_override_token_id=50,
            query_embed_overrides=[_vec(1)],
            item_embed_overrides=None,
        )
        self.assertEqual(input_ids, [[50, 10, 20, 30], [50, 10, 40, 50]])
        self.assertIsNotNone(injection)
        self.assertEqual(len(injection), 2)
        # Each item gets its own PositionalEmbeds with query override at pos 0
        self.assertEqual(injection[0].positions, [0])
        self.assertEqual(injection[1].positions, [0])

    def test_single_item_item_embeds(self):
        """Per-item overrides with correct position offsets."""
        _, input_ids, injection = self.mixin._build_token_id_inputs(
            query=[10, 20],
            items=[[50, 30]],
            item_first=False,
            use_multi_item_scoring=False,
            embed_override_token_id=50,
            query_embed_overrides=None,
            item_embed_overrides=[[_vec(2)]],
        )
        self.assertEqual(input_ids, [[10, 20, 50, 30]])
        self.assertIsNotNone(injection)
        # item placeholder at index 0 of item, offset by query length 2
        self.assertEqual(injection[0].positions, [2])

    def test_single_item_no_override_positions_returns_none_injection(self):
        """When no items have placeholders, injection should be None."""
        _, input_ids, injection = self.mixin._build_token_id_inputs(
            query=[10, 20],
            items=[[30, 40]],
            item_first=False,
            use_multi_item_scoring=False,
            embed_override_token_id=50,
            query_embed_overrides=None,
            item_embed_overrides=[None],
        )
        self.assertIsNone(injection)

    def test_single_item_query_and_item_embeds(self):
        """Single-item mode with both query and item overrides in one request."""
        _, input_ids, injection = self.mixin._build_token_id_inputs(
            query=[50, 10],
            items=[[20, 50]],
            item_first=False,
            use_multi_item_scoring=False,
            embed_override_token_id=50,
            query_embed_overrides=[_vec(1)],
            item_embed_overrides=[[_vec(2)]],
        )
        self.assertEqual(input_ids, [[50, 10, 20, 50]])
        self.assertIsNotNone(injection)
        pe = injection[0]
        # query override at pos 0, item override at pos 3 (query_len=2 + idx=1)
        self.assertEqual(pe.positions, [0, 3])
        self.assertEqual(pe.embeds.shape, (2, HIDDEN_DIM))

    def test_single_item_empty_query(self):
        """Empty query with item-only overrides (valid from score_prompts)."""
        _, input_ids, injection = self.mixin._build_token_id_inputs(
            query=[],
            items=[[50, 10]],
            item_first=False,
            use_multi_item_scoring=False,
            embed_override_token_id=50,
            query_embed_overrides=None,
            item_embed_overrides=[[_vec(1)]],
        )
        self.assertEqual(input_ids, [[50, 10]])
        self.assertIsNotNone(injection)
        # item placeholder at absolute pos 0 (offset=len([])=0)
        self.assertEqual(injection[0].positions, [0])

    # --- multi-item mode, with embeds ---

    def test_multi_item_with_query_and_item_embeds(self):
        """Multi-item mode resolves query overrides once and item overrides per item."""
        _, input_ids, injection = self.mixin._build_token_id_inputs(
            query=[50, 10],
            items=[[20, 50], [30, 40]],
            item_first=False,
            use_multi_item_scoring=True,
            embed_override_token_id=50,
            query_embed_overrides=[_vec(1)],
            item_embed_overrides=[[_vec(2)], None],
        )
        # query<D>item1<D>item2<D> = [50,10, 99, 20,50, 99, 30,40, 99]
        self.assertEqual(len(input_ids), 1)
        self.assertIsNotNone(injection)
        self.assertEqual(
            len(injection), 1
        )  # single PositionalEmbeds for combined sequence
        pe = injection[0]
        # query override at pos 0, item[0] override at pos 4 (query_len=2 + delim=1 + idx=1)
        self.assertIn(0, pe.positions)
        self.assertIn(4, pe.positions)
        self.assertEqual(pe.embeds.shape[0], 2)


# ========================================================================
# Score mixin: score_request validation
# ========================================================================


class TestScoreRequestValidation(CustomTestCase):
    """Test validation guards in score_request without running full pipeline."""

    def setUp(self):
        self.mixin = _FakeMixin()

    def _call(self, **kwargs):
        """Wrapper to call score_request synchronously."""
        import asyncio

        return asyncio.run(self.mixin.score_request(**kwargs))

    def test_generation_requires_label_token_ids(self):
        self.mixin.is_generation = True
        with self.assertRaisesRegex(ValueError, "label_token_ids is required"):
            self._call(
                query=[1, 2],
                items=[[3, 4]],
                label_token_ids=None,
            )

    def test_seq_classification_allows_none_label_token_ids(self):
        """SequenceClassification models should not require label_token_ids.
        Verify it passes validation and reaches generate_request."""
        self.mixin.is_generation = False
        mock_result = AsyncMock()
        mock_result.__anext__ = AsyncMock(
            return_value=[{"embedding": [0.1, 0.9], "meta_info": {"prompt_tokens": 2}}]
        )
        self.mixin.generate_request = MagicMock(return_value=mock_result)
        result = self._call(
            query=[1, 2],
            items=[[3, 4]],
            label_token_ids=None,
        )
        self.mixin.generate_request.assert_called_once()
        self.assertEqual(len(result.scores), 1)

    def test_items_none_raises(self):
        with self.assertRaisesRegex(ValueError, "items must be provided"):
            self._call(
                query=[1, 2],
                items=None,
                label_token_ids=[100],
            )

    def test_empty_items_returns_empty(self):
        result = self._call(
            query=[1, 2],
            items=[],
            label_token_ids=[100],
        )
        self.assertEqual(result.scores, [])
        self.assertEqual(result.prompt_tokens, 0)

    def test_embed_override_token_id_required_with_query_embeds(self):
        with self.assertRaisesRegex(ValueError, "embed_override_token_id is required"):
            self._call(
                query=[1, 2],
                items=[[3, 4]],
                label_token_ids=[100],
                query_embed_overrides=[_vec(1)],
                embed_override_token_id=None,
            )

    def test_embed_override_token_id_required_with_item_embeds(self):
        with self.assertRaisesRegex(ValueError, "embed_override_token_id is required"):
            self._call(
                query=[1, 2],
                items=[[3, 4]],
                label_token_ids=[100],
                item_embed_overrides=[[_vec(1)]],
                embed_override_token_id=None,
            )

    def test_item_first_with_embeds_raises(self):
        with self.assertRaisesRegex(ValueError, "item_first is not supported"):
            self._call(
                query=[1, 2],
                items=[[3, 4]],
                label_token_ids=[100],
                item_first=True,
                embed_override_token_id=50,
                query_embed_overrides=[_vec(1)],
            )

    def test_item_embed_overrides_length_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "must match items length"):
            self._call(
                query=[1, 2],
                items=[[3, 4], [5, 6]],
                label_token_ids=[100],
                embed_override_token_id=50,
                item_embed_overrides=[[_vec(1)]],  # 1 override for 2 items
            )


if __name__ == "__main__":
    unittest.main()
