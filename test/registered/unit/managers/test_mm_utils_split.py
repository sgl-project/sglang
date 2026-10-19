"""Unit tests for multimodal item splitting and token padding.

This is the load-bearing behavioral path for multi-image requests: a bundled
``MultimodalDataItem`` (one item carrying N image offsets + a concatenated
feature) must be split back into N per-image items so RadixAttention can cache
each image independently and chunked-prefill can encode them one at a time.

The MoonViT-style models (e.g. nvidia/LocateAnything-3B) carry their per-image
grids under ``image_grid_hws`` rather than ``image_grid_thw``; the splitter must
recognize both keys, fall back cleanly when no usable grid is present, and not
mis-split a degenerate flat grid. No server / GPU / weight loading involved.
"""

import unittest
import warnings
from array import array
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPattern,
    MultiModalityDataPaddingPatternMultimodalTokens,
    MultiModalityDataPaddingPatternTokenPairs,
    duplicate_pad_mm_input_ids,
    embed_mm_inputs,
    get_new_expanded_mm_items,
    pad_mm_token_pairs,
    pad_mm_tokens,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.models.ernie45_vl import Ernie4_5_VLMoeForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _bundled_item(grid_key=None, grid=None, feature_len=10, num_images=2):
    """A bundled IMAGE item: `num_images` offsets, one concatenated feature."""
    model_specific_data = {}
    if grid_key is not None:
        model_specific_data[grid_key] = grid
    # Distinct per-row values so slice boundaries are checkable.
    feature = torch.arange(feature_len * 3, dtype=torch.float32).reshape(feature_len, 3)
    offsets = [(0, 5), (5, feature_len)][:num_images]
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets,
        feature=feature,
        model_specific_data=model_specific_data,
    )


class TestGetNewExpandedMMItems(CustomTestCase):
    def test_image_grid_hws_splits_per_image(self):
        # grid rows [[2,3],[4,1]] -> prod = [6, 4] patches -> feature_len 10.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=[[2, 3], [4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertEqual([len(o.offsets) for o in out], [1, 1])
        self.assertEqual(out[0].offsets, [(0, 5)])
        self.assertEqual(out[1].offsets, [(5, 10)])
        # Feature sliced 0:6 and 6:10 along dim-0.
        self.assertEqual(out[0].feature.shape[0], 6)
        self.assertEqual(out[1].feature.shape[0], 4)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))
        # Split items must re-hash (pad value is recomputed per image).
        self.assertTrue(all(o.hash is None for o in out))

    def test_image_grid_hws_tensor_splits_per_image(self):
        # Same as above but the grid arrives as a rank-2 tensor (HF emits these).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([[2, 3], [4, 1]], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_image_grid_thw_still_splits(self):
        # The pre-existing image_grid_thw path must keep working:
        # [[1,2,3],[1,4,1]] -> [6,4].
        item = _bundled_item(
            grid_key="image_grid_thw",
            grid=[[1, 2, 3], [1, 4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_missing_grid_falls_back_to_simple_split(self):
        # No grid, but feature dim-0 == num offsets -> simple per-row split.
        item = _bundled_item(grid_key=None, feature_len=2, num_images=2)
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:1]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[1:2]))

    def test_flat_1d_grid_does_not_mis_split(self):
        # A flat 1-D grid (`tensor([2, 2])`) has length == num_items so it passes
        # the length check, but prod(dim=-1) would collapse it to a scalar and
        # corrupt the slice boundaries. The rank-2 guard must reject it. With
        # feature_len != num_items, the simple-split fallback also declines, so
        # the bundled item is passed through unchanged (never mis-sliced).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([2, 2], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)

    def test_numpy_grid_splits_per_image(self):
        # image_grid_hws can arrive as a numpy array from the HF image processor.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=np.array([[2, 3], [4, 1]], dtype=np.int64),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_non_bundled_item_passes_through(self):
        # A single-image item (one offset) is not bundled and is returned as-is.
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 5)],
            feature=torch.arange(18, dtype=torch.float32).reshape(6, 3),
            model_specific_data={"image_grid_hws": [[2, 3]]},
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)


class TestMMPadding(CustomTestCase):
    def test_ernie_visual_mask_recognizes_every_sentinel(self):
        item = MultimodalDataItem(modality=Modality.IMAGE, hash=7, offsets=[(1, 6)])
        item.set_pad_value()
        mm_inputs = MultimodalInputs(mm_items=[item], im_token_id=9)
        padded = pad_mm_tokens(array("q", [1] + [9] * 6 + [2]), mm_inputs)
        model = SimpleNamespace(_visual_token_ids_tensor_cache=torch.tensor([9]))
        batch = SimpleNamespace(mm_inputs=[None, mm_inputs])
        Ernie4_5_VLMoeForConditionalGeneration._set_visual_token_mask(
            model, torch.tensor(padded), batch
        )
        self.assertEqual(
            model.visual_token_mask.flatten().tolist(), [False] + [True] * 6 + [False]
        )

    def test_generic_padding_reaches_embedding_scatter(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for modality in Modality.all():
                with self.subTest(device=device, modality=modality):
                    item = MultimodalDataItem(
                        modality=modality,
                        hash=7,
                        offsets=[(1, 6)],
                        precomputed_embeddings=torch.arange(18, device=device)
                        .reshape(6, 3)
                        .float(),
                    )
                    item.set_pad_value()
                    mm_inputs = MultimodalInputs(
                        mm_items=[item],
                        im_token_id=9,
                        audio_token_id=9,
                        video_token_id=9,
                    )
                    padded = pad_mm_tokens(array("q", [1] + [9] * 6 + [2]), mm_inputs)
                    embedding = torch.nn.Embedding(10, 3, device=device)
                    # chunked prefill may begin at any of the four sentinels
                    for prefix in (0, 1, 2, 3, 4):
                        ids = torch.tensor(padded[prefix:7], device=device)
                        actual, _ = embed_mm_inputs(
                            [mm_inputs],
                            [prefix],
                            [len(ids)],
                            ids,
                            embedding,
                            data_embedding_func_mapping={modality: lambda _: None},
                        )
                        expected = item.precomputed_embeddings[max(0, prefix - 1) :]
                        self.assertTrue(
                            torch.equal(actual[1:] if prefix == 0 else actual, expected)
                        )

    def test_padding_requires_int64_arrays(self):
        mm_inputs = MultimodalInputs(mm_items=[])
        for pad in (
            pad_mm_tokens,
            partial(pad_mm_token_pairs, data_token_pairs=[(10, 11)]),
            MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens,
            MultiModalityDataPaddingPatternTokenPairs([(10, 11)]).pad_input_tokens,
        ):
            for input_ids in ([1, 2], array("i", [1, 2])):
                with self.subTest(pad=pad, input_ids=input_ids):
                    with self.assertRaisesRegex(AssertionError, "input_ids must be"):
                        pad(input_ids, mm_inputs)

    def test_multimodal_offsets_and_input_ownership(self):
        mm_inputs = MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE, pad_value=1000, offsets=[(1, 2), (7, 7)]
                ),
                MultimodalDataItem(
                    modality=Modality.IMAGE, pad_value=1001, offsets=[(3, 3)]
                ),
                MultimodalDataItem(
                    modality=Modality.AUDIO, pad_value=1002, offsets=[(4, 5)]
                ),
                MultimodalDataItem(
                    modality=Modality.VIDEO, pad_value=1003, offsets=[(8, 9)]
                ),
            ],
            im_token_id=10,
            audio_token_id=20,
            video_token_id=30,
        )
        original = array("q", [1, 10, 10, 10, 20, 20, 2, 10, 30, 30, 3])
        expected = array("q", [1, 1000, 1000, 1001, 1002, 1002, 2, 1000, 1003, 1003, 3])
        for pad in (
            pad_mm_tokens,
            MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens,
        ):
            with self.subTest(pad=pad):
                input_ids = original[:]
                padded = pad(input_ids, mm_inputs)
                self.assertEqual(padded, expected)
                self.assertEqual(input_ids, original)
                self.assertIsNot(padded, input_ids)

    def test_missing_modality_token_is_not_padded(self):
        mm_inputs = MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.AUDIO, pad_value=1000, offsets=[(0, 1)]
                )
            ],
            im_token_id=10,
        )
        input_ids = array("q", [20, 20])
        self.assertEqual(pad_mm_tokens(input_ids, mm_inputs), input_ids)

    def test_noop_preserves_identity_but_scheduler_owns_copy(self):
        for input_ids in (array("q"), array("q", [1, 2])):
            mm_inputs = MultimodalInputs(mm_items=[])
            for pad in (
                pad_mm_tokens,
                MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens,
            ):
                with self.subTest(input_ids=input_ids, pad=pad):
                    self.assertIs(pad(input_ids, mm_inputs), input_ids)
                    owned = duplicate_pad_mm_input_ids(input_ids, mm_inputs, pad)
                    self.assertEqual(owned, input_ids)
                    self.assertIsNot(owned, input_ids)

    def test_token_pairs_reuse_item_for_slices(self):
        pairs = [(10, 11), (12, 13), (20, 21)]
        starts = [10, 20]
        mm_inputs = MultimodalInputs(
            mm_items=[
                MultimodalDataItem(modality=Modality.IMAGE, pad_value=1000),
                MultimodalDataItem(modality=Modality.AUDIO, pad_value=2000),
            ]
        )
        original = array("q", [1, 10, 0, 0, 11, 12, 0, 13, 2, 20, 0, 21, 3])
        expected = array("q", [1, 10, 1000, 1000, 11, 12, 1000, 13, 2, 20, 2000, 21, 3])
        for pad in (
            partial(
                pad_mm_token_pairs, data_token_pairs=pairs, data_start_token_ids=starts
            ),
            MultiModalityDataPaddingPatternTokenPairs(pairs, starts).pad_input_tokens,
        ):
            with self.subTest(pad=pad):
                mm_inputs.data_offsets = [999]
                input_ids = original[:]
                self.assertEqual(pad(input_ids, mm_inputs), expected)
                self.assertEqual(input_ids, original)
                self.assertEqual(mm_inputs.data_offsets, [1, 9])

    def test_token_pairs_default_starts_and_repeated_regions(self):
        mm_inputs = MultimodalInputs(
            mm_items=[
                MultimodalDataItem(modality=Modality.IMAGE, pad_value=1000),
                MultimodalDataItem(modality=Modality.IMAGE, pad_value=2000),
            ]
        )
        input_ids = array("q", [10, 0, 11, 10, 11, 10, 0, 11])
        expected = array("q", [10, 1000, 11, 10, 11, 10, 2000, 11])
        for pad in (
            partial(pad_mm_token_pairs, data_token_pairs=[(10, 11)]),
            MultiModalityDataPaddingPatternTokenPairs([(10, 11)]).pad_input_tokens,
        ):
            with self.subTest(pad=pad):
                self.assertEqual(pad(input_ids, mm_inputs), expected)
                self.assertEqual(mm_inputs.data_offsets, [0, 3, 5])

    def test_unmatched_pairs_return_original_and_reset_offsets(self):
        mm_inputs = MultimodalInputs(
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE, pad_value=1000)]
        )
        for pad in (
            partial(pad_mm_token_pairs, data_token_pairs=[(10, 11)]),
            MultiModalityDataPaddingPatternTokenPairs([(10, 11)]).pad_input_tokens,
        ):
            with self.subTest(pad=pad):
                mm_inputs.data_offsets = [999]
                input_ids = array("q", [1, 10, 0])
                self.assertIs(pad(input_ids, mm_inputs), input_ids)
                self.assertEqual(mm_inputs.data_offsets, [])

    def test_legacy_configuration_remains_mutable_without_warnings(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pattern = MultiModalityDataPaddingPatternTokenPairs(
                data_token_pairs=[(10, 11)], data_start_token_ids=[10]
            )
            self.assertIsInstance(pattern, MultiModalityDataPaddingPattern)
            pattern.data_token_id_pairs = [(20, 21)]
            pattern.data_start_token_ids = []
            mm_inputs = MultimodalInputs(
                mm_items=[
                    MultimodalDataItem(modality=Modality.IMAGE, pad_value=1000),
                    MultimodalDataItem(modality=Modality.IMAGE, pad_value=2000),
                ]
            )
            self.assertEqual(
                pattern.pad_input_tokens(array("q", [20, 0, 21]), mm_inputs),
                array("q", [20, 2000, 21]),
            )
            self.assertEqual(mm_inputs.data_offsets, [])
        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
