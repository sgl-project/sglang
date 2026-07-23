"""
Unit tests for MultiModalMixin._collect_mm_kwargs' handling of 5D
pixel_values features in the generic Transformers fallback backend
(sglang.srt.models.transformers).

"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.transformers import MultiModalMixin


def _make_item(modality_name, feature, model_specific_data=None):
    return SimpleNamespace(
        modality=SimpleNamespace(name=modality_name),
        feature=feature,
        model_specific_data=model_specific_data or {},
    )


def _make_mm_input(items):
    return SimpleNamespace(mm_items=items)


def _make_forward_batch(mm_inputs, is_decode=False, contains_mm_inputs=True):
    return SimpleNamespace(
        token_type_ids=None,
        forward_mode=SimpleNamespace(is_decode=lambda: is_decode),
        mm_inputs=mm_inputs,
        contains_mm_inputs=lambda: contains_mm_inputs,
    )


def _make_self():
    """Lightweight stand-in for a TransformersForCausalLM instance -- only
    `.model` (for the device lookup) and the mixin's own feature-key map
    are actually used by `_collect_mm_kwargs`."""
    return SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        _mm_feature_kwarg=MultiModalMixin._mm_feature_kwarg,
    )


class TestCollectMmKwargs5DPadding(unittest.TestCase):
    def test_equal_patch_counts_no_padding(self):
        """Sanity check: same num_patches across items concatenates cleanly."""
        item1 = _make_item("IMAGE", torch.full((1, 3, 3, 4, 4), 1.0))
        item2 = _make_item("IMAGE", torch.full((1, 3, 3, 4, 4), 2.0))
        forward_batch = _make_forward_batch(
            [_make_mm_input([item1]), _make_mm_input([item2])]
        )

        kwargs = MultiModalMixin._collect_mm_kwargs(_make_self(), forward_batch)

        pixel_values = kwargs["pixel_values"]
        self.assertEqual(pixel_values.shape, (2, 3, 3, 4, 4))
        self.assertTrue(torch.all(pixel_values[0] == 1.0))
        self.assertTrue(torch.all(pixel_values[1] == 2.0))

    def test_different_patch_counts_padded_to_batch_max(self):
        """Test: items with a different tile count must be
        zero-padded to the batch-wide max num_patches, not just concatenated
        as-is (which would crash on mismatched shapes or misalign data)."""
        item_small = _make_item("IMAGE", torch.full((1, 3, 3, 4, 4), 1.0))  # 3 patches
        item_large = _make_item("IMAGE", torch.full((1, 5, 3, 4, 4), 2.0))  # 5 patches
        forward_batch = _make_forward_batch(
            [_make_mm_input([item_small]), _make_mm_input([item_large])]
        )

        kwargs = MultiModalMixin._collect_mm_kwargs(_make_self(), forward_batch)

        pixel_values = kwargs["pixel_values"]
        self.assertEqual(pixel_values.shape, (2, 5, 3, 4, 4))
        # item_small's real 3 patches are preserved...
        self.assertTrue(torch.all(pixel_values[0, :3] == 1.0))
        # ...and its padding (patches 3-4) is zeroed, not garbage/leftover data.
        self.assertTrue(torch.all(pixel_values[0, 3:] == 0.0))
        # item_large needed no padding at all.
        self.assertTrue(torch.all(pixel_values[1] == 2.0))

    def test_multi_image_item_with_different_patch_counts_within_one_item(self):
        """A single multi-image item/request can itself already contain
        per-image padding applied by the HF processor; the batch-level
        padding must still pad up to the overall max without disturbing it."""
        # 2 images already padded to 4 patches by the HF processor, batched
        # against another item that only needed 2 patches.
        item_multi_image = _make_item("IMAGE", torch.full((2, 4, 3, 4, 4), 1.0))
        item_single = _make_item("IMAGE", torch.full((1, 2, 3, 4, 4), 2.0))
        forward_batch = _make_forward_batch(
            [_make_mm_input([item_multi_image]), _make_mm_input([item_single])]
        )

        kwargs = MultiModalMixin._collect_mm_kwargs(_make_self(), forward_batch)

        pixel_values = kwargs["pixel_values"]
        self.assertEqual(pixel_values.shape, (3, 4, 3, 4, 4))
        self.assertTrue(torch.all(pixel_values[:2] == 1.0))
        self.assertTrue(torch.all(pixel_values[2, :2] == 2.0))
        self.assertTrue(torch.all(pixel_values[2, 2:] == 0.0))

    def test_decode_mode_skips_collection(self):
        """During decode (no new mm inputs to process this step), no
        multimodal kwargs should be produced even if mm_inputs is present."""
        item = _make_item("IMAGE", torch.full((1, 3, 3, 4, 4), 1.0))
        forward_batch = _make_forward_batch([_make_mm_input([item])], is_decode=True)

        kwargs = MultiModalMixin._collect_mm_kwargs(_make_self(), forward_batch)

        self.assertNotIn("pixel_values", kwargs)

    def test_non_image_modality_uses_correct_feature_key(self):
        """Video features (also potentially 5D) must land under their own
        kwarg key, not be mixed in with image pixel_values."""
        item = _make_item("VIDEO", torch.full((1, 3, 3, 4, 4), 1.0))
        forward_batch = _make_forward_batch([_make_mm_input([item])])

        kwargs = MultiModalMixin._collect_mm_kwargs(_make_self(), forward_batch)

        self.assertIn("pixel_values_videos", kwargs)
        self.assertNotIn("pixel_values", kwargs)


if __name__ == "__main__":
    unittest.main()
