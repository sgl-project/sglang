"""
Unit tests for sglang.srt.hardware_backend.npu.modules.minimax_m3_processor.

Tests are ordered to match the source file:
  1. _round_by_factor                              (L33)
  2. _ceil_by_factor                               (L37)
  3. _floor_by_factor                              (L41)
  4. _smart_resize                                  (L45)
  5. npu_wrapper_minimax_m3_image_preprocess       (L70)
  6. npu_wrapper_minimax_m3_video_preprocess       (L179)
  7. npu_apply_minimax_m3_image_preprocess_patch   (L288)
  8. npu_apply_minimax_m3_video_preprocess_patch   (L296)
"""

import math
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies BEFORE importing sglang.
for _ in (
    "triton",
    "triton.language",
    "triton.runtime",
    "IPython",
    "IPython.display",
    "aiohttp",
    "vllm_ascend",
    "batch_invariant_ops",
):
    sys.modules.setdefault(_, MagicMock())

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=4, suite="stage-a-unit-test-npu")

from sglang.srt.hardware_backend.npu.modules.minimax_m3_processor import (
    _ceil_by_factor,
    _floor_by_factor,
    _round_by_factor,
    _smart_resize,
    npu_apply_minimax_m3_image_preprocess_patch,
    npu_apply_minimax_m3_video_preprocess_patch,
    npu_wrapper_minimax_m3_image_preprocess,
    npu_wrapper_minimax_m3_video_preprocess,
)


def _make_processor():
    """Mock processor: resize and rescale_and_normalize are identity."""
    return SimpleNamespace(
        resize=MagicMock(side_effect=lambda x, *a, **kw: x),
        rescale_and_normalize=MagicMock(side_effect=lambda x, *a, **kw: x),
    )


# ===========================================================================
# 1. _round_by_factor  (source L33–34)
# ===========================================================================
class TestRoundByFactor(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(_round_by_factor(28, 28), 28)
        self.assertEqual(_round_by_factor(56, 28), 56)

    def test_less_than_half_rounds_to_zero(self):
        """round(0.5) in Python uses banker's rounding → 0."""
        self.assertEqual(_round_by_factor(14, 28), 0)
        self.assertEqual(_round_by_factor(13, 28), 0)

    def test_just_below_one_rounds_up(self):
        """round(0.96) = 1 → rounds up to factor."""
        self.assertEqual(_round_by_factor(27, 28), 28)

    def test_above_one_rounds_up(self):
        """round(1.5) = 2 (banker's rounding to even)."""
        self.assertEqual(_round_by_factor(42, 28), 56)

    def test_zero(self):
        self.assertEqual(_round_by_factor(0, 28), 0)

    def test_factor_one(self):
        self.assertEqual(_round_by_factor(5, 1), 5)

    def test_large_number(self):
        self.assertEqual(_round_by_factor(1000, 28), 1008)


# ===========================================================================
# 2. _ceil_by_factor  (source L37–38)
# ===========================================================================
class TestCeilByFactor(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(_ceil_by_factor(28, 28), 28)
        self.assertEqual(_ceil_by_factor(56, 28), 56)

    def test_non_exact_rounds_up(self):
        self.assertEqual(_ceil_by_factor(29, 28), 56)
        self.assertEqual(_ceil_by_factor(1, 28), 28)

    def test_zero(self):
        self.assertEqual(_ceil_by_factor(0, 28), 0)

    def test_factor_one(self):
        self.assertEqual(_ceil_by_factor(5, 1), 5)

    def test_large_number(self):
        self.assertEqual(_ceil_by_factor(1000, 28), 1008)


# ===========================================================================
# 3. _floor_by_factor  (source L41–42)
# ===========================================================================
class TestFloorByFactor(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(_floor_by_factor(28, 28), 28)
        self.assertEqual(_floor_by_factor(56, 28), 56)

    def test_non_exact_rounds_down(self):
        self.assertEqual(_floor_by_factor(27, 28), 0)
        self.assertEqual(_floor_by_factor(55, 28), 28)

    def test_zero(self):
        self.assertEqual(_floor_by_factor(0, 28), 0)

    def test_factor_one(self):
        self.assertEqual(_floor_by_factor(5, 1), 5)

    def test_large_number(self):
        self.assertEqual(_floor_by_factor(1000, 28), 980)


# ===========================================================================
# 4. _smart_resize  (source L45–67)
# ===========================================================================
class TestSmartResize(unittest.TestCase):
    # -- L57-58: exact multiple, no scaling --
    def test_exact_multiple_no_scaling(self):
        """56 is a multiple of 28; 56*56=3136=min_pixels → no scaling."""
        h, w = _smart_resize(56, 56, factor=28)
        self.assertEqual((h, w), (56, 56))

    # -- L57: dimensions clamped to at least factor --
    def test_small_dimensions_clamped_to_factor(self):
        """10 → round(10/28)=0 → max(28, 0)=28, then scaled up by min_pixels."""
        h, w = _smart_resize(10, 10, factor=28)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)
        self.assertGreaterEqual(h * w, 4 * 28 * 28)

    # -- L57: non-multiple rounds to nearest factor --
    def test_non_multiple_rounded_no_scaling(self):
        """With min_pixels=0, max_pixels=large, 29 rounds to 28, no scaling."""
        h, w = _smart_resize(29, 29, factor=28, min_pixels=0, max_pixels=1000000)
        self.assertEqual((h, w), (28, 28))

    def test_non_multiple_rounds_up_no_scaling(self):
        """100 → round(3.57)=4 → 112, no scaling."""
        h, w = _smart_resize(100, 100, factor=28, min_pixels=0, max_pixels=1000000)
        self.assertEqual((h, w), (112, 112))

    # -- L59-62: h_bar * w_bar > max_pixels → scale down --
    def test_large_image_scaled_down(self):
        h, w = _smart_resize(1000, 1000, factor=28, max_pixels=451584)
        self.assertLessEqual(h * w, 451584 + 28 * 28)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    # -- L63-66: h_bar * w_bar < min_pixels → scale up --
    def test_small_image_scaled_up(self):
        h, w = _smart_resize(10, 10, factor=28, min_pixels=3136)
        self.assertGreaterEqual(h * w, 3136)
        self.assertEqual(h % 28, 0)
        self.assertEqual(w % 28, 0)

    # -- L52: aspect ratio > MAX_RATIO(200) → ValueError --
    def test_aspect_ratio_exceeds_max_raises(self):
        with self.assertRaises(ValueError):
            _smart_resize(1, 201, factor=28)

    # -- L52: aspect ratio == MAX_RATIO → no raise --
    def test_aspect_ratio_at_max_boundary(self):
        try:
            _smart_resize(1, 200, factor=28)
        except ValueError:
            self.fail("Aspect ratio of exactly 200 should not raise ValueError")

    # -- L57-58: non-square image --
    def test_non_square_image(self):
        h, w = _smart_resize(100, 200, factor=28, min_pixels=0, max_pixels=1000000)
        self.assertEqual(h, 112)
        self.assertEqual(w, 196)
        self.assertNotEqual(h, w)

    # -- L48-50: custom factor --
    def test_custom_factor(self):
        h, w = _smart_resize(5, 5, factor=2, min_pixels=4, max_pixels=10000)
        self.assertEqual((h, w), (4, 4))

    # -- L51: returns tuple of ints --
    def test_returns_tuple_of_ints(self):
        result = _smart_resize(100, 100, factor=28)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        for v in result:
            self.assertIsInstance(v, int)


# ===========================================================================
# 5. npu_wrapper_minimax_m3_image_preprocess  (source L70–176)
# ===========================================================================
class TestNpuWrapperMinimaxM3ImagePreprocess(unittest.TestCase):
    def _call(self, proc, images, **ov):
        """Call the image _preprocess wrapper with sensible defaults."""
        wrapped = npu_wrapper_minimax_m3_image_preprocess(MagicMock())
        return wrapped(
            self=proc,
            images=images,
            do_resize=ov.get("do_resize", False),
            size=MagicMock(shortest_edge=4, longest_edge=10000),
            resample=None,
            do_rescale=False,
            rescale_factor=1.0,
            do_normalize=False,
            image_mean=None,
            image_std=None,
            patch_size=ov.get("patch_size", 2),
            temporal_patch_size=ov.get("temporal_patch_size", 1),
            merge_size=ov.get("merge_size", 1),
            max_pixels=ov.get("max_pixels", 10000),
            disable_grouping=True,
            return_tensors="pt",
        )

    # -- L70: decorator returns a callable --
    def test_decorator_returns_callable(self):
        self.assertTrue(callable(npu_wrapper_minimax_m3_image_preprocess(MagicMock())))

    # -- L98-109: do_resize=False → resize NOT called --
    def test_no_resize_when_do_resize_false(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(3, 4, 4)])
        self.assertFalse(proc.resize.called)

    # -- L98-109: do_resize=True → resize called --
    def test_resize_called_when_do_resize_true(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(3, 4, 4)], do_resize=True)
        self.assertTrue(proc.resize.called)

    # -- L123-130: rescale_and_normalize called --
    def test_rescale_and_normalize_called(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(3, 4, 4)])
        self.assertTrue(proc.rescale_and_normalize.called)

    # -- L131-132: patches.ndim==4 → unsqueeze(1) adds temporal dim --
    def test_4d_input_unsqueezed_to_5d(self):
        """Identity rescale returns 4D; unsqueeze makes it 5D for transform."""
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["pixel_values"].shape, (4, 12))

    # -- L134-142: temporal padding when tps does not divide temporal dim --
    def test_temporal_padding_when_not_divisible(self):
        """temporal_patch_size=2 with 1 frame → padded to 2."""
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)], temporal_patch_size=2)
        # last_dim = channel * tps * ps * ps = 3*2*2*2 = 24
        self.assertEqual(result["pixel_values"].shape, (4, 24))
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L134: temporal_patch_size=1 → no padding --
    def test_no_padding_when_divisible(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)], temporal_patch_size=1)
        self.assertEqual(result["pixel_values"].shape, (4, 12))

    # -- L148-158: transform_patches_to_flatten (verified via output shape) --
    def test_pixel_values_shape(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["pixel_values"].shape, (4, 12))

    def test_multiple_images_same_shape(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4), torch.randn(3, 4, 4)])
        self.assertEqual(result["pixel_values"].shape, (8, 12))

    # -- L161: image_grid_thw values --
    def test_image_grid_thw_values(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L169: image_grid_thw dtype is long --
    def test_image_grid_thw_dtype(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["image_grid_thw"].dtype, torch.long)

    # -- L171-174: output is BatchFeature with pixel_values and image_grid_thw --
    def test_output_is_batch_feature(self):
        from transformers.image_processing_utils import BatchFeature

        result = self._call(_make_processor(), [torch.randn(3, 4, 4)])
        self.assertIsInstance(result, BatchFeature)
        self.assertIn("pixel_values", result.data)
        self.assertIn("image_grid_thw", result.data)


# ===========================================================================
# 6. npu_wrapper_minimax_m3_video_preprocess  (source L179–285)
# ===========================================================================
class TestNpuWrapperMinimaxM3VideoPreprocess(unittest.TestCase):
    def _call(self, proc, videos, **ov):
        """Call the video _preprocess wrapper with sensible defaults."""
        wrapped = npu_wrapper_minimax_m3_video_preprocess(MagicMock())
        return wrapped(
            self=proc,
            videos=videos,
            do_convert_rgb=True,
            do_resize=ov.get("do_resize", False),
            size=MagicMock(shortest_edge=4, longest_edge=10000),
            resample=None,
            do_rescale=False,
            rescale_factor=1.0,
            do_normalize=False,
            image_mean=None,
            image_std=None,
            patch_size=ov.get("patch_size", 2),
            temporal_patch_size=ov.get("temporal_patch_size", 1),
            merge_size=ov.get("merge_size", 1),
            min_pixels=ov.get("min_pixels", 4),
            max_pixels=ov.get("max_pixels", 10000),
            return_tensors="pt",
        )

    # -- L179: decorator returns a callable --
    def test_decorator_returns_callable(self):
        self.assertTrue(callable(npu_wrapper_minimax_m3_video_preprocess(MagicMock())))

    # -- L207-230: do_resize=False → resize NOT called --
    def test_no_resize_when_do_resize_false(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertFalse(proc.resize.called)

    # -- L207-230: do_resize=True → resize called --
    def test_resize_called_when_do_resize_true(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(1, 3, 4, 4)], do_resize=True)
        self.assertTrue(proc.resize.called)

    # -- L238-245: rescale_and_normalize called --
    def test_rescale_and_normalize_called(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertTrue(proc.rescale_and_normalize.called)

    # -- L247-249: temporal padding (expand, not repeat) --
    def test_temporal_padding_when_not_divisible(self):
        """temporal_patch_size=2 with 1 frame → expand to 2."""
        proc = _make_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)], temporal_patch_size=2)
        self.assertEqual(result["pixel_values_videos"].shape, (4, 24))
        self.assertEqual(result["video_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L247: temporal_patch_size=1 → no padding --
    def test_no_padding_when_divisible(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)], temporal_patch_size=1)
        self.assertEqual(result["pixel_values_videos"].shape, (4, 12))

    # -- L255-265: transform_patches_to_flatten (verified via output shape) --
    def test_pixel_values_videos_shape(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertEqual(result["pixel_values_videos"].shape, (4, 12))

    def test_multiple_videos_same_shape(self):
        proc = _make_processor()
        result = self._call(
            proc, [torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)]
        )
        self.assertEqual(result["pixel_values_videos"].shape, (8, 12))

    # -- L268: video_grid_thw values --
    def test_video_grid_thw_values(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertEqual(result["video_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L275: video_grid_thw dtype is long --
    def test_video_grid_thw_dtype(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertEqual(result["video_grid_thw"].dtype, torch.long)

    # -- L277-283: output is BatchFeature with video keys --
    def test_output_is_batch_feature(self):
        from transformers.image_processing_utils import BatchFeature

        result = self._call(_make_processor(), [torch.randn(1, 3, 4, 4)])
        self.assertIsInstance(result, BatchFeature)
        self.assertIn("pixel_values_videos", result.data)
        self.assertIn("video_grid_thw", result.data)


# ===========================================================================
# 7. npu_apply_minimax_m3_image_preprocess_patch  (source L288–293)
# ===========================================================================
class TestNpuApplyMinimaxM3ImagePreprocessPatch(unittest.TestCase):
    def _make_processor(self):
        class FakeProcessor:
            def _preprocess(self):
                pass

        return FakeProcessor()

    # -- L293: sets _sglang_npu_patched flag --
    def test_sets_patched_flag(self):
        proc = self._make_processor()
        npu_apply_minimax_m3_image_preprocess_patch(proc)
        self.assertTrue(getattr(type(proc), "_sglang_npu_patched", False))

    # -- L292: replaces _preprocess --
    def test_replaces_preprocess(self):
        proc = self._make_processor()
        original = type(proc)._preprocess
        npu_apply_minimax_m3_image_preprocess_patch(proc)
        self.assertIsNot(type(proc)._preprocess, original)

    # -- L290-291: idempotent — second call is a no-op --
    def test_idempotent(self):
        proc = self._make_processor()
        npu_apply_minimax_m3_image_preprocess_patch(proc)
        first = type(proc)._preprocess
        npu_apply_minimax_m3_image_preprocess_patch(proc)
        self.assertIs(type(proc)._preprocess, first)

    # -- L290: already patched → _preprocess NOT replaced --
    def test_does_not_patch_when_already_patched(self):
        proc = self._make_processor()
        type(proc)._sglang_npu_patched = True
        original = type(proc)._preprocess
        npu_apply_minimax_m3_image_preprocess_patch(proc)
        self.assertIs(type(proc)._preprocess, original)


# ===========================================================================
# 8. npu_apply_minimax_m3_video_preprocess_patch  (source L296–301)
# ===========================================================================
class TestNpuApplyMinimaxM3VideoPreprocessPatch(unittest.TestCase):
    def _make_processor(self):
        class FakeProcessor:
            def _preprocess(self):
                pass

        return FakeProcessor()

    # -- L301: sets _sglang_npu_video_patched flag --
    def test_sets_patched_flag(self):
        proc = self._make_processor()
        npu_apply_minimax_m3_video_preprocess_patch(proc)
        self.assertTrue(getattr(type(proc), "_sglang_npu_video_patched", False))

    # -- L300: replaces _preprocess --
    def test_replaces_preprocess(self):
        proc = self._make_processor()
        original = type(proc)._preprocess
        npu_apply_minimax_m3_video_preprocess_patch(proc)
        self.assertIsNot(type(proc)._preprocess, original)

    # -- L298-299: idempotent --
    def test_idempotent(self):
        proc = self._make_processor()
        npu_apply_minimax_m3_video_preprocess_patch(proc)
        first = type(proc)._preprocess
        npu_apply_minimax_m3_video_preprocess_patch(proc)
        self.assertIs(type(proc)._preprocess, first)

    # -- L298: already patched → _preprocess NOT replaced --
    def test_does_not_patch_when_already_patched(self):
        proc = self._make_processor()
        type(proc)._sglang_npu_video_patched = True
        original = type(proc)._preprocess
        npu_apply_minimax_m3_video_preprocess_patch(proc)
        self.assertIs(type(proc)._preprocess, original)


if __name__ == "__main__":
    unittest.main()
