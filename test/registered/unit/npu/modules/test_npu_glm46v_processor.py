"""
Unit tests for sglang.srt.hardware_backend.npu.modules.glm46v_processor.

Tests are ordered to match the source file:
  1. npu_wrapper_glm46v_preprocess        (L36)  — image preprocessing
  2. npu_wrapper_glm46v_video_preprocess  (L154) — video preprocessing
  3. npu_apply_glm46v_image_preprocess_patch (L271)
"""

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

from sglang.srt.hardware_backend.npu.modules.glm46v_processor import (
    npu_apply_glm46v_image_preprocess_patch,
    npu_wrapper_glm46v_preprocess,
    npu_wrapper_glm46v_video_preprocess,
)


def _make_image_processor():
    """Mock processor: resize and rescale_and_normalize are identity."""
    return SimpleNamespace(
        resize=MagicMock(side_effect=lambda x, *a, **kw: x),
        rescale_and_normalize=MagicMock(side_effect=lambda x, *a, **kw: x),
    )


# ===========================================================================
# 1. npu_wrapper_glm46v_preprocess  (source L36–149)
# ===========================================================================
class TestNpuWrapperGlm46vPreprocess(unittest.TestCase):
    def _call(self, proc, images, **ov):
        """Call the image _preprocess wrapper with sensible defaults."""
        wrapped = npu_wrapper_glm46v_preprocess(MagicMock())
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
            disable_grouping=True,
            return_tensors="pt",
        )

    # -- L36: decorator returns a callable --
    def test_decorator_returns_callable(self):
        self.assertTrue(callable(npu_wrapper_glm46v_preprocess(MagicMock())))

    # -- L56-77: do_resize=False → resize NOT called --
    def test_no_resize_when_do_resize_false(self):
        proc = _make_image_processor()
        self._call(proc, [torch.randn(3, 4, 4)])
        self.assertFalse(proc.resize.called)

    # -- L62-76: do_resize=True → resize called --
    def test_resize_called_when_do_resize_true(self):
        proc = _make_image_processor()
        self._call(proc, [torch.randn(3, 4, 4)], do_resize=True)
        self.assertTrue(proc.resize.called)

    # -- L90-97: rescale_and_normalize called --
    def test_rescale_and_normalize_called(self):
        proc = _make_image_processor()
        self._call(proc, [torch.randn(3, 4, 4)])
        self.assertTrue(proc.rescale_and_normalize.called)

    # -- L98-99: patches.ndim==4 → unsqueeze(1) adds temporal dim --
    def test_4d_input_unsqueezed_to_5d(self):
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        # With unsqueeze, the output shape is (4, 12) for 1 image, patch_size=2
        self.assertEqual(result["pixel_values"].shape, (4, 12))

    # -- L101-109: temporal padding when tps does not divide temporal dim --
    def test_temporal_padding_when_not_divisible(self):
        """temporal_patch_size=2 with 1 frame → 1 frame padded to 2."""
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)], temporal_patch_size=2)
        # After padding: 2 frames, grid_t=1, grid_h=2, grid_w=2
        # last_dim = channel * tps * ps * ps = 3*2*2*2 = 24
        self.assertEqual(result["pixel_values"].shape, (4, 24))
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L101: temporal_patch_size=1 → no padding --
    def test_no_padding_when_divisible(self):
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)], temporal_patch_size=1)
        self.assertEqual(result["pixel_values"].shape, (4, 12))

    # -- L118-128: transform_patches_to_flatten called (verified via output shape) --
    def test_pixel_values_shape(self):
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["pixel_values"].shape, (4, 12))

    def test_multiple_images_same_shape(self):
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(3, 4, 4), torch.randn(3, 4, 4)])
        self.assertEqual(result["pixel_values"].shape, (8, 12))

    # -- L134: image_grid_thw values --
    def test_image_grid_thw_values(self):
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L144-147: output is BatchFeature with pixel_values and image_grid_thw --
    def test_output_is_batch_feature(self):
        from transformers.image_processing_utils import BatchFeature

        result = self._call(_make_image_processor(), [torch.randn(3, 4, 4)])
        self.assertIsInstance(result, BatchFeature)
        self.assertIn("pixel_values", result.data)
        self.assertIn("image_grid_thw", result.data)


# ===========================================================================
# 2. npu_wrapper_glm46v_video_preprocess  (source L154–265)
# ===========================================================================
class TestNpuWrapperGlm46vVideoPreprocess(unittest.TestCase):
    def _call(self, proc, videos, **ov):
        """Call the video _preprocess wrapper with sensible defaults."""
        wrapped = npu_wrapper_glm46v_video_preprocess(MagicMock())
        return wrapped(
            self=proc,
            videos=videos,
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
            return_tensors="pt",
        )

    # -- L154: decorator returns a callable --
    def test_decorator_returns_callable(self):
        self.assertTrue(callable(npu_wrapper_glm46v_video_preprocess(MagicMock())))

    # -- L180-198: do_resize=False → resize NOT called --
    def test_no_resize_when_do_resize_false(self):
        proc = _make_image_processor()
        self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertFalse(proc.resize.called)

    # -- L180-198: do_resize=True → resize called --
    def test_resize_called_when_do_resize_true(self):
        proc = _make_image_processor()
        self._call(proc, [torch.randn(1, 3, 4, 4)], do_resize=True)
        self.assertTrue(proc.resize.called)

    # -- L213-220: rescale_and_normalize called --
    def test_rescale_and_normalize_called(self):
        proc = _make_image_processor()
        self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertTrue(proc.rescale_and_normalize.called)

    # -- L224-226: temporal padding when tps does not divide temporal dim --
    def test_temporal_padding_when_not_divisible(self):
        """temporal_patch_size=2 with 1 frame → padded to 2."""
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)], temporal_patch_size=2)
        # After padding: 2 frames, grid_t=1, last_dim = 3*2*2*2 = 24
        self.assertEqual(result["pixel_values_videos"].shape, (4, 24))
        self.assertEqual(result["video_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L224: temporal_patch_size=1 → no padding --
    def test_no_padding_when_divisible(self):
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)], temporal_patch_size=1)
        self.assertEqual(result["pixel_values_videos"].shape, (4, 12))

    # -- L234-244: transform_patches_to_flatten (verified via output shape) --
    def test_pixel_values_videos_shape(self):
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertEqual(result["pixel_values_videos"].shape, (4, 12))

    def test_multiple_videos_same_shape(self):
        proc = _make_image_processor()
        result = self._call(
            proc, [torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)]
        )
        self.assertEqual(result["pixel_values_videos"].shape, (8, 12))

    # -- L250: video_grid_thw values --
    def test_video_grid_thw_values(self):
        proc = _make_image_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertEqual(result["video_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L263: output is BatchFeature with video keys --
    def test_output_is_batch_feature(self):
        from transformers.image_processing_utils import BatchFeature

        result = self._call(_make_image_processor(), [torch.randn(1, 3, 4, 4)])
        self.assertIsInstance(result, BatchFeature)
        self.assertIn("pixel_values_videos", result.data)
        self.assertIn("video_grid_thw", result.data)


# ===========================================================================
# 3. npu_apply_glm46v_image_preprocess_patch  (source L271–285)
# ===========================================================================
class TestNpuApplyGlm46vImagePreprocessPatch(unittest.TestCase):
    def setUp(self):
        import sglang.srt.hardware_backend.npu.modules.glm46v_processor as mod
        self._mod = mod
        self._original_flag = mod._npu_glm46v_preprocess_patched
        mod._npu_glm46v_preprocess_patched = False

    def tearDown(self):
        self._mod._npu_glm46v_preprocess_patched = self._original_flag

    # -- L275-284: calls apply_module_patch for both image and video --
    @patch("sglang.srt.hardware_backend.npu.modules.glm46v_processor.apply_module_patch")
    def test_calls_apply_module_patch(self, mock_apply):
        npu_apply_glm46v_image_preprocess_patch()
        self.assertEqual(mock_apply.call_count, 2)
        targets = [call.args[0] for call in mock_apply.call_args_list]
        self.assertIn(
            "transformers.models.glm46v.image_processing_glm46v_fast.Glm46VImageProcessorFast",
            targets,
        )
        self.assertIn(
            "transformers.models.glm46v.video_processing_glm46v.Glm46VVideoProcessor",
            targets,
        )

    # -- L285: sets global flag --
    @patch("sglang.srt.hardware_backend.npu.modules.glm46v_processor.apply_module_patch")
    def test_sets_global_flag(self, mock_apply):
        npu_apply_glm46v_image_preprocess_patch()
        self.assertTrue(self._mod._npu_glm46v_preprocess_patched)

    # -- L273-274: idempotent — second call is a no-op --
    @patch("sglang.srt.hardware_backend.npu.modules.glm46v_processor.apply_module_patch")
    def test_idempotent(self, mock_apply):
        npu_apply_glm46v_image_preprocess_patch()
        npu_apply_glm46v_image_preprocess_patch()
        self.assertEqual(mock_apply.call_count, 2)

    # -- L273: already patched → apply_module_patch NOT called --
    @patch("sglang.srt.hardware_backend.npu.modules.glm46v_processor.apply_module_patch")
    def test_does_not_call_when_already_patched(self, mock_apply):
        self._mod._npu_glm46v_preprocess_patched = True
        npu_apply_glm46v_image_preprocess_patch()
        mock_apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
