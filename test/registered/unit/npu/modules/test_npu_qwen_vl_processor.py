"""
Unit tests for sglang.srt.hardware_backend.npu.modules.qwen_vl_processor.

Tests are ordered to match the source file:
  1. transform_patches_to_flatten            (L21)
  2. npu_wrapper_preprocess                  (L63)  — image
  3. npu_wrapper_video_preprocess            (L172) — video
  4. npu_apply_qwen_image_preprocess_patch   (L290)
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

from sglang.srt.hardware_backend.npu.modules.qwen_vl_processor import (
    npu_apply_qwen_image_preprocess_patch,
    npu_wrapper_preprocess,
    npu_wrapper_video_preprocess,
    transform_patches_to_flatten,
)


def _make_processor():
    """Mock processor: resize and rescale_and_normalize are identity.

    resize handles both positional (video L209) and keyword ``image=``
    (image L99) call styles.
    """
    return SimpleNamespace(
        resize=MagicMock(side_effect=lambda *a, **kw: a[0] if a else kw.get("image")),
        rescale_and_normalize=MagicMock(side_effect=lambda x, *a, **kw: x),
    )


# ===========================================================================
# 1. transform_patches_to_flatten  (source L21–58)
# ===========================================================================
class TestTransformPatchesToFlatten(unittest.TestCase):
    def _make_input(self, batch_size, grid_t, tps, channel, grid_h, grid_w, patch_size):
        """Create an input tensor of shape (B, grid_t*tps, C, grid_h*ps, grid_w*ps)."""
        T = grid_t * tps
        H = grid_h * patch_size
        W = grid_w * patch_size
        return torch.arange(
            batch_size * T * channel * H * W, dtype=torch.float32
        ).reshape(batch_size, T, channel, H, W)

    def test_output_shape_basic(self):
        patches = self._make_input(1, 1, 1, 3, 2, 2, 2)
        out = transform_patches_to_flatten(patches, 1, 1, 1, 3, 2, 2, 2, 1)
        self.assertEqual(out.shape, (1, 4, 12))

    def test_output_shape_multiple_batch(self):
        patches = self._make_input(2, 1, 1, 3, 2, 2, 2)
        out = transform_patches_to_flatten(patches, 2, 1, 1, 3, 2, 2, 2, 1)
        self.assertEqual(out.shape, (2, 4, 12))

    def test_output_shape_multiple_grid_t(self):
        patches = self._make_input(1, 3, 1, 3, 2, 2, 2)
        out = transform_patches_to_flatten(patches, 1, 3, 1, 3, 2, 2, 2, 1)
        self.assertEqual(out.shape, (1, 12, 12))

    def test_output_shape_temporal_patch_size_2(self):
        patches = self._make_input(1, 1, 2, 3, 2, 2, 2)
        out = transform_patches_to_flatten(patches, 1, 1, 2, 3, 2, 2, 2, 1)
        self.assertEqual(out.shape, (1, 4, 24))

    def test_output_shape_merge_size_2(self):
        patches = self._make_input(1, 1, 1, 3, 2, 4, 2)
        out = transform_patches_to_flatten(patches, 1, 1, 1, 3, 2, 4, 2, 2)
        self.assertEqual(out.shape, (1, 8, 12))

    def test_output_shape_large_patch(self):
        patches = self._make_input(1, 1, 1, 4, 3, 3, 4)
        out = transform_patches_to_flatten(patches, 1, 1, 1, 4, 3, 3, 4, 1)
        self.assertEqual(out.shape, (1, 9, 64))

    def test_element_preservation(self):
        patches = self._make_input(2, 2, 2, 3, 2, 2, 2)
        out = transform_patches_to_flatten(patches, 2, 2, 2, 3, 2, 2, 2, 1)
        self.assertEqual(out.numel(), patches.numel())
        self.assertTrue(torch.equal(out.flatten().sort().values, patches.flatten().sort().values))

    def test_all_zeros_input(self):
        patches = torch.zeros(1, 1, 3, 4, 4)
        out = transform_patches_to_flatten(patches, 1, 1, 1, 3, 2, 2, 2, 1)
        self.assertTrue(torch.all(out == 0))

    def test_contiguous_output(self):
        patches = self._make_input(1, 1, 1, 3, 2, 2, 2)
        out = transform_patches_to_flatten(patches, 1, 1, 1, 3, 2, 2, 2, 1)
        self.assertTrue(out.is_contiguous())

    def test_last_dim_equals_channel_tps_ps_ps(self):
        """Last dimension = channel * temporal_patch_size * patch_size * patch_size."""
        channel, tps, ps = 5, 3, 4
        patches = self._make_input(1, 1, tps, channel, 1, 1, ps)
        out = transform_patches_to_flatten(patches, 1, 1, tps, channel, 1, 1, ps, 1)
        self.assertEqual(out.shape[-1], channel * tps * ps * ps)

    def test_second_dim_equals_grid_t_h_w(self):
        """Second dimension = grid_t * grid_h * grid_w."""
        gt, gh, gw = 2, 3, 4
        patches = self._make_input(1, gt, 1, 1, gh, gw, 1)
        out = transform_patches_to_flatten(patches, 1, gt, 1, 1, gh, gw, 1, 1)
        self.assertEqual(out.shape[1], gt * gh * gw)

    def test_merge_size_2_element_preservation(self):
        patches = self._make_input(1, 1, 1, 3, 2, 4, 2)
        out = transform_patches_to_flatten(patches, 1, 1, 1, 3, 2, 4, 2, 2)
        self.assertEqual(out.numel(), patches.numel())
        self.assertTrue(torch.equal(out.flatten().sort().values, patches.flatten().sort().values))


# ===========================================================================
# 2. npu_wrapper_preprocess  (source L63–167) — image
# ===========================================================================
class TestNpuWrapperPreprocess(unittest.TestCase):
    def _call(self, proc, images, **ov):
        """Call the image _preprocess wrapper with sensible defaults."""
        wrapped = npu_wrapper_preprocess(MagicMock())
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

    # -- L63: decorator returns a callable --
    def test_decorator_returns_callable(self):
        self.assertTrue(callable(npu_wrapper_preprocess(MagicMock())))

    # -- L90-103: do_resize=False → resize NOT called --
    def test_no_resize_when_do_resize_false(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(3, 4, 4)])
        self.assertFalse(proc.resize.called)

    # -- L90-103: do_resize=True → resize called (uses image= keyword) --
    def test_resize_called_when_do_resize_true(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(3, 4, 4)], do_resize=True)
        self.assertTrue(proc.resize.called)

    # -- L116-123: rescale_and_normalize called --
    def test_rescale_and_normalize_called(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(3, 4, 4)])
        self.assertTrue(proc.rescale_and_normalize.called)

    # -- L124-126: patches.ndim==4 → unsqueeze(1) adds temporal dim --
    def test_4d_input_unsqueezed_to_5d(self):
        """Identity rescale returns 4D; unsqueeze makes it 5D for transform."""
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["pixel_values"].shape, (4, 12))

    # -- L127-129: temporal padding (uses tps-1, NOT tps - remainder) --
    def test_temporal_padding_when_not_divisible(self):
        """temporal_patch_size=2 with 1 frame → repeat(tps-1=1) → 2 frames."""
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)], temporal_patch_size=2)
        # last_dim = channel * tps * ps * ps = 3*2*2*2 = 24
        self.assertEqual(result["pixel_values"].shape, (4, 24))
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L127: temporal_patch_size=1 → no padding --
    def test_no_padding_when_divisible(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)], temporal_patch_size=1)
        self.assertEqual(result["pixel_values"].shape, (4, 12))

    # -- L137-147: transform_patches_to_flatten (verified via output shape) --
    def test_pixel_values_shape(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["pixel_values"].shape, (4, 12))

    def test_multiple_images_same_shape(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4), torch.randn(3, 4, 4)])
        self.assertEqual(result["pixel_values"].shape, (8, 12))

    # -- L153: image_grid_thw values --
    def test_image_grid_thw_values(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L160: image_grid_thw dtype (torch.tensor default is int64) --
    def test_image_grid_thw_dtype(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(3, 4, 4)])
        self.assertEqual(result["image_grid_thw"].dtype, torch.int64)

    # -- L162-165: output is BatchFeature with pixel_values and image_grid_thw --
    def test_output_is_batch_feature(self):
        from transformers.image_processing_utils import BatchFeature

        result = self._call(_make_processor(), [torch.randn(3, 4, 4)])
        self.assertIsInstance(result, BatchFeature)
        self.assertIn("pixel_values", result.data)
        self.assertIn("image_grid_thw", result.data)


# ===========================================================================
# 3. npu_wrapper_video_preprocess  (source L172–284) — video
# ===========================================================================
class TestNpuWrapperVideoPreprocess(unittest.TestCase):
    def _call(self, proc, videos, **ov):
        """Call the video _preprocess wrapper with sensible defaults."""
        wrapped = npu_wrapper_video_preprocess(MagicMock())
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

    # -- L172: decorator returns a callable --
    def test_decorator_returns_callable(self):
        self.assertTrue(callable(npu_wrapper_video_preprocess(MagicMock())))

    # -- L198-217: do_resize=False → resize NOT called --
    def test_no_resize_when_do_resize_false(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertFalse(proc.resize.called)

    # -- L198-217: do_resize=True → resize called (uses positional arg) --
    def test_resize_called_when_do_resize_true(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(1, 3, 4, 4)], do_resize=True)
        self.assertTrue(proc.resize.called)

    # -- L231-238: rescale_and_normalize called --
    def test_rescale_and_normalize_called(self):
        proc = _make_processor()
        self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertTrue(proc.rescale_and_normalize.called)

    # -- L242-245: temporal padding (expand, -T % tps formula) --
    def test_temporal_padding_when_not_divisible(self):
        """temporal_patch_size=2 with 1 frame → pad=1 → expand to 2."""
        proc = _make_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)], temporal_patch_size=2)
        self.assertEqual(result["pixel_values_videos"].shape, (4, 24))
        self.assertEqual(result["video_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L243: temporal_patch_size=1 → no padding --
    def test_no_padding_when_divisible(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)], temporal_patch_size=1)
        self.assertEqual(result["pixel_values_videos"].shape, (4, 12))

    # -- L253-263: transform_patches_to_flatten (verified via output shape) --
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

    # -- L269: video_grid_thw values --
    def test_video_grid_thw_values(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertEqual(result["video_grid_thw"].tolist(), [[1, 2, 2]])

    # -- L276: video_grid_thw dtype --
    def test_video_grid_thw_dtype(self):
        proc = _make_processor()
        result = self._call(proc, [torch.randn(1, 3, 4, 4)])
        self.assertEqual(result["video_grid_thw"].dtype, torch.int64)

    # -- L277-282: output is BatchFeature with video keys --
    def test_output_is_batch_feature(self):
        from transformers.image_processing_utils import BatchFeature

        result = self._call(_make_processor(), [torch.randn(1, 3, 4, 4)])
        self.assertIsInstance(result, BatchFeature)
        self.assertIn("pixel_values_videos", result.data)
        self.assertIn("video_grid_thw", result.data)


# ===========================================================================
# 4. npu_apply_qwen_image_preprocess_patch  (source L290–304)
# ===========================================================================
class TestNpuApplyQwenImagePreprocessPatch(unittest.TestCase):
    def setUp(self):
        import sglang.srt.hardware_backend.npu.modules.qwen_vl_processor as mod
        self._mod = mod
        self._original_flag = mod._npu_preprocess_patched
        mod._npu_preprocess_patched = False

    def tearDown(self):
        self._mod._npu_preprocess_patched = self._original_flag

    # -- L294-303: calls apply_module_patch for both image and video --
    @patch("sglang.srt.hardware_backend.npu.modules.qwen_vl_processor.apply_module_patch")
    def test_calls_apply_module_patch(self, mock_apply):
        npu_apply_qwen_image_preprocess_patch()
        self.assertEqual(mock_apply.call_count, 2)
        targets = [call.args[0] for call in mock_apply.call_args_list]
        self.assertIn(
            "transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor",
            targets,
        )
        self.assertIn(
            "transformers.models.qwen3_vl.video_processing_qwen3_vl.Qwen3VLVideoProcessor",
            targets,
        )

    # -- L304: sets global flag --
    @patch("sglang.srt.hardware_backend.npu.modules.qwen_vl_processor.apply_module_patch")
    def test_sets_global_flag(self, mock_apply):
        npu_apply_qwen_image_preprocess_patch()
        self.assertTrue(self._mod._npu_preprocess_patched)

    # -- L292-293: idempotent — second call is a no-op --
    @patch("sglang.srt.hardware_backend.npu.modules.qwen_vl_processor.apply_module_patch")
    def test_idempotent(self, mock_apply):
        npu_apply_qwen_image_preprocess_patch()
        npu_apply_qwen_image_preprocess_patch()
        self.assertEqual(mock_apply.call_count, 2)

    # -- L292: already patched → apply_module_patch NOT called --
    @patch("sglang.srt.hardware_backend.npu.modules.qwen_vl_processor.apply_module_patch")
    def test_does_not_call_when_already_patched(self, mock_apply):
        self._mod._npu_preprocess_patched = True
        npu_apply_qwen_image_preprocess_patch()
        mock_apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
