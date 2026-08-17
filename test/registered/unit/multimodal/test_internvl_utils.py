"""Unit tests for srt/multimodal/internvl_utils.py — no server, no model weights.

Note: `internvl_utils.py` depends on `torchvision`. If `torchvision` is not
installed in the unit-test environment, these tests are skipped rather than
failing import-time.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

import torch
from PIL import Image

from sglang.test.test_utils import CustomTestCase

try:
    from sglang.srt.multimodal import internvl_utils
except ImportError:  # pragma: no cover
    internvl_utils = None


@unittest.skipIf(internvl_utils is None, "torchvision / internvl_utils not available")
class TestFindClosestAspectRatio(CustomTestCase):
    def test_picks_closest_ratio(self):
        ratios = [(1, 1), (2, 1), (3, 2)]
        best = internvl_utils.find_closest_aspect_ratio(
            aspect_ratio=1.6, target_ratios=ratios, width=160, height=100, image_size=32
        )
        self.assertEqual(best, (3, 2))

    def test_tie_breaker_prefers_larger_area_conditionally(self):
        ratios = [(1, 1), (2, 2)]
        # Same aspect ratio; tie broken by area threshold check.
        best = internvl_utils.find_closest_aspect_ratio(
            aspect_ratio=1.0, target_ratios=ratios, width=128, height=128, image_size=32
        )
        self.assertEqual(best, (2, 2))

    def test_empty_target_ratios_returns_default(self):
        best = internvl_utils.find_closest_aspect_ratio(
            aspect_ratio=1.0, target_ratios=[], width=10, height=10, image_size=32
        )
        self.assertEqual(best, (1, 1))

    def test_exact_match_is_selected(self):
        ratios = [(1, 1), (4, 3), (16, 9)]
        best = internvl_utils.find_closest_aspect_ratio(
            aspect_ratio=4 / 3,
            target_ratios=ratios,
            width=400,
            height=300,
            image_size=32,
        )
        self.assertEqual(best, (4, 3))


@unittest.skipIf(internvl_utils is None, "torchvision / internvl_utils not available")
class TestDynamicPreprocess(CustomTestCase):
    def test_single_block_no_thumbnail(self):
        img = Image.new("RGB", (64, 64), color=(1, 2, 3))
        tiles = internvl_utils.dynamic_preprocess(
            img, min_num=1, max_num=1, image_size=32, use_thumbnail=True
        )
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].size, (32, 32))

    def test_multiple_blocks_adds_thumbnail_when_enabled(self):
        img = Image.new("RGB", (128, 64), color=(1, 2, 3))
        tiles = internvl_utils.dynamic_preprocess(
            img, min_num=1, max_num=2, image_size=32, use_thumbnail=True
        )
        # When blocks != 1, thumbnail is appended.
        self.assertGreaterEqual(len(tiles), 2)
        self.assertEqual(tiles[-1].size, (32, 32))

    def test_multiple_blocks_no_thumbnail_when_disabled(self):
        img = Image.new("RGB", (128, 64), color=(1, 2, 3))
        tiles = internvl_utils.dynamic_preprocess(
            img, min_num=1, max_num=2, image_size=32, use_thumbnail=False
        )
        self.assertGreaterEqual(len(tiles), 1)
        self.assertNotEqual(len(tiles), 0)

    def test_all_tiles_have_expected_size(self):
        img = Image.new("RGB", (191, 83), color=(1, 2, 3))
        tiles = internvl_utils.dynamic_preprocess(
            img, min_num=1, max_num=4, image_size=32, use_thumbnail=True
        )
        self.assertTrue(all(tile.size == (32, 32) for tile in tiles))

    def test_extreme_wide_image_still_produces_tiles(self):
        img = Image.new("RGB", (512, 32), color=(1, 2, 3))
        tiles = internvl_utils.dynamic_preprocess(
            img, min_num=1, max_num=4, image_size=32, use_thumbnail=False
        )
        self.assertGreaterEqual(len(tiles), 1)


@unittest.skipIf(internvl_utils is None, "torchvision / internvl_utils not available")
class TestImageToPixelValues(CustomTestCase):
    def test_returns_stacked_tensor(self):
        img = Image.new("RGB", (64, 64), color=(1, 2, 3))
        pixel_values = internvl_utils.image_to_pixel_values(
            img, input_size=32, max_num_tiles=1, use_thumbnail=False
        )
        self.assertIsInstance(pixel_values, torch.Tensor)
        self.assertEqual(pixel_values.shape, (1, 3, 32, 32))

    def test_thumbnail_increases_tile_count(self):
        img = Image.new("RGB", (128, 64), color=(1, 2, 3))
        pv_no_thumb = internvl_utils.image_to_pixel_values(
            img, input_size=32, max_num_tiles=2, use_thumbnail=False
        )
        pv_thumb = internvl_utils.image_to_pixel_values(
            img, input_size=32, max_num_tiles=2, use_thumbnail=True
        )
        self.assertGreaterEqual(pv_thumb.shape[0], pv_no_thumb.shape[0])

    def test_output_is_normalized_float_tensor(self):
        img = Image.new("RGB", (64, 64), color=(255, 255, 255))
        pixel_values = internvl_utils.image_to_pixel_values(
            img, input_size=32, max_num_tiles=1, use_thumbnail=False
        )
        self.assertEqual(pixel_values.dtype, torch.float32)
        self.assertTrue(torch.isfinite(pixel_values).all().item())

    def test_custom_mean_std_changes_values(self):
        img = Image.new("RGB", (64, 64), color=(127, 127, 127))
        default_pv = internvl_utils.image_to_pixel_values(
            img, input_size=32, max_num_tiles=1, use_thumbnail=False
        )
        custom_pv = internvl_utils.image_to_pixel_values(
            img,
            input_size=32,
            max_num_tiles=1,
            use_thumbnail=False,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        self.assertFalse(torch.allclose(default_pv, custom_pv))


if __name__ == "__main__":
    unittest.main()
