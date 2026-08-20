"""Unit tests for MiniMax M3's ``max_long_side_pixel`` media sizing.

The rules come from MiniMax's provider spec ("M3 外部供应商质检手册",
格式正确性检查 §4) and are exercised by the public conformance suite
(MiniMax-AI/MiniMax-Provider-Verifier, ``m3_format_check``). The tier numbers
below are the ones that suite asserts on, so they are pinned here rather than
recomputed.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import math
import unittest

from sglang.srt.multimodal.processors.minimax_m3_vl import (
    MAX_TOTAL_PIXELS_IMAGE,
    MIN_SHORT_SIDE_PIXEL,
    check_total_pixels,
    resolve_media_size,
)
from sglang.srt.utils import round_up
from sglang.test.test_utils import CustomTestCase

PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE  # 28


def _smart_resize(height, width, max_pixels, factor=FACTOR, min_pixels=4 * 28 * 28):
    """A copy of the M3 HF image processor's resize, needed to predict tokens.

    It lives in the model repo's remote code (``image_processor.py``), so this is
    a model of a dependency, not of anything in this repo: if MiniMax changes it,
    these token counts go stale rather than red. Re-derive them against the
    checkpoint when bumping the model.
    """
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _image_tokens(width, height, cap):
    """Merged vision tokens for one image through the full sizing pipeline."""
    width, height = resolve_media_size(width, height, cap)
    if cap is not None:
        check_total_pixels(width, height, 1, MAX_TOTAL_PIXELS_IMAGE)
    max_pixels = cap**2 if cap is not None else 451584
    h, w = _smart_resize(height, width, max_pixels)
    return (h // PATCH_SIZE) * (w // PATCH_SIZE) // (MERGE_SIZE**2)


class TestResolveMediaSize(CustomTestCase):
    def test_long_side_capped(self):
        self.assertEqual(resolve_media_size(5000, 3000, 1008), (1008, 605))
        self.assertEqual(resolve_media_size(3000, 5000, 1008), (605, 1008))

    def test_within_cap_is_untouched(self):
        self.assertEqual(resolve_media_size(500, 400, 1008), (500, 400))
        self.assertEqual(resolve_media_size(500, 400, None), (500, 400))

    def test_short_side_floor(self):
        # A cap only shrinks; a short side under the floor is scaled up instead.
        self.assertEqual(resolve_media_size(1, 1, None), (112, 112))
        self.assertEqual(resolve_media_size(200, 50, None), (448, 112))
        self.assertEqual(min(resolve_media_size(64, 64, 1008)), MIN_SHORT_SIDE_PIXEL)

    def test_cap_wins_over_floor(self):
        # Long side over the cap: downscale, and do not then re-upscale.
        self.assertEqual(resolve_media_size(4000, 100, 1008), (1008, 25))

    def test_never_degenerate(self):
        for width, height in [(1, 1), (10000, 3), (3, 10000)]:
            new_w, new_h = resolve_media_size(width, height, 252)
            self.assertGreaterEqual(new_w, 1)
            self.assertGreaterEqual(new_h, 1)


class TestTotalPixelCeiling(CustomTestCase):
    def test_at_and_over_the_limit(self):
        # 3584 x 3584 == MAX_TOTAL_PIXELS_IMAGE exactly, so 3584 is the largest
        # square cap a caller may request.
        check_total_pixels(3584, 3584, 1, MAX_TOTAL_PIXELS_IMAGE)
        with self.assertRaises(ValueError):
            check_total_pixels(3612, 3612, 1, MAX_TOTAL_PIXELS_IMAGE)

    def test_frames_count_toward_the_limit(self):
        check_total_pixels(1008, 588, 500, 301_056_000)
        with self.assertRaises(ValueError):
            check_total_pixels(1008, 588, 600, 301_056_000)


class TestTierTokenMonotonicity(CustomTestCase):
    """The conformance suite requires prompt_tokens to grow with the cap."""

    def test_image_tiers_strictly_monotonic(self):
        tokens = {cap: _image_tokens(5000, 3000, cap) for cap in (252, 504, 1008)}
        self.assertEqual(tokens, {252: 45, 504: 198, 1008: 792})
        self.assertLess(tokens[252], tokens[504])
        self.assertLess(tokens[504], tokens[1008])

    def test_uncapped_images_are_still_accepted(self):
        # Regression: applying the total-pixel ceiling to the decoded size made
        # every image above 12.8M pixels a 400, including the 5000x3000 the
        # conformance suite requires to return 200. The ceiling bounds the size
        # the caller asked for, never the size that arrived.
        for width, height in [
            (500, 400),
            (2000, 1000),
            (3000, 2000),
            (5000, 3000),
            (4000, 4000),
            (4000, 500),
        ]:
            self.assertGreater(_image_tokens(width, height, None), 0)


class TestVideoFrameCap(CustomTestCase):
    """The video path caps the long side through get_hw_multiple_of's int mode."""

    def test_frame_cap_is_the_binding_constraint(self):
        from sglang.srt.multimodal.processors.minimax_m3_vl import get_hw_multiple_of

        sizes = {
            cap: get_hw_multiple_of((3840, 2160), FACTOR, cap)
            for cap in (504, 1008, 2016)
        }
        for cap, (width, height) in sizes.items():
            self.assertLessEqual(max(width, height), round_up(cap, FACTOR))
            self.assertEqual(width % FACTOR, 0)
            self.assertEqual(height % FACTOR, 0)
        areas = [sizes[cap][0] * sizes[cap][1] for cap in (504, 1008, 2016)]
        self.assertEqual(areas, sorted(areas))
        self.assertLess(areas[0], areas[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
