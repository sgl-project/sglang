"""Dynamic-resolution budgets for images with a minimum-sized short side."""

import unittest

import torch
import torch.nn.functional as F
from PIL import Image

from sglang.srt.multimodal.internvl_utils import (
    compute_budgeted_image_sizes,
    compute_dynamic_image_size,
    resize_image_to_pixels,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDynamicImagePatchBudget(unittest.TestCase):
    def test_extreme_landscape_and_portrait_fit_the_patch_limit(self):
        for max_patches in (4, 16, 64, 256):
            for portrait in (False, True):
                with self.subTest(max_patches=max_patches, portrait=portrait):
                    size = (16, 16384) if portrait else (16384, 16)
                    result = compute_dynamic_image_size(*size, 16, 0.5, 4, max_patches)
                    long_side = (max_patches // 2) * 16
                    expected = (
                        (32, long_side, max_patches // 4)
                        if portrait
                        else (long_side, 32, max_patches // 4)
                    )
                    self.assertEqual(result, expected)

    def test_supported_downsampling_grids_keep_both_constraints(self):
        for ds in (1, 2, 4):
            for portrait in (False, True):
                with self.subTest(ds=ds, portrait=portrait):
                    size = (1, 16384) if portrait else (16384, 1)
                    width, height, tokens = compute_dynamic_image_size(
                        *size, 16, 1 / ds, ds * ds, 256
                    )
                    patches = (width // 16) * (height // 16)
                    self.assertLessEqual(patches, 256)
                    self.assertGreaterEqual(patches, ds * ds)
                    self.assertEqual(width % (16 * ds), 0)
                    self.assertEqual(height % (16 * ds), 0)
                    self.assertEqual(tokens, patches // (ds * ds))

    def test_non_grid_patch_limits_use_the_last_fitting_grid(self):
        for max_patches in (17, 63, 255):
            for portrait in (False, True):
                with self.subTest(max_patches=max_patches, portrait=portrait):
                    size = (16, 16384) if portrait else (16384, 16)
                    width, height, tokens = compute_dynamic_image_size(
                        *size, 16, 0.5, 4, max_patches
                    )
                    patches = (width // 16) * (height // 16)
                    self.assertEqual(patches, max_patches // 4 * 4)
                    self.assertEqual(min(width, height), 32)
                    self.assertEqual(tokens, patches // 4)

    def test_multiple_extreme_images_fit_the_prompt_token_budget(self):
        sizes = [(16384, 16), (16, 16384)]
        original_sizes = sizes.copy()
        results = compute_budgeted_image_sizes(sizes, 128, 16, 0.5, 4, 512)
        self.assertEqual(results, [(2048, 32, 64), (32, 2048, 64)])
        self.assertEqual(sum(tokens for _, _, tokens in results), 128)
        self.assertEqual(sizes, original_sizes)

    def test_real_resized_pixels_and_downsampled_grid_fit_the_budget(self):
        for mode, size in (("RGB", (16384, 1)), ("L", (1, 16384))):
            with self.subTest(mode=mode):
                image = Image.new(mode, size, color=127)
                original = image.tobytes()
                ((width, height, tokens),) = compute_budgeted_image_sizes(
                    [image.size], 64, 16, 0.5, 4, 256
                )
                pixels = resize_image_to_pixels(
                    image, width, height, mean=(0, 0, 0), std=(1, 1, 1)
                )
                self.assertEqual(pixels.shape, (1, 3, height, width))
                self.assertLessEqual(pixels.numel(), 3 * 16 * 16 * 256)
                patch_grid = pixels.unfold(2, 16, 16).unfold(3, 16, 16)
                patch_features = patch_grid.mean(dim=(-1, -2))
                shuffled = F.pixel_unshuffle(patch_features, 2)
                self.assertEqual(shuffled.shape[-2] * shuffled.shape[-1], tokens)
                self.assertLessEqual(tokens, 64)
                self.assertTrue(torch.isfinite(shuffled).all())
                self.assertEqual(image.size, size)
                self.assertEqual(image.mode, mode)
                self.assertEqual(image.tobytes(), original)

    def test_ordinary_dimensions_keep_existing_sizes(self):
        cases = (
            ((1024, 768, 4, 256), (288, 224, 63)),
            ((128, 64, 4, 256), (128, 64, 8)),
            ((16, 16, 64, 256), (128, 128, 16)),
        )
        for (width, height, minimum, maximum), expected in cases:
            with self.subTest(size=(width, height), minimum=minimum):
                actual = compute_dynamic_image_size(
                    width, height, 16, 0.5, minimum, maximum
                )
                self.assertEqual(actual, expected)
                transposed = compute_dynamic_image_size(
                    height, width, 16, 0.5, minimum, maximum
                )
                self.assertEqual(transposed, (expected[1], expected[0], expected[2]))

    def test_empty_and_unconstrained_prompt_controls(self):
        self.assertEqual(compute_budgeted_image_sizes([], 128, 16, 0.5, 4, 256), [])
        self.assertEqual(
            compute_budgeted_image_sizes([(128, 64), (64, 128)], 32, 16, 0.5, 4, 256),
            [(128, 64, 8), (64, 128, 8)],
        )


if __name__ == "__main__":
    unittest.main()
