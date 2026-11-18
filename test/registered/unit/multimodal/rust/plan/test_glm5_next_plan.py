import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from transformers.image_processing_backends import TorchvisionBackend
from transformers.models.glm5_next.image_processing_glm5_next import (
    Glm5NextImageProcessor,
)

from sglang.srt.multimodal.media_io import smart_to_rgb
from sglang.srt.multimodal.processors.glm4v import _vendor_smart_resize_canvas
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import load_core  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

CORE = load_core()
PROCESSOR = Glm5NextImageProcessor()
IMAGE_TOKEN_ID = 59280
EXIF_ORIENTATION_TAG = 0x0112
CONFIGS = (
    dict(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        patch_expand_factor=1,
        min_image_tokens=16,
        max_image_tokens=8000,
    ),
    dict(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        patch_expand_factor=4,
        min_image_tokens=16,
        max_image_tokens=500,
    ),
    dict(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=1,
        patch_expand_factor=1,
        min_image_tokens=256,
        max_image_tokens=256,
    ),
    dict(
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        patch_expand_factor=1,
        min_image_tokens=4,
        max_image_tokens=1024,
    ),
)
EDGE_SIZES = [
    (1, 1),
    (1, 2),
    (2, 1),
    (27, 27),
    (28, 28),
    (29, 29),
    (84, 112),
    (111, 112),
    (112, 112),
    (112, 113),
    (113, 112),
    (447, 449),
    (448, 448),
    (449, 447),
    (895, 897),
    (896, 896),
    (897, 895),
    (2799, 2241),
    (2800, 2240),
    (2800, 2241),
    (2801, 2240),
    (4000, 27),
    (27, 4000),
    (1, 100000),
    (100000, 1),
    (3, 65535),
    (8192, 8192),
    (30000, 30000),
    (65535, 65535),
]


def _ones(*shape):
    return torch.ones(1, dtype=torch.uint8).expand(*shape)


def _resize_without_resampling(self, image, size, **kwargs):
    return _ones(*image.shape[:-2], size.height, size.width)


def reference_plan(
    height,
    width,
    *,
    patch_size,
    merge_size,
    temporal_patch_size,
    patch_expand_factor,
    min_image_tokens,
    max_image_tokens,
):
    canvas = PROCESSOR.resize(
        images=_ones(1, 1, height, width),
        resample=None,
        factor=patch_size * merge_size * patch_expand_factor,
        temporal_factor=temporal_patch_size,
        min_image_tokens=min_image_tokens,
        max_image_tokens=max_image_tokens,
    )
    canvas_height, canvas_width = canvas.shape[-2:]
    content_height = int(canvas[0, 0, :, 0].count_nonzero())
    content_width = int(canvas[0, 0, 0, :].count_nonzero())
    grid = (1, canvas_height // patch_size, canvas_width // patch_size)
    tokens = grid[1] * grid[2] // merge_size**2
    return (
        canvas_height,
        canvas_width,
        content_height,
        content_width,
        grid,
        [IMAGE_TOKEN_ID] * tokens,
    )


def sglang_canvas(
    height,
    width,
    *,
    patch_size,
    merge_size,
    temporal_patch_size,
    patch_expand_factor,
    min_image_tokens,
    max_image_tokens,
):
    factor = patch_size * merge_size * patch_expand_factor
    pixels_per_token = temporal_patch_size * factor**2
    return _vendor_smart_resize_canvas(
        temporal_patch_size,
        height,
        width,
        temporal_factor=temporal_patch_size,
        factor=factor,
        min_pixels=min_image_tokens * pixels_per_token,
        max_pixels=max_image_tokens * pixels_per_token,
    )


def rust_plan(height, width, orientation=1, **config):
    return CORE.plan.glm5_next_image(
        height,
        width,
        orientation=orientation,
        image_token_id=IMAGE_TOKEN_ID,
        **config,
    )


def sample_sizes():
    rng = random.Random(0)
    sizes = list(EDGE_SIZES)
    sizes += [(rng.randint(1, 4096), rng.randint(1, 4096)) for _ in range(400)]
    sizes += [(rng.randint(1, 64), rng.randint(1, 20000)) for _ in range(60)]
    sizes += [(rng.randint(1, 20000), rng.randint(1, 64)) for _ in range(60)]
    for factor in (28, 32, 112):
        sizes += [
            (factor * rng.randint(1, 100), factor * rng.randint(1, 100))
            for _ in range(30)
        ]
    return sizes


@unittest.skipUnless(CORE, "sglang-mm extension not built")
class TestRustGlm5NextImagePlan(CustomTestCase):
    def test_plan_matches_hf_image_processor_resize(self):
        sizes = sample_sizes()
        with patch.object(TorchvisionBackend, "resize", _resize_without_resampling):
            for config in CONFIGS:
                for height, width in sizes:
                    with self.subTest(height=height, width=width, **config):
                        self.assertEqual(
                            rust_plan(height, width, **config),
                            reference_plan(height, width, **config),
                        )

    def test_canvas_matches_sglang_vendor_replica(self):
        for config in CONFIGS:
            for height, width in sample_sizes():
                with self.subTest(height=height, width=width, **config):
                    self.assertEqual(
                        rust_plan(height, width, **config)[:2],
                        sglang_canvas(height, width, **config),
                    )

    def test_orientation_matches_smart_to_rgb_transpose(self):
        config = CONFIGS[0]
        with patch.object(TorchvisionBackend, "resize", _resize_without_resampling):
            for height, width in ((100, 300), (84, 112), (33, 500)):
                for orientation in range(1, 9):
                    image = Image.new("RGB", (width, height))
                    image.getexif()[EXIF_ORIENTATION_TAG] = orientation
                    upright = smart_to_rgb(image)
                    with self.subTest(
                        height=height, width=width, orientation=orientation
                    ):
                        self.assertEqual(
                            rust_plan(height, width, orientation, **config),
                            reference_plan(upright.height, upright.width, **config),
                        )


if __name__ == "__main__":
    unittest.main()
