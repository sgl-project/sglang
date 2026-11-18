import random
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from sglang.srt.multimodal.processors.minimax_m3_vl import (
    DEFAULT_IMAGE_LONG_SIDE,
    MAX_IMAGE_PIXELS,
    MiniMaxM3VLProcessor,
    _check_max_long_side_pixel,
    _compute_aligned_size,
    resize_images_for_request,
    smart_resize,
)
from sglang.srt.multimodal.processors.minimax_m31_media import MiniMaxM31ImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import load_core  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

CORE = load_core()
PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE
TOKEN_IDS = {
    MiniMaxM3VLProcessor.IMAGE_TOKEN: 10,
    MiniMaxM3VLProcessor.IMAGE_START_TOKEN: 12,
    MiniMaxM3VLProcessor.IMAGE_END_TOKEN: 13,
}
MEDIA_TOKEN = re.compile(r"\]<\].*?\[>\[")
LONG_SIDES = (None, 112, 140, 252, 476, 1008, 2016, 2044, 3556, 3584)
BAD_LONG_SIDES = (-28, 0, 28, 84, 111, 113, 500, 2017, 3585, 3612, 4032)
MAX_PIXELS = (None, 3136, 200704, 451584, MAX_IMAGE_PIXELS)
EDGE_SIZES = (
    (1, 1),
    (1, 200),
    (1, 201),
    (200, 1),
    (201, 1),
    (5, 1000),
    (5, 1001),
    (10, 2000),
    (10, 2001),
    (18, 3584),
    (17, 3584),
    (13, 14),
    (14, 14),
    (27, 28),
    (28, 28),
    (42, 42),
    (50, 100),
    (50, 1000),
    (50, 2000),
    (70, 70),
    (111, 111),
    (111, 112),
    (112, 112),
    (113, 113),
    (126, 126),
    (154, 154),
    (111, 2016),
    (112, 2016),
    (112, 2017),
    (345, 1000),
    (1007, 1008),
    (1008, 1008),
    (1009, 1008),
    (1080, 1920),
    (2015, 2015),
    (2016, 2016),
    (2017, 2017),
    (2016, 112),
    (2030, 2030),
    (2058, 2058),
    (3583, 3583),
    (3584, 3584),
    (3585, 3584),
    (3584, 18),
    (4000, 3000),
    (8192, 8192),
    (27, 5400),
    (27, 5401),
    (60000, 300),
    (60000, 299),
    (65535, 65535),
)


def placeholder_processor():
    processor = object.__new__(MiniMaxM3VLProcessor)
    processor._tokenizer = None
    processor._processor = SimpleNamespace(
        image_processor=SimpleNamespace(merge_size=MERGE_SIZE)
    )
    return processor


PROCESSOR = placeholder_processor()


def reference_expansion(grid):
    text = PROCESSOR._image_placeholder_text(grid)
    return [TOKEN_IDS[token] for token in MEDIA_TOKEN.findall(text)]


def reference_size(height, width, *, max_pixels, max_long_side_pixel):
    limit = _check_max_long_side_pixel(max_long_side_pixel, FACTOR)
    if limit is None and max_pixels is not None:
        resized_height, resized_width = smart_resize(
            height=height, width=width, factor=FACTOR, max_pixels=max_pixels
        )
        if resized_height == 0 or resized_width == 0:
            raise ValueError("degenerate resize")
        return resized_height, resized_width
    resized_height, resized_width = _compute_aligned_size(
        height, width, FACTOR, DEFAULT_IMAGE_LONG_SIDE if limit is None else limit
    )
    if resized_height * resized_width > MAX_IMAGE_PIXELS:
        raise ValueError("image pixels exceed the limit")
    return resized_height, resized_width


def reference_plan(height, width, **options):
    try:
        resized_height, resized_width = reference_size(height, width, **options)
    except ValueError:
        return "ValueError"
    grid = (1, resized_height // PATCH_SIZE, resized_width // PATCH_SIZE)
    return resized_height, resized_width, grid, reference_expansion(grid)


def rust_plan(height, width, **options):
    try:
        return CORE.plan.minimax_m3_image(
            height,
            width,
            image_token_id=TOKEN_IDS[MiniMaxM3VLProcessor.IMAGE_TOKEN],
            image_start_token_id=TOKEN_IDS[MiniMaxM3VLProcessor.IMAGE_START_TOKEN],
            image_end_token_id=TOKEN_IDS[MiniMaxM3VLProcessor.IMAGE_END_TOKEN],
            patch_size=PATCH_SIZE,
            merge_size=MERGE_SIZE,
            **options,
        )
    except ValueError:
        return "ValueError"


def processor_plan(height, width, *, image_processor, max_pixels, max_long_side_pixel):
    item = {"image": Image.new("RGB", (width, height))}
    if max_long_side_pixel is not None:
        item["max_long_side_pixel"] = max_long_side_pixel
    try:
        resized = resize_images_for_request(
            [item["image"]],
            [item],
            image_processor=image_processor,
            max_pixels=max_pixels,
        )
    except ValueError:
        return "ValueError"
    output = image_processor(images=resized, return_tensors="pt", do_resize=False)
    grid = tuple(output["image_grid_thw"][0].tolist())
    resized_width, resized_height = resized[0].size
    return resized_height, resized_width, grid, reference_expansion(grid)


@unittest.skipUnless(CORE, "sglang-mm extension not built")
class TestRustMiniMaxM3ImagePlan(CustomTestCase):
    def assert_same_plan(self, height, width, **options):
        with self.subTest(height=height, width=width, **options):
            self.assertEqual(
                rust_plan(height, width, **options),
                reference_plan(height, width, **options),
            )

    def test_edge_sizes_match_python_for_every_option(self):
        for height, width in EDGE_SIZES:
            for max_pixels in MAX_PIXELS:
                for max_long_side_pixel in LONG_SIDES + BAD_LONG_SIDES:
                    self.assert_same_plan(
                        height,
                        width,
                        max_pixels=max_pixels,
                        max_long_side_pixel=max_long_side_pixel,
                    )

    def test_random_sizes_match_python(self):
        rng = random.Random(0)
        sizes = [(rng.randint(1, 4096), rng.randint(1, 4096)) for _ in range(400)]
        sizes += [(rng.randint(1, 64), rng.randint(1, 9000)) for _ in range(100)]
        sizes += [(rng.randint(1, 9000), rng.randint(1, 64)) for _ in range(100)]
        sizes += [(rng.randint(84, 140), rng.randint(84, 140)) for _ in range(100)]
        sizes += [(FACTOR * rng.randint(1, 140), FACTOR) for _ in range(50)]
        sizes += [(rng.randint(1, 999), FACTOR * rng.randint(1, 99)) for _ in range(50)]
        for height, width in sizes:
            for max_pixels in (None, rng.choice(MAX_PIXELS[1:])):
                for max_long_side_pixel in (
                    None,
                    FACTOR * rng.randint(4, 128),
                    rng.randint(-100, 4000),
                ):
                    self.assert_same_plan(
                        height,
                        width,
                        max_pixels=max_pixels,
                        max_long_side_pixel=max_long_side_pixel,
                    )

    def test_plan_matches_the_image_processor_output(self):
        rng = random.Random(1)
        image_processor = MiniMaxM31ImageProcessor(max_size=(672, 672))
        sizes = [(50, 100), (112, 112), (126, 154), (3, 700), (345, 600), (640, 480)]
        sizes += [(rng.randint(8, 700), rng.randint(8, 700)) for _ in range(30)]
        for height, width in sizes:
            for max_pixels in (None, 200704):
                for max_long_side_pixel in (None, 112, 252, 100):
                    options = dict(
                        max_pixels=max_pixels, max_long_side_pixel=max_long_side_pixel
                    )
                    with self.subTest(height=height, width=width, **options):
                        self.assertEqual(
                            rust_plan(height, width, **options),
                            processor_plan(
                                height,
                                width,
                                image_processor=image_processor,
                                **options,
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
