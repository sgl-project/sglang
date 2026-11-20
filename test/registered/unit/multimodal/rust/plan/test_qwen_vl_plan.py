import random
import sys
import unittest
from pathlib import Path

from sglang.srt.multimodal.qwen_vl_media_processing import smart_resize
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import load_core  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CORE = load_core()
IMAGE_TOKEN_ID = 151655
CONFIGS = (
    dict(patch_size=14, merge_size=2, min_pixels=56 * 56, max_pixels=28 * 28 * 1280),
    dict(patch_size=16, merge_size=2, min_pixels=64 * 64, max_pixels=32 * 32 * 16384),
    dict(patch_size=14, merge_size=2, min_pixels=256 * 28 * 28, max_pixels=512 * 784),
)


def reference_plan(height, width, *, patch_size, merge_size, min_pixels, max_pixels):
    try:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    except ValueError:
        return "ValueError"
    if resized_height == 0 or resized_width == 0:
        return "ValueError"
    grid = (1, resized_height // patch_size, resized_width // patch_size)
    tokens = grid[1] * grid[2] // merge_size**2
    return resized_height, resized_width, grid, [IMAGE_TOKEN_ID] * tokens


def rust_plan(height, width, **config):
    try:
        return CORE.plan.qwen_vl_image(
            height, width, image_token_id=IMAGE_TOKEN_ID, **config
        )
    except ValueError:
        return "ValueError"


@unittest.skipUnless(CORE, "sglang-mm extension not built")
class TestRustQwenVlImagePlan(CustomTestCase):
    def test_plan_matches_python_smart_resize(self):
        rng = random.Random(0)
        sizes = [(1, 1), (1, 200), (1, 201), (14, 14), (27, 4000), (8192, 8192)]
        sizes += [(rng.randint(1, 4096), rng.randint(1, 4096)) for _ in range(400)]
        sizes += [(rng.randint(1, 64), rng.randint(1, 9000)) for _ in range(100)]
        for config in CONFIGS:
            for height, width in sizes:
                with self.subTest(height=height, width=width, **config):
                    self.assertEqual(
                        rust_plan(height, width, **config),
                        reference_plan(height, width, **config),
                    )


if __name__ == "__main__":
    unittest.main()
