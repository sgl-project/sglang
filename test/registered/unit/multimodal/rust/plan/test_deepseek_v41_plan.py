import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from sglang.srt.multimodal.deepseek_v41_image_processing import (
    image_token_types,
    patchify_image_rust,
    plan_image_grid,
    prepare_image,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import load_core  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

CORE = load_core()
IMAGE_TOKEN_ID = 129264
CONFIGS = (
    dict(
        patch_size=14,
        downsample_ratio=3,
        max_n_token=1024,
        min_pixels=295936,
        max_wh_ratio=None,
    ),
    dict(
        patch_size=14,
        downsample_ratio=3,
        max_n_token=1024,
        min_pixels=295936,
        max_wh_ratio=2,
    ),
    dict(
        patch_size=16,
        downsample_ratio=4,
        max_n_token=256,
        min_pixels=65536,
        max_wh_ratio=None,
    ),
    dict(
        patch_size=14,
        downsample_ratio=3,
        max_n_token=576,
        min_pixels=0,
        max_wh_ratio=5,
    ),
)
EDGE_SIZES = [
    (1, 1),
    (1, 2),
    (2, 1),
    (13, 13),
    (14, 14),
    (15, 15),
    (27, 29),
    (28, 28),
    (42, 42),
    (43, 41),
    (255, 257),
    (256, 256),
    (257, 255),
    (543, 545),
    (544, 543),
    (544, 544),
    (544, 545),
    (545, 543),
    (546, 546),
    (547, 547),
    (1, 295935),
    (1, 295936),
    (1, 295937),
    (295936, 1),
    (25, 1024),
    (1024, 25),
    (16, 49),
    (32, 98),
    (80, 245),
    (6, 8),
    (49, 112),
    (51, 112),
    (100, 100),
    (100, 199),
    (100, 200),
    (100, 201),
    (100, 499),
    (100, 500),
    (100, 501),
    (588, 3024),
    (589, 3024),
    (588, 3025),
    (3024, 588),
    (3024, 589),
    (1301, 1301),
    (1302, 1302),
    (1303, 1303),
    (1302, 1303),
    (1036, 1036),
    (3024, 4032),
    (4032, 3024),
    (10, 10200),
    (10, 10210),
    (10, 10220),
    (5100, 10),
    (5110, 10),
    (5120, 10),
    (2, 30000),
    (30000, 2),
    (1, 65535),
    (65535, 1),
    (1, 100000),
    (100000, 1),
    (3, 200000),
]
LARGE_SIZES = [
    (8192, 8192),
    (8191, 8193),
    (16384, 4096),
    (4096, 16384),
    (65535, 65535),
    (65535, 4000),
    (4000, 65535),
]


def python_args(config):
    return SimpleNamespace(
        vision_patch_size=config["patch_size"],
        vision_downsample_ratio=config["downsample_ratio"],
        vision_max_n_token=config["max_n_token"],
        vision_min_pixels=config["min_pixels"],
        vision_max_wh_ratio=config["max_wh_ratio"],
    )


def reference_plan(height, width, config):
    args = python_args(config)
    image = Image.new("RGB", (width, height))
    _, plan, llm_h, llm_w = prepare_image(image, args)
    if 0 in (plan["resize_h"], plan["resize_w"]):
        try:
            patchify_image_rust(image, args, resize_patchify=CORE.dsv41.resize_patchify)
        except ValueError:
            return "ValueError"
    patch = plan["patch_size"]
    return (
        plan["resize_h"],
        plan["resize_w"],
        (1, plan["height"] // patch, plan["width"] // patch),
        [IMAGE_TOKEN_ID] * len(image_token_types(llm_h, llm_w)),
        (plan["height"], plan["width"]),
        (plan["top"], plan["left"]),
        patch,
        (llm_h, llm_w),
    )


def reference_grid(height, width, config):
    patch = config["patch_size"]
    llm_h, llm_w, best_height, best_width = plan_image_grid(
        width, height, python_args(config)
    )
    return (
        (1, best_height // patch, best_width // patch),
        [IMAGE_TOKEN_ID] * len(image_token_types(llm_h, llm_w)),
        (best_height, best_width),
        patch,
        (llm_h, llm_w),
    )


def rust_plan(height, width, config):
    try:
        return CORE.plan.deepseek_v41_image(
            height, width, image_token_id=IMAGE_TOKEN_ID, **config
        )
    except ValueError:
        return "ValueError"


@unittest.skipUnless(CORE, "sglang-mm extension not built")
class TestRustDeepseekV41ImagePlan(CustomTestCase):
    def test_plan_matches_python_prepare_image(self):
        rng = random.Random(0)
        sizes = list(EDGE_SIZES)
        sizes += [(rng.randint(1, 2048), rng.randint(1, 2048)) for _ in range(300)]
        sizes += [(rng.randint(1, 600), rng.randint(1, 600)) for _ in range(100)]
        sizes += [(rng.randint(1, 64), rng.randint(1, 9000)) for _ in range(100)]
        sizes += [(rng.randint(1, 9000), rng.randint(1, 64)) for _ in range(100)]
        for config in CONFIGS:
            for height, width in sizes:
                with self.subTest(height=height, width=width, **config):
                    self.assertEqual(
                        rust_plan(height, width, config),
                        reference_plan(height, width, config),
                    )

    def test_large_image_grid_matches_python_plan_image_grid(self):
        rng = random.Random(1)
        sizes = list(LARGE_SIZES)
        sizes += [
            (rng.randint(2049, 20000), rng.randint(2049, 20000)) for _ in range(200)
        ]
        for config in CONFIGS:
            for height, width in sizes:
                with self.subTest(height=height, width=width, **config):
                    plan = CORE.plan.deepseek_v41_image(
                        height, width, image_token_id=IMAGE_TOKEN_ID, **config
                    )
                    self.assertEqual(
                        (plan[2], plan[3], plan[4], plan[6], plan[7]),
                        reference_grid(height, width, config),
                    )

    def test_zero_size_is_a_value_error(self):
        for height, width in ((0, 10), (10, 0), (0, 0)):
            with self.subTest(height=height, width=width):
                self.assertEqual(rust_plan(height, width, CONFIGS[0]), "ValueError")


if __name__ == "__main__":
    unittest.main()
