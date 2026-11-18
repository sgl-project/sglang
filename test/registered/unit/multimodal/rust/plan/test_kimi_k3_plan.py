import random
import sys
import unittest
from pathlib import Path

from sglang.srt.multimodal.processors.kimi_k3 import _expand_k3_image_prompt_token_ids
from sglang.srt.multimodal.processors.kimi_k25 import (
    _grid_thw_from_resize_config,
    navit_resize_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import load_core  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

CORE = load_core()
IMAGE_TOKEN_ID = 7
CHAR_ID_BASE = 1000
CONFIGS = (
    dict(
        patch_size=14,
        merge_kernel_size=2,
        in_patch_limit=16384,
        patch_limit_on_one_side=512,
        fixed_output_tokens=None,
    ),
    dict(
        patch_size=14,
        merge_kernel_size=2,
        in_patch_limit=4096,
        patch_limit_on_one_side=256,
        fixed_output_tokens=None,
    ),
    dict(
        patch_size=16,
        merge_kernel_size=3,
        in_patch_limit=1000,
        patch_limit_on_one_side=48,
        fixed_output_tokens=None,
    ),
    dict(
        patch_size=14,
        merge_kernel_size=2,
        in_patch_limit=16384,
        patch_limit_on_one_side=512,
        fixed_output_tokens=64,
    ),
)


class CharTokenizer:
    def __init__(self):
        self.texts = []

    def encode(self, text, allowed_special):
        if allowed_special != "all":
            raise AssertionError(allowed_special)
        self.texts.append(text)
        return [CHAR_ID_BASE + ord(char) for char in text]


def reference_plan(
    height,
    width,
    *,
    patch_size,
    merge_kernel_size,
    in_patch_limit,
    patch_limit_on_one_side,
    fixed_output_tokens,
):
    tokenizer = CharTokenizer()
    resize_config = navit_resize_config(
        width,
        height,
        patch_size,
        merge_kernel_size,
        in_patch_limit,
        patch_limit_on_one_side,
        fixed_output_tokens,
    )
    grid = _grid_thw_from_resize_config(resize_config, patch_size)
    expansion = (
        _expand_k3_image_prompt_token_ids(
            [IMAGE_TOKEN_ID],
            IMAGE_TOKEN_ID,
            [resize_config["num_tokens"]],
            [(width, height)],
            tokenizer,
        )
        .flatten()
        .tolist()
    )
    plan = (
        resize_config["new_height"],
        resize_config["new_width"],
        grid,
        expansion,
        (resize_config["pad_height"], resize_config["pad_width"]),
        (expansion.index(IMAGE_TOKEN_ID), resize_config["num_tokens"]),
    )
    return plan, tokenizer.texts


def rust_plan(height, width, **config):
    tokenizer = CharTokenizer()
    plan = CORE.plan.kimi_k3_image(
        height,
        width,
        lambda text: tokenizer.encode(text, allowed_special="all"),
        image_token_id=IMAGE_TOKEN_ID,
        **config,
    )
    return plan, tokenizer.texts


def edge_sizes():
    sizes = [(1, 1), (1, 2), (2, 1), (13, 13), (14, 14), (15, 15), (27, 28), (28, 28)]
    sizes += [(28, 29), (29, 28), (56, 56), (57, 57), (1, 9000), (9000, 1)]
    sizes += [(7168, 7168), (7169, 7168), (7168, 7169), (20000, 10), (10, 20000)]
    sizes += [(1792, 1792), (1793, 1792), (1792, 1793), (3000, 5000), (65535, 65535)]
    sizes += [(767, 768), (768, 768), (769, 768), (768, 2304), (2304, 768)]
    return sizes


@unittest.skipUnless(CORE, "sglang-mm extension not built")
class TestRustKimiK3ImagePlan(CustomTestCase):
    def test_plan_matches_python_processor(self):
        rng = random.Random(0)
        sizes = edge_sizes()
        sizes += [(rng.randint(1, 4096), rng.randint(1, 4096)) for _ in range(300)]
        sizes += [(rng.randint(1, 64), rng.randint(1, 9000)) for _ in range(60)]
        sizes += [(rng.randint(1, 9000), rng.randint(1, 64)) for _ in range(60)]
        sizes += [(rng.randint(4096, 20000), rng.randint(1, 20000)) for _ in range(60)]
        for config in CONFIGS:
            for height, width in sizes:
                with self.subTest(height=height, width=width, **config):
                    self.assertEqual(
                        rust_plan(height, width, **config),
                        reference_plan(height, width, **config),
                    )

    def test_placeholder_span_is_the_only_image_token_run(self):
        config = CONFIGS[0]
        plan, texts = rust_plan(100, 60, **config)
        expansion = plan[3]
        start, count = plan[5]
        self.assertEqual(
            texts, ["<|media_begin|>image 60x100<|media_content|>", "<|media_end|>"]
        )
        self.assertEqual(expansion[start : start + count], [IMAGE_TOKEN_ID] * count)
        self.assertEqual(expansion.count(IMAGE_TOKEN_ID), count)
        self.assertEqual(len(expansion), len(texts[0]) + count + len(texts[1]))

    def test_zero_size_is_a_value_error_in_rust_only(self):
        config = CONFIGS[0]
        for height, width in ((0, 10), (10, 0), (0, 0)):
            with self.subTest(height=height, width=width):
                with self.assertRaises(ValueError):
                    rust_plan(height, width, **config)
                with self.assertRaises(ZeroDivisionError):
                    reference_plan(height, width, **config)

    def test_encode_error_propagates(self):
        def encode(text):
            raise KeyError(text)

        with self.assertRaises(KeyError):
            CORE.plan.kimi_k3_image(
                10, 10, encode, image_token_id=IMAGE_TOKEN_ID, **CONFIGS[0]
            )


if __name__ == "__main__":
    unittest.main()
