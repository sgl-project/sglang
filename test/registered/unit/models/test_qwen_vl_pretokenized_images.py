"""Image-artifact prompt normalization for pre-tokenized Qwen-VL requests."""

import re
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwenVLPretokenizedImages(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        self.processor.IM_TOKEN_ID = 12
        self.processor.vision_start_token_id = 10
        self.processor.vision_end_token_id = 11
        self.processor._spatial_merge_size = 2
        self.processor.mm_tokens = SimpleNamespace(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>|<image>"
            ),
        )

    def test_expanded_single_image_round_trip(self):
        prompt = [1, 10, 12, 12, 12, 12, 11, 2]
        original = prompt.copy()
        normalized = self.processor._normalize_prompt_for_fast_path(prompt, 1)
        rebuilt, offsets, modalities = self.processor.build_input_ids(
            normalized, img_grid_thw=torch.tensor([[1, 4, 4]])
        )
        self.assertEqual(normalized, [1, 10, 12, 11, 2])
        self.assertEqual(rebuilt, original)
        self.assertEqual(offsets, [(2, 5)])
        self.assertEqual(len(modalities), 1)
        self.assertEqual(prompt, original)

    def test_two_images_keep_order_and_text_tokens(self):
        prompt = [1, 10, 12, 12, 12, 12, 11, 2, 10, 12, 12, 11, 3]
        normalized = self.processor._normalize_prompt_for_fast_path(prompt, 2)
        self.assertEqual(normalized, [1, 10, 12, 11, 2, 10, 12, 11, 3])
        rebuilt, offsets, modalities = self.processor.build_input_ids(
            normalized, img_grid_thw=torch.tensor([[1, 4, 4], [1, 2, 4]])
        )
        self.assertEqual(rebuilt, prompt)
        self.assertEqual(offsets, [(2, 5), (9, 10)])
        self.assertEqual(len(modalities), 2)

    def test_compact_tokens_are_unchanged(self):
        prompt = [1, 10, 12, 11, 2]
        self.assertEqual(
            self.processor._normalize_prompt_for_fast_path(prompt, 1), prompt
        )

    def test_malformed_boundaries_and_count_use_fallback(self):
        for prompt, count in (
            ([1, 12, 12, 2], 1),
            ([1, 10, 12, 12], 1),
            ([1, 10, 12, 12, 2], 1),
            ([1, 10, 12, 11, 2], 2),
            ([1, 2], 1),
        ):
            with self.subTest(prompt=prompt, count=count):
                self.assertIsNone(
                    self.processor._normalize_prompt_for_fast_path(prompt, count)
                )

    def test_string_and_legacy_image_placeholders(self):
        native = self.processor.mm_tokens.image_token
        expanded = "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>"
        self.assertEqual(
            self.processor._normalize_prompt_for_fast_path(
                f"before {expanded} between <image> after", 2
            ),
            f"before {native} between {native} after",
        )
        self.assertIsNone(self.processor._normalize_prompt_for_fast_path("<image>", 2))


if __name__ == "__main__":
    unittest.main()
