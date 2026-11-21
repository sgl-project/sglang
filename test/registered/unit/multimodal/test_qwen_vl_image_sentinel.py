"""Qwen-VL image placeholder counting with the legacy ``<image>`` sentinel.

A chat-templated prompt already carries one native vision placeholder per image,
so a literal ``<image>`` in the question text (as in MMMU) is not an image slot.
A raw ``/generate`` prompt without native tokens still uses ``<image>`` as one.
"""

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImagePreprocessArtifact,
    QwenVLImageProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

NATIVE_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
CHAT_PROMPT = (
    "<|im_start|>user\nThis Roman portrait mummy <image> is from the 1st century AD."
    f"{NATIVE_IMAGE}Which technique was used?<|im_end|>\n<|im_start|>assistant\n"
)
RAW_PROMPT = "<image>\nDescribe the image."


class _Reached(Exception):
    pass


def _base_init(self, hf_config, *args, **kwargs):
    self.hf_config = hf_config


def _make_processor(model_type):
    hf_config = SimpleNamespace(
        model_type=model_type,
        vision_start_token_id=1,
        vision_end_token_id=2,
        image_token_id=3,
        video_token_id=4,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    hf_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(convert_ids_to_tokens=lambda ids: ["<|video_pad|>"])
    )
    with patch.object(BaseMultimodalProcessor, "__init__", _base_init):
        processor = QwenVLImageProcessor(hf_config, None, hf_processor)
    processor.mm_preprocess_cache = SimpleNamespace(enabled=False)
    processor.skip_tokenizer_init = False
    processor.video_config = {}
    return processor


def _request():
    return SimpleNamespace(video_data=None, audio_data=None, rid="rid")


class TestQwenVLImageSentinel(CustomTestCase):
    def _hf_processor_input(self, prompt):
        """Run the uncached path up to the HF processor call; return its input."""
        processor = _make_processor("qwen2_5_vl")
        processor.io_executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(processor.io_executor.shutdown)
        seen = {}

        async def capture(base_output, mm_tokens, **kwargs):
            seen["text"] = base_output.input_text
            seen["num_images"] = len(base_output.images)
            raise _Reached

        processor.process_and_combine_mm_data_async = capture
        image = Image.new("RGB", (32, 32))
        with self.assertRaises(_Reached):
            asyncio.run(processor.process_mm_data_async([image], prompt, _request()))
        return seen

    def _artifact_fast_path_input(self, prompt):
        """Run the artifact fast path up to prompt expansion; return its input."""
        processor = _make_processor("qwen3_vl")
        artifact = QwenVLImagePreprocessArtifact(
            content_digest="digest",
            artifact_key="key",
            feature_hash=0,
            feature=None,
            model_specific_data={"image_grid_thw": torch.tensor([[1, 4, 4]])},
        )

        async def prepare(image_data, content_hashes=None):
            return [artifact]

        async def uncached(*args, **kwargs):
            raise AssertionError("fast path fell back to full preprocessing")

        seen = {}

        def build(input_text, grid_key):
            seen["text"] = input_text
            raise _Reached

        processor.prepare_media_artifacts_without_cache = prepare
        processor._process_mm_data_uncached = uncached
        processor._build_image_prompt_template = build
        with self.assertRaises(_Reached):
            asyncio.run(processor.process_mm_data_async(["image"], prompt, _request()))
        return seen["text"]

    def test_literal_image_text_next_to_native_placeholder_is_one_image(self):
        """A literal <image> beside a native placeholder must not count as a second
        image: the uncached path used to fail with "An exception occurred while
        loading multimodal data" and the fast path used to be skipped."""
        seen = self._hf_processor_input(CHAT_PROMPT)
        self.assertEqual(seen["num_images"], 1)
        self.assertEqual(seen["text"], CHAT_PROMPT)

        self.assertEqual(self._artifact_fast_path_input(CHAT_PROMPT), CHAT_PROMPT)

    def test_legacy_image_sentinel_stays_on_artifact_fast_path(self):
        expected = RAW_PROMPT.replace("<image>", NATIVE_IMAGE)
        self.assertEqual(self._artifact_fast_path_input(RAW_PROMPT), expected)


if __name__ == "__main__":
    unittest.main()
