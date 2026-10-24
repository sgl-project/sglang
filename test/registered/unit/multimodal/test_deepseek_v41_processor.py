"""Unit tests for the DeepSeek-V4.1 image processor."""

import asyncio
import io
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch
from PIL import Image

from sglang.srt.multimodal.cache import MultimodalPreprocessCache
from sglang.srt.multimodal.deepseek_v41_image_processing import patchify_image
from sglang.srt.multimodal.processors.deepseek_v41 import DeepseekV41ImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

IMAGE_TOKEN_ID = 9
HF_CONFIG = SimpleNamespace(
    vision_patch_size=14,
    vision_downsample_ratio=3,
    vision_max_n_token=1024,
    vision_min_pixels=0,
    vision_max_wh_ratio=None,
)


def _png(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (40, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def _request():
    return SimpleNamespace(mm_content_hashes=None)


class TestDeepseekV41Processor(CustomTestCase):
    def _make_processor(self):
        processor = object.__new__(DeepseekV41ImageProcessor)
        processor.hf_config = HF_CONFIG
        processor.image_token_id = IMAGE_TOKEN_ID
        processor.mm_tokens = None
        processor.preprocess_backend = "cpu"
        processor.cpu_patchify = Mock(wraps=patchify_image)
        processor.mm_feature_transport = "cpu"
        processor.use_cuda_ipc = False
        processor.processor_fingerprint = "processor"
        processor.trust_mm_content_hashes = False
        processor.mm_preprocess_cache = MultimodalPreprocessCache(64 * 1024 * 1024)
        processor.mm_processor_executor = None
        processor.io_executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(processor.io_executor.shutdown)
        return processor

    def test_cached_path_composes_the_same_request_as_uncached(self):
        """A preprocess-cache request must match the legacy path item for item."""
        images = [_png(84, 84), _png(126, 84)]
        ids = [1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID, 3]
        processor = self._make_processor()
        decoded = SimpleNamespace(images=[Image.open(io.BytesIO(i)) for i in images])
        cached = asyncio.run(processor.process_mm_data_async(images, ids, _request()))
        with patch.object(processor, "load_mm_data", AsyncMock(return_value=decoded)):
            uncached = asyncio.run(
                processor._process_mm_data_uncached(images, ids, _request())
            )

        self.assertEqual(cached.input_ids, uncached.input_ids)
        self.assertEqual(len(cached.mm_items), 2)
        for cached_item, uncached_item in zip(cached.mm_items, uncached.mm_items):
            self.assertEqual(cached_item.offsets, uncached_item.offsets)
            self.assertEqual(
                cached_item.model_specific_data, uncached_item.model_specific_data
            )
            torch.testing.assert_close(cached_item.feature, uncached_item.feature)

    def test_repeated_image_skips_patchify_and_is_placed_per_prompt(self):
        """A cache hit reuses the patches and re-places them in the new prompt."""
        image = _png(84, 84)
        processor = self._make_processor()
        first = asyncio.run(
            processor.process_mm_data_async([image], [1, IMAGE_TOKEN_ID, 2], _request())
        )
        second = asyncio.run(
            processor.process_mm_data_async(
                [image], [3, 4, 5, IMAGE_TOKEN_ID], _request()
            )
        )

        self.assertEqual(processor.cpu_patchify.call_count, 1)
        first_start, first_end = first.mm_items[0].offsets[0]
        second_start, second_end = second.mm_items[0].offsets[0]
        self.assertEqual((first_start, second_start), (1, 3))
        self.assertEqual(first_end - first_start, second_end - second_start)
        self.assertEqual(first.mm_items[0].hash, second.mm_items[0].hash)
        torch.testing.assert_close(
            first.mm_items[0].feature, second.mm_items[0].feature
        )


if __name__ == "__main__":
    unittest.main()
