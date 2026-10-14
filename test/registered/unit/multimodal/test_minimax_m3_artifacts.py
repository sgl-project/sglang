"""CPU regressions for MiniMax M3's prompt-independent image cache."""

import asyncio
import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
from PIL import Image

from sglang.srt.multimodal.cache import (
    MultimodalPreprocessCache,
    estimate_cache_size_bytes,
)
from sglang.srt.multimodal.processors.minimax_m3_vl import MiniMaxM3VLProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _ImageProcessor:
    merge_size = 2

    def __init__(self):
        self.calls = []
        self.scale = 1

    def to_dict(self):
        return {"merge_size": self.merge_size, "scale": self.scale}

    def __call__(self, images, **kwargs):
        self.calls.append(len(images))
        rows = []
        grids = []
        for image in images:
            value = image.getpixel((0, 0))[0] * self.scale
            grid = [1, 2, 2 * (image.width // 28)]
            grids.append(grid)
            rows.append(torch.full((grid[1] * grid[2], 3), float(value)))
        return {"pixel_values": torch.cat(rows), "image_grid_thw": torch.tensor(grids)}


class _HFProcessor:
    """Small deterministic fixture with HF's grid-dependent token expansion."""

    def __init__(self):
        self.image_processor = _ImageProcessor()

    def __call__(self, text, images, **kwargs):
        if not all(isinstance(image, Image.Image) for image in images):
            raise ValueError("Expected valid images for HF input layout")
        if hasattr(self.image_processor, "fetch_images"):
            images = self.image_processor.fetch_images(images)
        data = self.image_processor(images, **kwargs)
        grids = iter(data["image_grid_thw"])
        ids = []
        for part in re.split(r"(\]<\]image\[>\[)", text):
            if part == MiniMaxM3VLProcessor.IMAGE_TOKEN:
                count = int(next(grids).prod()) // self.image_processor.merge_size**2
                ids += [98] + [99] * count + [100]
            else:
                ids += [ord(c) for c in part]
        return {**data, "input_ids": torch.tensor([ids])}


def _processor():
    processor = object.__new__(MiniMaxM3VLProcessor)
    processor._processor = _HFProcessor()
    processor.image_config = {}
    processor.gpu_image_decode = False
    processor.image_processor_backend = "pil"
    processor.mm_feature_transport = "cpu"
    processor.video_config = {}
    processor.audio_config = {}
    processor.processor_fingerprint = "config-v1"
    processor.mm_preprocess_cache = MultimodalPreprocessCache(1024 * 1024)
    processor.trust_mm_content_hashes = False
    processor.mm_processor_executor = None
    processor.io_executor = ThreadPoolExecutor(max_workers=2)
    processor.IM_TOKEN_ID = 99
    processor.IM_START_TOKEN_ID = 98
    processor.IM_END_TOKEN_ID = 100
    processor.VIDEO_TOKEN_ID = 101
    processor.mm_tokens = SimpleNamespace(
        image_token_regex=re.compile(r"<image>|\]<\]image\[>\[")
    )
    owner = processor
    processor.process_mm_data = lambda input_text, images, processor=None: (
        processor or owner._processor
    )(text=input_text, images=images)
    processor._finalize_mm_items = lambda items, images: items
    processor._is_preprocessed_input = lambda item: isinstance(item, dict)
    return processor


class TestMiniMaxM3Artifacts(unittest.TestCase):
    def setUp(self):
        self.processor = _processor()
        self.red = Image.new("RGB", (28, 28), (10, 0, 0))
        self.blue = Image.new("RGB", (56, 28), (20, 0, 0))
        self.green = Image.new("RGB", (28, 28), (30, 0, 0))

    def tearDown(self):
        self.processor.io_executor.shutdown()

    def test_cold_hot_partial_hits_and_order(self):
        async def scenario():
            first = await self.processor.prepare_media_artifacts([self.red, self.blue])
            hot = await self.processor.prepare_media_artifacts([self.blue, self.red])
            partial = await self.processor.prepare_media_artifacts(
                [self.green, self.red]
            )
            return first, hot, partial

        first, hot, partial = asyncio.run(scenario())
        self.assertEqual(self.processor._processor.image_processor.calls, [2, 1])
        self.assertIs(first[0], hot[1])
        self.assertIs(first[0], partial[1])
        self.assertEqual(partial[0].feature[0, 0].item(), 30)
        self.assertNotEqual(
            first[0].feature.untyped_storage().data_ptr(),
            first[1].feature.untyped_storage().data_ptr(),
        )
        self.assertGreaterEqual(
            estimate_cache_size_bytes(first[0]),
            first[0].feature.numel() * first[0].feature.element_size(),
        )

    def test_prompt_reconstruction_and_no_feature_copy(self):
        artifacts = asyncio.run(
            self.processor.prepare_media_artifacts([self.red, self.blue])
        )
        image = MiniMaxM3VLProcessor.IMAGE_TOKEN
        prompt = f"A{image}B{image}"
        expected = self.processor._processor(text=prompt, images=[self.red, self.blue])
        calls = len(self.processor._processor.image_processor.calls)
        output = self.processor.compose_image_artifacts(prompt, artifacts)
        changed = self.processor.compose_image_artifacts(f"longer {prompt}", artifacts)
        self.assertEqual(output.input_ids, expected["input_ids"].flatten().tolist())
        self.assertNotEqual(output.input_ids, changed.input_ids)
        self.assertNotEqual(output.mm_items[0].offsets, changed.mm_items[0].offsets)
        self.assertIs(output.mm_items[0].feature, artifacts[0].feature)
        start, end = output.mm_items[1].offsets[0]
        self.assertEqual(end - start + 1, 2)
        self.assertEqual(len(self.processor._processor.image_processor.calls), calls)
        self.assertIsInstance(
            self.processor._processor.image_processor, _ImageProcessor
        )

    def test_processor_identity_separates_cache_entries(self):
        first = asyncio.run(self.processor.prepare_media_artifacts([self.red]))[0]
        before = self.processor.preprocess_fingerprint_payload()
        self.processor._processor.image_processor.scale = 2
        after = self.processor.preprocess_fingerprint_payload()
        self.assertEqual(before["minimax_image_processor"]["scale"], 1)
        self.assertEqual(after["minimax_image_processor"]["scale"], 2)
        self.assertNotEqual(before, after)
        self.processor.processor_fingerprint = "config-v2"
        second = asyncio.run(self.processor.prepare_media_artifacts([self.red]))[0]
        self.assertNotEqual(first.artifact_key, second.artifact_key)
        self.assertEqual(second.feature[0, 0].item(), 20)

    def test_non_rgb_decode_matches_legacy_loading(self):
        self.processor.image_config = {"do_convert_rgb": False}
        for mode, color in [("RGBA", (10, 20, 30, 40)), ("L", 10)]:
            image = Image.new(mode, (28, 28), color)
            artifact = asyncio.run(self.processor.prepare_media_artifacts([image]))[0]
            expected = self.processor._processor.image_processor(
                [image.convert("RGB")], do_convert_rgb=False
            )["pixel_values"]
            self.assertTrue(torch.equal(artifact.feature, expected))

    def test_missing_placeholder_falls_back(self):
        class LegacyPath(Exception):
            pass

        async def legacy(**kwargs):
            raise LegacyPath

        self.processor.load_mm_data = legacy
        request = SimpleNamespace(video_data=None, mm_content_hashes=None)
        with self.assertRaises(LegacyPath):
            asyncio.run(
                self.processor.process_mm_data_async(
                    image_data=[self.red],
                    audio_data=None,
                    input_text="no placeholder",
                    request_obj=request,
                )
            )

    def test_image_config_return_tensors_has_no_duplicate_keyword(self):
        self.processor.image_config = {"return_tensors": "np"}
        artifact = asyncio.run(self.processor.prepare_media_artifacts([self.red]))[0]
        self.assertTrue(isinstance(artifact.feature, torch.Tensor))

    def test_non_cpu_tensor_falls_back(self):
        class LegacyPath(Exception):
            pass

        async def legacy(**kwargs):
            raise LegacyPath

        self.processor.load_mm_data = legacy
        request = SimpleNamespace(video_data=None, mm_content_hashes=None)
        with self.assertRaises(LegacyPath):
            asyncio.run(
                self.processor.process_mm_data_async(
                    image_data=[torch.empty((3, 28, 28), device="meta")],
                    audio_data=None,
                    input_text="<image>",
                    request_obj=request,
                )
            )

    def test_mixed_video_falls_back(self):
        class LegacyPath(Exception):
            pass

        async def legacy(**kwargs):
            raise LegacyPath

        self.processor.load_mm_data = legacy
        request = SimpleNamespace(video_data=["video"], mm_content_hashes=None)
        with self.assertRaises(LegacyPath):
            asyncio.run(
                self.processor.process_mm_data_async(
                    image_data=[self.red],
                    audio_data=None,
                    input_text="<image>",
                    request_obj=request,
                )
            )
        self.assertEqual(self.processor._processor.image_processor.calls, [])


if __name__ == "__main__":
    unittest.main()
