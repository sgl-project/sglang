"""CPU regressions for MiniMax M3's prompt-independent image cache."""

import asyncio
import re
import string
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
from PIL import Image
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.models.minimax_m3_vl import (
    MiniMaxM3VLImageProcessor,
)
from transformers.models.minimax_m3_vl import (
    MiniMaxM3VLProcessor as HFMiniMaxM3VLProcessor,
)
from transformers.models.minimax_m3_vl import (
    MiniMaxM3VLVideoProcessor,
)

from sglang.srt.multimodal.cache import (
    MultimodalPreprocessCache,
    estimate_cache_size_bytes,
)
from sglang.srt.multimodal.processors.minimax_m3_vl import MiniMaxM3VLProcessor
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
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


class _RecordingImageProcessor:
    def __init__(self, processor, outputs):
        self.processor = processor
        self.outputs = outputs
        self.merge_size = processor.merge_size

    def __call__(self, images, **kwargs):
        result = self.processor(images, **kwargs)
        self.outputs.append(result["pixel_values"])
        return result


class _Tokenizer:
    _SPECIAL = {
        MiniMaxM3VLProcessor.IMAGE_START_TOKEN: 98,
        MiniMaxM3VLProcessor.IMAGE_TOKEN: 99,
        MiniMaxM3VLProcessor.IMAGE_END_TOKEN: 100,
    }
    _PATTERN = re.compile("(" + "|".join(map(re.escape, _SPECIAL)) + ")")

    def __call__(self, texts, **kwargs):
        ids = []
        for part in self._PATTERN.split(texts[0]):
            ids += [self._SPECIAL[part]] if part in self._SPECIAL else map(ord, part)
        return {"input_ids": torch.tensor([ids])}


class _HFProcessor:
    """Small deterministic fixture with HF's grid-dependent token expansion."""

    def __init__(self):
        self.image_processor = _ImageProcessor()
        self.tokenizer = _Tokenizer()

    def __call__(self, text, images, **kwargs):
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
    processor._tokenizer = processor._processor.tokenizer
    processor._tokenizer_auto_adds_specials = False
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

    def test_single_image_miss_owns_storage_without_clone(self):
        outputs = []
        self.processor._processor.image_processor = _RecordingImageProcessor(
            self.processor._processor.image_processor, outputs
        )
        artifact = asyncio.run(self.processor.prepare_media_artifacts([self.red]))[0]
        feature = artifact.feature
        self.assertEqual(
            feature.untyped_storage().data_ptr(),
            outputs[0].untyped_storage().data_ptr(),
        )
        self.assertEqual(
            feature.untyped_storage().nbytes(),
            feature.numel() * feature.element_size(),
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

    def test_alias_placeholder_falls_back(self):
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
                    input_text="<image>",
                    request_obj=request,
                )
            )
        self.assertEqual(self.processor._processor.image_processor.calls, [])


def _real_hf_processor():
    # Character-level tokenizer with M3's multimodal special tokens; the image
    # processor and placeholder expansion are the real Transformers classes.
    vocab = {"<unk>": 0, "<pad>": 1}
    vocab.update({c: i + 2 for i, c in enumerate(string.printable)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>"
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                MiniMaxM3VLProcessor.IMAGE_TOKEN,
                MiniMaxM3VLProcessor.VIDEO_TOKEN,
                MiniMaxM3VLProcessor.IMAGE_START_TOKEN,
                MiniMaxM3VLProcessor.IMAGE_END_TOKEN,
            ]
        }
    )
    return HFMiniMaxM3VLProcessor(
        image_processor=MiniMaxM3VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=MiniMaxM3VLVideoProcessor(),
    )


class TestMiniMaxM3CacheParity(unittest.TestCase):
    """The cached path must match the legacy path on real HF processing."""

    def setUp(self):
        self.addCleanup(reset_context)
        hf_config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            to_dict=lambda: {"model_type": "minimax_m3_vl"},
        )
        self.processors = {}
        for name, cache_mb in [("legacy", 0), ("cached", 64)]:
            server_args = ServerArgs(
                model_path="dummy",
                device="cpu",
                mm_process_config={},
                mm_preprocess_cache_size_mb=cache_mb,
            )
            publish(server_args, role="test")
            self.processors[name] = MiniMaxM3VLProcessor(
                hf_config, server_args, _real_hf_processor(), "default"
            )
        self.assertFalse(self.processors["legacy"].mm_preprocess_cache.enabled)
        self.assertTrue(self.processors["cached"].mm_preprocess_cache.enabled)
        # Keep the worker pool on: it passes a per-thread processor clone.
        self.assertIsNotNone(self.processors["cached"].mm_processor_executor)

    def _run(self, name, images, prompt):
        request = SimpleNamespace(video_data=None, mm_content_hashes=None)
        return asyncio.run(
            self.processors[name].process_mm_data_async(
                image_data=images,
                audio_data=None,
                input_text=prompt,
                request_obj=request,
            )
        )

    def test_cold_and_hot_match_legacy(self):
        images = [
            Image.new("RGB", (56, 56), (10, 20, 30)),
            Image.new("RGB", (112, 56), (200, 0, 5)),
        ]
        image = MiniMaxM3VLProcessor.IMAGE_TOKEN
        prompt = f"A{image}B{image}C"
        legacy = self._run("legacy", images, prompt)
        for cached in (
            self._run("cached", images, prompt),
            self._run("cached", images, prompt),
        ):
            self.assertEqual(cached.input_ids, legacy.input_ids)
            self.assertEqual(len(cached.mm_items), len(legacy.mm_items))
            for got, expected in zip(cached.mm_items, legacy.mm_items):
                self.assertEqual(got.offsets, expected.offsets)
                self.assertTrue(torch.equal(got.feature, expected.feature))
                self.assertTrue(
                    torch.equal(
                        got.model_specific_data["image_grid_thw"],
                        expected.model_specific_data["image_grid_thw"],
                    )
                )


if __name__ == "__main__":
    unittest.main()
