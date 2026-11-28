import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from sglang.srt.multimodal.cache import (
    MultimodalPreprocessCache,
    resolve_multimodal_item_hash,
)
from sglang.srt.multimodal.processors.minicpmv4_6 import (
    MiniCPMV4_6ImageProcessor,
    MiniCPMV4_6MultimodalProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _image(width: int, height: int, offset: int = 0) -> Image.Image:
    data = (
        np.arange(width * height * 3, dtype=np.uint32).reshape(height, width, 3)
        + offset
    ) % 256
    return Image.fromarray(data.astype(np.uint8), mode="RGB")


def _palette_image(palette_offset: int) -> Image.Image:
    indices = (np.arange(32 * 32, dtype=np.uint32) % 251).astype(np.uint8)
    image = Image.fromarray(indices.reshape(32, 32), mode="P")
    palette = [(palette_offset + i) % 256 for i in range(256)]
    image.putpalette(palette * 3)
    return image


class TestMiniCPMVPreprocessCache(CustomTestCase):
    def _processor(self, cache_bytes: int = 16 * 1024 * 1024):
        # Only the artifact path is exercised: no HF processor / tokenizer.
        processor = object.__new__(MiniCPMV4_6MultimodalProcessor)
        processor.image_processor = MiniCPMV4_6ImageProcessor()
        processor.processor_fingerprint = "minicpmv-test"
        processor.trust_mm_content_hashes = False
        processor.mm_preprocess_cache = MultimodalPreprocessCache(cache_bytes)
        processor.mm_processor_executor = None
        processor.io_executor = ThreadPoolExecutor(max_workers=2)
        return processor

    def _assert_same_artifact(self, artifact, expected) -> None:
        self.assertEqual(len(artifact.pixel_values), len(expected["pixel_values"][0]))
        for actual_tile, expected_tile in zip(
            artifact.pixel_values, expected["pixel_values"][0]
        ):
            torch.testing.assert_close(actual_tile, expected_tile, rtol=0, atol=0)
        self.assertEqual(
            [list(size) for size in artifact.tgt_sizes], expected["tgt_sizes"][0]
        )
        self.assertEqual(list(artifact.grid), expected["grids"][0])

    def test_repeated_image_hits_and_matches_uncached(self):
        processor = self._processor()
        try:
            image = _image(640, 480)
            expected = processor.image_processor.preprocess([image])
            artifacts = asyncio.run(processor.prepare_media_artifacts([image, image]))
        finally:
            processor.io_executor.shutdown()

        self.assertEqual(len(artifacts), 2)
        self.assertIs(artifacts[0], artifacts[1])
        self._assert_same_artifact(artifacts[0], expected)
        self.assertEqual(len(processor.mm_preprocess_cache), 1)

    def test_feature_hash_is_derived_from_the_artifact(self):
        processor = self._processor()
        try:
            image = _image(448, 448, offset=7)
            artifact = asyncio.run(processor.prepare_media_artifacts([image]))[0]
        finally:
            processor.io_executor.shutdown()

        self.assertEqual(
            artifact.feature_hash,
            resolve_multimodal_item_hash(
                feature=list(artifact.pixel_values), namespace=artifact.artifact_key
            ),
        )

    def test_palette_images_do_not_share_a_cache_entry(self):
        processor = self._processor()
        try:
            first = _palette_image(0)
            second = _palette_image(17)
            self.assertEqual(first.tobytes(), second.tobytes())
            artifacts = asyncio.run(processor.prepare_media_artifacts([first, second]))
            expected_first = processor.image_processor.preprocess([first])
            expected_second = processor.image_processor.preprocess([second])
        finally:
            processor.io_executor.shutdown()

        self.assertIsNot(artifacts[0], artifacts[1])
        self.assertNotEqual(artifacts[0].content_digest, artifacts[1].content_digest)
        self._assert_same_artifact(artifacts[0], expected_first)
        self._assert_same_artifact(artifacts[1], expected_second)
        self.assertEqual(len(processor.mm_preprocess_cache), 2)

    def test_multi_image_batch_and_non_square(self):
        processor = self._processor()
        images = [
            _image(640, 480, offset=1),
            _image(480, 640, offset=2),
            _image(1024, 768, offset=3),
        ]
        try:
            expected = processor.image_processor.preprocess(images)
            artifacts = asyncio.run(processor.prepare_media_artifacts(images))
        finally:
            processor.io_executor.shutdown()

        self.assertEqual(len(artifacts), len(images))
        for index, artifact in enumerate(artifacts):
            self.assertEqual(
                [list(size) for size in artifact.tgt_sizes],
                expected["tgt_sizes"][index],
            )
            self.assertEqual(list(artifact.grid), expected["grids"][index])
            for actual_tile, expected_tile in zip(
                artifact.pixel_values, expected["pixel_values"][index]
            ):
                torch.testing.assert_close(actual_tile, expected_tile, rtol=0, atol=0)

    def test_cache_is_bounded_and_clear_empties(self):
        processor = self._processor(cache_bytes=1)
        try:
            image = _image(64, 64)
            asyncio.run(processor.prepare_media_artifacts([image]))
            self.assertEqual(len(processor.mm_preprocess_cache), 0)
            processor.mm_preprocess_cache.clear()
            self.assertEqual(len(processor.mm_preprocess_cache), 0)
        finally:
            processor.io_executor.shutdown()


if __name__ == "__main__":
    unittest.main()
