import asyncio
import unittest

from sglang.srt.multimodal.processors.unlimited_ocr import (
    UnlimitedOCRProcessor,
    _resolve_mode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnlimitedOCRProcessor(CustomTestCase):
    def test_unlimited_ocr_resolve_mode_null_and_multi_image_defaults(self):
        # Explicit None image_mode falls back to default single-image mode ("gundam")
        self.assertEqual(
            _resolve_mode({"image_mode": None}, num_images=1),
            {"base_size": 1024, "image_size": 640, "crop_mode": True},
        )
        # Multi-image requests without an explicit image_mode default to "base"
        self.assertEqual(
            _resolve_mode(None, num_images=2),
            {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        )
        self.assertEqual(
            _resolve_mode({"image_mode": None}, num_images=3),
            {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        )
        # Explicit unsupported mode on multi-image still raises ValueError
        with self.assertRaises(ValueError):
            _resolve_mode({"image_mode": "gundam"}, num_images=2)

    def test_unlimited_ocr_empty_or_none_image_data_returns_none(self):
        proc = UnlimitedOCRProcessor.__new__(UnlimitedOCRProcessor)
        self.assertIsNone(asyncio.run(proc.process_mm_data_async(None, "hello")))
        self.assertIsNone(asyncio.run(proc.process_mm_data_async([], "hello")))


if __name__ == "__main__":
    unittest.main()
