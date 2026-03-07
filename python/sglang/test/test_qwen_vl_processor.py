import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.multimodal.processors.qwen_vl import (
    IMAGE_MAX_PIXELS_FALLBACK,
    configure_processor_max_pixels,
)
from sglang.test.test_utils import CustomTestCase


class TestQwenVLImageMaxPixels(CustomTestCase):
    def tearDown(self):
        envs.SGLANG_IMAGE_MAX_PIXELS.clear()

    def test_uses_processor_default_when_env_not_set(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace(max_pixels=50176))

        resolved = configure_processor_max_pixels(processor)

        self.assertEqual(resolved, 50176)
        self.assertEqual(processor.image_processor.max_pixels, 50176)

    def test_env_override_takes_precedence(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace(max_pixels=50176))

        with envs.SGLANG_IMAGE_MAX_PIXELS.override(123456):
            resolved = configure_processor_max_pixels(processor)

        self.assertEqual(resolved, 123456)
        self.assertEqual(processor.image_processor.max_pixels, 123456)

    def test_falls_back_when_processor_has_no_max_pixels(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace())

        resolved = configure_processor_max_pixels(processor)

        self.assertEqual(resolved, IMAGE_MAX_PIXELS_FALLBACK)
        self.assertEqual(processor.image_processor.max_pixels, IMAGE_MAX_PIXELS_FALLBACK)


if __name__ == "__main__":
    unittest.main()