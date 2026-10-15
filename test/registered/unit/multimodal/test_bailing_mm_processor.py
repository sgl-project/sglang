"""Regression tests for the Ling image/video-only processor contract."""

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.multimodal.processors.bailing_mm import (
    BailingMMMultimodalProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBailingMMProcessor(CustomTestCase):
    def test_audio_request_fails_before_preprocessing(self):
        """The public image/video checkpoint must reject audio at the API boundary."""
        processor = BailingMMMultimodalProcessor.__new__(BailingMMMultimodalProcessor)
        request = SimpleNamespace(audio_data=["audio.wav"])

        with self.assertRaisesRegex(ValueError, "Audio inputs are not supported"):
            asyncio.run(
                processor.process_mm_data_async(
                    image_data=[],
                    audio_data=request.audio_data,
                    input_text="test",
                    request_obj=request,
                )
            )


if __name__ == "__main__":
    unittest.main()
