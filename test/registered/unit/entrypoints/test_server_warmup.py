"""Unit tests for model-specific server warmup inputs."""

import base64
import struct
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.http_server import (
    KIMI_VLM_WARMUP_PNG_PICTURE_BASE64,
    MINIMUM_PNG_PICTURE_BASE64,
    _get_vlm_warmup_image_base64,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ServerArgs:
    """Minimal server-args stub for testing warmup-image selection."""

    def __init__(self, architectures):
        self._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures)
        )

    def get_model_config(self):
        return self._model_config


class TestVlmWarmupImage(CustomTestCase):
    def test_kimi_uses_representative_vision_image(self):
        image_base64 = _get_vlm_warmup_image_base64(
            _ServerArgs(["KimiK25ForConditionalGeneration"])
        )

        self.assertEqual(image_base64, KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)
        png = base64.b64decode(image_base64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (512, 512))

    def test_non_kimi_keeps_minimal_startup_image(self):
        self.assertEqual(
            _get_vlm_warmup_image_base64(
                _ServerArgs(["Qwen3VLForConditionalGeneration"])
            ),
            MINIMUM_PNG_PICTURE_BASE64,
        )
        self.assertEqual(
            _get_vlm_warmup_image_base64(_ServerArgs(None)),
            MINIMUM_PNG_PICTURE_BASE64,
        )


if __name__ == "__main__":
    unittest.main()
