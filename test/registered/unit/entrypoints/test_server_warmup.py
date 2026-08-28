"""Unit tests for model-specific server warmup inputs."""

import base64
import struct
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.http_server import (
    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
    KIMI_VLM_WARMUP_PNG_PICTURE_BASE64,
    MINIMUM_PNG_PICTURE_BASE64,
    _get_vlm_warmup_image_base64,
    _resolve_warmup_prefill_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPrefillWarmup(unittest.TestCase):
    def test_caps_to_chunked_prefill_size(self):
        server_args = SimpleNamespace(chunked_prefill_size=512, context_length=4096)
        self.assertEqual(_resolve_warmup_prefill_tokens(server_args, 2048, 8), 512)

    def test_caps_to_context_length(self):
        server_args = SimpleNamespace(chunked_prefill_size=-1, context_length=1024)
        self.assertEqual(_resolve_warmup_prefill_tokens(server_args, 2048, 8), 1016)

    def test_preserves_configured_length_within_limits(self):
        server_args = SimpleNamespace(chunked_prefill_size=4096, context_length=4096)
        self.assertEqual(_resolve_warmup_prefill_tokens(server_args, 2048, 8), 2048)

    def test_rejects_context_without_input_room(self):
        server_args = SimpleNamespace(chunked_prefill_size=-1, context_length=8)
        with self.assertRaisesRegex(ValueError, "context_length must exceed"):
            _resolve_warmup_prefill_tokens(server_args, 2048, 8)


class TestVlmWarmupImage(CustomTestCase):
    def test_kimi_k2_uses_representative_vision_image(self):
        image_base64 = _get_vlm_warmup_image_base64(
            {"architectures": ["KimiK25ForConditionalGeneration"]}
        )
        self.assertEqual(image_base64, KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)

        png = base64.b64decode(KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (512, 512))

    def test_kimi_k3_uses_native_patch_grid_image(self):
        for model_info in (
            {"architectures": ["KimiK3ForConditionalGeneration"]},
            {"architectures": None, "model_type": "kimi_k3"},
        ):
            with self.subTest(model_info=model_info):
                self.assertEqual(
                    _get_vlm_warmup_image_base64(model_info),
                    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
                )

        png = base64.b64decode(KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (448, 448))

    def test_other_vlms_keep_minimal_startup_image(self):
        self.assertEqual(
            _get_vlm_warmup_image_base64(
                {"architectures": ["Qwen3VLForConditionalGeneration"]}
            ),
            MINIMUM_PNG_PICTURE_BASE64,
        )
        self.assertEqual(
            _get_vlm_warmup_image_base64({"architectures": None}),
            MINIMUM_PNG_PICTURE_BASE64,
        )


if __name__ == "__main__":
    unittest.main()
