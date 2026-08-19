"""Unit tests for model-specific server warmup inputs."""

import base64
import struct
import unittest
import zlib

from sglang.srt.entrypoints.http_server import (
    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
    KIMI_VLM_WARMUP_PNG_PICTURE_BASE64,
    MINIMUM_PNG_PICTURE_BASE64,
    _get_vlm_warmup_image_base64,
    _solid_png_base64,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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

    def test_paddleocr_vl_uses_a_page_sized_image(self):
        for model_info in (
            {"architectures": ["PaddleOCRVLForConditionalGeneration"]},
            {"architectures": None, "model_type": "paddleocr_vl"},
        ):
            with self.subTest(model_info=model_info):
                png = base64.b64decode(_get_vlm_warmup_image_base64(model_info))
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(struct.unpack(">II", png[16:24]), (1008, 1008))

        # A page is worth ~1300 image tokens; the minimal image is worth one, so
        # warming with it would leave every shape-dependent setup to the first
        # real page. Patch 14 with a 2x2 merge means 28px per output token.
        self.assertEqual((1008 // 14) * (1008 // 14) // 4, 1296)
        minimal = base64.b64decode(MINIMUM_PNG_PICTURE_BASE64)
        self.assertEqual(struct.unpack(">II", minimal[16:24]), (32, 32))

    def test_generated_png_is_decodable_truecolor(self):
        """Processors assume three channels, and PNG chunk CRCs must be right."""
        png = base64.b64decode(_solid_png_base64(56, 28))
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height, depth, colour_type = struct.unpack(">IIBB", png[16:26])
        self.assertEqual((width, height), (56, 28))
        self.assertEqual((depth, colour_type), (8, 2))  # 8-bit RGB

        offset = 8
        chunks = []
        while offset < len(png):
            (length,) = struct.unpack(">I", png[offset : offset + 4])
            kind = png[offset + 4 : offset + 8]
            payload = png[offset + 8 : offset + 8 + length]
            (crc,) = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])
            self.assertEqual(crc, zlib.crc32(kind + payload) & 0xFFFFFFFF, kind)
            chunks.append(kind)
            offset += 12 + length
        self.assertEqual(chunks, [b"IHDR", b"IDAT", b"IEND"])

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
