"""Unit tests for model-specific server warmup inputs."""

import base64
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import (
    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
    KIMI_VLM_WARMUP_PNG_PICTURE_BASE64,
    MINIMUM_PNG_PICTURE_BASE64,
    _get_vlm_warmup_image_base64,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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


class TestVlmWarmupRoute(CustomTestCase):
    # What the Rust frontend's /model_info reports for a VLM.
    RUST_VLM_MODEL_INFO = {
        "model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "served_model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "tokenizer_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "is_generation": True,
        "has_image_understanding": True,
        "has_audio_understanding": False,
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
    }

    def _warmup_request(self, rust_server):
        override = get_context().override_server_args()
        server_args = override.install()
        self.addCleanup(override.restore)
        model_info = MagicMock(status_code=200)
        model_info.json.return_value = self.RUST_VLM_MODEL_INFO
        tokenizer_manager = SimpleNamespace(served_model_name="served")
        with (
            envs.SGLANG_RUST_SERVER.override(rust_server),
            patch.object(
                http_server,
                "_global_state",
                SimpleNamespace(tokenizer_manager=tokenizer_manager),
            ),
            patch.object(http_server.time, "sleep"),
            patch.object(http_server.requests, "get", return_value=model_info),
            patch.object(
                http_server.requests, "post", return_value=MagicMock(status_code=200)
            ) as post,
        ):
            self.assertTrue(http_server._execute_server_warmup(server_args))
        (url,), kwargs = post.call_args
        return url, kwargs["json"]

    def test_rust_server_warms_up_vlm_through_text_generate(self):
        # The Rust chat route rejects image content, so an image chat warmup
        # would fail startup.
        url, body = self._warmup_request(rust_server=True)
        self.assertTrue(url.endswith("/generate"), url)
        self.assertEqual(body["text"], "The capital city of France is")
        self.assertNotIn("messages", body)

    def test_python_frontend_keeps_image_chat_warmup(self):
        url, body = self._warmup_request(rust_server=False)
        self.assertTrue(url.endswith("/v1/chat/completions"), url)
        self.assertEqual(body["messages"][0]["content"][0]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
