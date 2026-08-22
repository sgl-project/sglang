"""DeepSeek-OCR-2 image recognition and layout parsing on Ascend NPU.

Launches the sglang server with the DeepSeek-OCR-2 model on Ascend NPU and
verifies:
  1. Basic image text recognition (pure-text OCR prompt).
  2. Layout-aware document parsing (markdown conversion prompt).

[Test Category] Model
[Test Target] deepseek-ai/DeepSeek-OCR-2
"""

import json
import os
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_OCR_2_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-4-npu-a3", nightly=True)

# The CI convention places weights under /root/.cache/modelscope/hub/models/.
# On local dev hosts the ModelScope cache may use the newer layout
# (~/.cache/modelscope/models/<owner>--<name>/snapshots/<revision>); fall back
# to it so the test also runs outside CI.
if not os.path.exists(DEEPSEEK_OCR_2_WEIGHTS_PATH):
    _local_ocr2 = os.path.expanduser(
        "~/.cache/modelscope/models/deepseek-ai--DeepSeek-OCR-2/snapshots/master"
    )
    if os.path.exists(_local_ocr2):
        DEEPSEEK_OCR_2_WEIGHTS_PATH = _local_ocr2


class TestDeepSeekOCR2(CustomTestCase):
    """Verify DeepSeek-OCR-2 inference on Ascend NPU.

    [Test Category] Model
    [Test Target] deepseek-ai/DeepSeek-OCR-2
    """

    model = DEEPSEEK_OCR_2_WEIGHTS_PATH
    image_path = str(
        Path(__file__).resolve().parents[4] / "examples/assets/example_image.png"
    )
    timeout_for_server_launch = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    other_args = [
        "--trust-remote-code",
        "--enable-multimodal",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.35",
        "--tp-size",
        "1",
        "--log-level",
        "info",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=cls.timeout_for_server_launch,
            other_args=list(cls.other_args),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            kill_process_tree(cls.process.pid)

    def _run_ocr(self, prompt: str, max_new_tokens: int = 256):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "image_data": self.image_path,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=180,
        )
        self.assertEqual(
            response.status_code, 200, f"OCR request failed: {response.text[:500]}"
        )
        return response.json()

    def test_ocr_pure_text_recognition(self):
        """Basic image-to-text recognition produces non-empty output."""
        ret = self._run_ocr("<image>\n<|grounding|>Convert the document to pure text.")
        print(json.dumps(ret, ensure_ascii=False, indent=2))

        self.assertIn("text", ret)
        self.assertTrue(
            len(ret["text"].strip()) > 0, f"Empty OCR output: {ret['text']!r}"
        )
        self.assertIn("meta_info", ret)
        self.assertGreater(
            ret["meta_info"].get("image_tokens", 0), 0, "No image tokens processed"
        )
        self.assertEqual(ret["meta_info"]["finish_reason"]["type"], "stop")

    def test_ocr_layout_parsing(self):
        """Layout-aware markdown conversion prompt produces structured output."""
        ret = self._run_ocr(
            "<image>\n<|grounding|>Convert the document to markdown.",
            max_new_tokens=512,
        )
        print(json.dumps(ret, ensure_ascii=False, indent=2))

        self.assertIn("text", ret)
        self.assertTrue(
            len(ret["text"].strip()) > 0, f"Empty OCR output: {ret['text']!r}"
        )


if __name__ == "__main__":
    unittest.main()
