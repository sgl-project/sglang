"""DeepSeek-OCR-2 boot + short-decode smoke test on Intel XPU (1-GPU nightly).

Mirrors ``test_deepseek_ocr.py`` but targets the newer OCR-2 checkpoint and
registers to the XPU 1-GPU nightly suite.
"""

import json
import os
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers import get_tokenizer
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=900, suite="nightly-xpu-1-gpu", nightly=True)


class TestDeepSeekOCR2XPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "deepseek-ai/DeepSeek-OCR-2"
        cls.tokenizer = get_tokenizer(cls.model)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.image_path = str(
            (Path(__file__).resolve().parents[3] / "examples/assets/example_image.png")
        )
        if not os.path.exists(cls.image_path):
            raise FileNotFoundError(f"Image not found: {cls.image_path}")
        cls.common_args = [
            "--device",
            "xpu",
            "--attention-backend",
            "intel_xpu",
            "--mem-fraction-static",
            "0.50",
        ]
        os.environ["SGLANG_USE_SGL_XPU"] = "1"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *cls.common_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=30)
            except Exception:
                kill_process_tree(cls.process.pid)

    def get_request_json(self, max_new_tokens=32, n=1):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "<image>\n<|grounding|>Convert the document to pure text.",
                "image_data": self.image_path,
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
        return response.json()

    def run_decode(self, max_new_tokens=128, n=1):
        ret = self.get_request_json(max_new_tokens=max_new_tokens, n=n)
        print(json.dumps(ret, indent=2))

        def assert_one_item(item):
            if item["meta_info"]["finish_reason"]["type"] == "stop":
                self.assertEqual(
                    item["meta_info"]["finish_reason"]["matched"],
                    self.tokenizer.eos_token_id,
                )
            elif item["meta_info"]["finish_reason"]["type"] == "length":
                self.assertEqual(
                    len(item["output_ids"]), item["meta_info"]["completion_tokens"]
                )
                self.assertEqual(len(item["output_ids"]), max_new_tokens)

        if n == 1:
            assert_one_item(ret)
        else:
            self.assertEqual(len(ret), n)
            for i in range(n):
                assert_one_item(ret[i])

    def test_decode(self):
        self.run_decode()


if __name__ == "__main__":
    unittest.main()
