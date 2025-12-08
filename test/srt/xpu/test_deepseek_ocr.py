"""
python3 -m unittest test_deepseek_ocr.py
"""

import json
import os
import unittest

import requests
from transformers import AutoProcessor, AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    download_image_with_retry,
    popen_launch_server,
)


class TestDeepSeekOCR(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.image_url = DEFAULT_IMAGE_URL
        cls.image = download_image_with_retry(cls.image_url)
        cls.model = "deepseek-ai/DeepSeek-OCR"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, use_fast=False)
        cls.processor = AutoProcessor.from_pretrained(cls.model, trust_remote_code=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.common_args = [
            "--device",
            "xpu",
            "--attention-backend",
            "intel_xpu",
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
        cls.eos_token_id = [cls.tokenizer.eos_token_id]

    @classmethod
    def tearDownClass(cls):
        """Fixture that is run once after all tests in the class."""
        kill_process_tree(cls.process.pid)

    def get_request_json(self, max_new_tokens=32, n=1):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "<image>\n<|grounding|>Convert the document to pure text.",
                "image_data": self.image,
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
        return response.json()

    def run_decode(
        self,
        max_new_tokens=128,
        n=1,
    ):

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

        # Determine whether to assert a single item or multiple items based on n
        if n == 1:
            assert_one_item(ret)
        else:
            self.assertEqual(len(ret), n)
            for i in range(n):
                assert_one_item(ret[i])

        print("=" * 100)

    def test_triton_moe(self):
        self.run_decode()


if __name__ == "__main__":
    unittest.main()
