"""
XPU tests for Qwen3 models.

Follows the same structure as test_deepseek_ocr.py: launch server with
--device xpu and --attention-backend intel_xpu, then exercise /generate
and assert on response meta_info and output_ids.

Usage:
  python3 -m unittest test_qwen3.TestQwen3.test_decode
  python3 -m unittest test_qwen3
"""

import json
import os
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestQwen3(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-0.6B"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, use_fast=False)
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

    @classmethod
    def tearDownClass(cls):
        """Fixture that is run once after all tests in the class."""
        kill_process_tree(cls.process.pid)

    def get_request_json(self, max_new_tokens=32, n=1):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Hello, how are you?",
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

        if n == 1:
            assert_one_item(ret)
        else:
            self.assertEqual(len(ret), n)
            for i in range(n):
                assert_one_item(ret[i])

        print("=" * 100)

    def test_decode(self):
        self.run_decode()


class TestQwen3Triton(TestQwen3):
    """Qwen3 on XPU with Triton (non-SGL) backend."""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-0.6B"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, use_fast=False)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.common_args = [
            "--device",
            "xpu",
            "--attention-backend",
            "intel_xpu",
        ]
        os.environ["SGLANG_USE_SGL_XPU"] = "0"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *cls.common_args,
            ],
        )


if __name__ == "__main__":
    unittest.main()
