"""
gRPC Router E2E Test - Test Request Length Validation

This test file is REUSED from test/srt/openai_server/validation/test_request_length_validation.py
with minimal changes:
    num_workers=2,
- Swap popen_launch_server() → popen_launch_grpc_router()
- Update teardown to cleanup router + workers
- All test logic and assertions remain identical

Run with:
    pytest py_test/e2e_grpc/e2e_grpc/validation/test_request_length_validation.py -v
"""

import unittest

# CHANGE: Import router launcher instead of server launcher
import sys
from pathlib import Path
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_grpc_router

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,

)

class TestRequestLengthValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # CHANGE: Launch gRPC router with integrated workers (single command)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.model = "/home/ubuntu/models/llama-3.1-8b-instruct"
        # Start server with auto truncate disabled
        cls.cluster = popen_launch_grpc_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            worker_args=("--max-total-tokens", "1000", "--context-length", "1000"),
            num_workers=1,
            tp_size=2,
        )

    @classmethod
    def tearDownClass(cls):
        # Cleanup router and workers
        kill_process_tree(cls.cluster["process"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    def test_input_length_longer_than_context_length(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello " * 1200  # Will tokenize to more than context length

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_input_length_longer_than_maximum_allowed_length(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello " * 999  # the maximum allowed length is 994 tokens

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        self.assertIn("is longer than the model's context length", str(cm.exception))

    def test_max_tokens_validation(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello "

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
                max_tokens=1200,
            )

        self.assertIn(
            "max_completion_tokens is too large",
            str(cm.exception),
        )

if __name__ == "__main__":
    unittest.main()
