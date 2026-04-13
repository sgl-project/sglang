"""
python3 -m unittest test_llama_tp.py
"""

import gc
import json
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Register for nightly XPU tests only
register_xpu_ci(est_time=300, suite="nightly-xpu", nightly=True)

# Small model for TP=1 (fits in single 12GB B580)
SMALL_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
# Large model for TP=2 (needs 2x 12GB B580)
LLAMA_8B_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _cleanup_xpu_memory():
    gc.collect()
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
    except Exception:
        # Best-effort cleanup only
        pass


class TestLlamaTP1(CustomTestCase):
    """Test Llama 3.2 1B with TP=1 (fits in single 12GB B580)"""

    @classmethod
    def setUpClass(cls):
        _cleanup_xpu_memory()
        cls.model = SMALL_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.environ["SGLANG_USE_SGL_XPU"] = "1"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="xpu",
            other_args=[
                "--attention-backend",
                "intel_xpu",
                "--tp",
                "1",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        _cleanup_xpu_memory()

    def test_generate(self):
        """Test basic text generation with TP=1"""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "What is the capital of France?",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        ret = response.json()
        print(json.dumps(ret, indent=2))

        self.assertIn("text", ret)
        self.assertTrue(len(ret["text"]) > 0)
        # Check that the response mentions Paris
        self.assertIn("Paris", ret["text"])

    def test_chat_completion(self):
        """Test chat completion API with TP=1"""
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "temperature": 0,
                "max_tokens": 16,
            },
        )
        ret = response.json()
        print(json.dumps(ret, indent=2))

        self.assertIn("choices", ret)
        self.assertTrue(len(ret["choices"]) > 0)
        self.assertIn("message", ret["choices"][0])


class TestLlamaTP2(CustomTestCase):
    """Test Llama 3.1 8B with TP=2 (needs 2x 12GB B580)"""

    @classmethod
    def setUpClass(cls):
        _cleanup_xpu_memory()
        cls.model = LLAMA_8B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.environ["SGLANG_USE_SGL_XPU"] = "1"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="xpu",
            other_args=[
                "--attention-backend",
                "intel_xpu",
                "--tp",
                "2",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        _cleanup_xpu_memory()

    def test_generate(self):
        """Test basic text generation with TP=2"""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "What is the capital of France?",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        ret = response.json()
        print(json.dumps(ret, indent=2))

        self.assertIn("text", ret)
        self.assertTrue(len(ret["text"]) > 0)
        # Check that the response mentions Paris
        self.assertIn("Paris", ret["text"])

    def test_chat_completion(self):
        """Test chat completion API with TP=2"""
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "temperature": 0,
                "max_tokens": 16,
            },
        )
        ret = response.json()
        print(json.dumps(ret, indent=2))

        self.assertIn("choices", ret)
        self.assertTrue(len(ret["choices"]) > 0)
        self.assertIn("message", ret["choices"][0])


if __name__ == "__main__":
    unittest.main()
