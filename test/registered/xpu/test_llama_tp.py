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


class TestLlamaTP2(CustomTestCase):
    """Test Llama 3.2 3B with TP=2 (fits in 2x 12GB B580)"""

    @classmethod
    def _cleanup_xpu_memory(cls):
        gc.collect()
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
        except Exception:
            # Best-effort cleanup only; tests should continue if cleanup is unavailable.
            pass

    @classmethod
    def setUpClass(cls):
        cls._cleanup_xpu_memory()
        cls.model = "meta-llama/Llama-3.2-3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.common_args = [
            "--device",
            "xpu",
            "--attention-backend",
            "intel_xpu",
            "--tp",
            "2",
            "--trust-remote-code",
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
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        cls._cleanup_xpu_memory()

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
