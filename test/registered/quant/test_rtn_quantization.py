import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class _BaseRTNQuantizationTest(CustomTestCase):
    MODEL = ""

    @classmethod
    def setUpClass(cls):
        cls.model = cls.MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--quantization", "rtn"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.6)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "ignore_eos": True,
            },
        )
        return response.json()

    def test_throughput(self):
        max_tokens = 256

        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(res["text"])
        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")
        assert throughput >= 15

    def test_basic_generation(self):
        """Test basic text generation with RTN quantization."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "What is machine learning?",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 50,
                },
            },
        )

        result = response.json()
        self.assertIn("text", result)
        self.assertIsInstance(result["text"], str)
        self.assertGreater(len(result["text"]), 0)

    def test_chat_completion(self):
        """Test chat completion with RTN quantization."""
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 50,
                "temperature": 0,
            },
        )

        result = response.json()
        self.assertIn("choices", result)
        self.assertGreater(len(result["choices"]), 0)
        self.assertIn("message", result["choices"][0])
        self.assertIn("content", result["choices"][0]["message"])


class TestRTNQuantizationLinear(_BaseRTNQuantizationTest):
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"


class TestRTNQuantizationMoE(_BaseRTNQuantizationTest):
    MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


if __name__ == "__main__":
    unittest.main()
