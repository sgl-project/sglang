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


class MOFP8Test(CustomTestCase):
    model: str = None
    quantization: str = None
    # TODO (jingyu-ml): Add the test case wo fp8_e4m3 kv cache
    kv_cache_dtype: str = None
    gsm8k_accuracy_threshold: float = None
    throughput_threshold: float = None

    @classmethod
    def setUpClass(cls):
        if cls is MOFP8Test:
            raise unittest.SkipTest("Skip base test class")

        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = []
        if cls.quantization:
            other_args.extend(["--quantization", cls.quantization])
        if cls.kv_cache_dtype:
            other_args.extend(["--kv-cache-dtype", cls.kv_cache_dtype])

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls is MOFP8Test:
            return
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        if self.gsm8k_accuracy_threshold is None:
            self.skipTest("gsm8k_accuracy_threshold not set for this test")

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
        self.assertGreater(metrics["accuracy"], self.gsm8k_accuracy_threshold)

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
        # Warm-up
        self.run_decode(max_new_tokens=32)
        max_tokens = 256
        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(res["text"])
        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")
        self.assertGreaterEqual(throughput, self.throughput_threshold)


# At the top of the file
LLAMA_MODEL = "nvidia/Llama-3.1-8B-Instruct-FP8"
LLAMA_GSM8K_ACC_THRESHOLD = 0.69
LLAMA_THROUGHPUT_THRESHOLD = 120

QWEN_MODEL = "nvidia/Qwen3-8B-FP8"
QWEN_GSM8K_ACC_THRESHOLD = 0.90
QWEN_THROUGHPUT_THRESHOLD = 120

class TestMOLlamaFP8(MOFP8Test):
    model = LLAMA_MODEL
    quantization = "modelopt"
    kv_cache_dtype = "fp8_e4m3"
    gsm8k_accuracy_threshold = LLAMA_GSM8K_ACC_THRESHOLD
    throughput_threshold = LLAMA_THROUGHPUT_THRESHOLD

class TestMOQwenFP8(MOFP8Test):
    model = QWEN_MODEL
    quantization = "modelopt"
    kv_cache_dtype = "fp8_e4m3"
    gsm8k_accuracy_threshold = QWEN_GSM8K_ACC_THRESHOLD
    throughput_threshold = QWEN_THROUGHPUT_THRESHOLD

if __name__ == "__main__":
    unittest.main()
