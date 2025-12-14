import time
import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def _get_compute_capability():
    """Get the compute capability of the current GPU."""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major * 10 + minor
    return 0


@unittest.skipIf(
    torch.version.hip is not None,
    "ModelOpt quantization is unsupported on ROCm/AMD GPUs",
)
@unittest.skipIf(
    not torch.cuda.is_available(), "ModelOpt FP8 tests require CUDA-enabled NVIDIA GPU"
)
@unittest.skipIf(
    _get_compute_capability() < 89,
    "ModelOpt FP8 requires compute capability 8.9+ (Hopper or newer GPUs)",
)
class MOTest(CustomTestCase):
    model: str = None
    quantization: str = None
    # TODO (jingyu-ml): Add the test case wo fp8_e4m3 kv cache
    kv_cache_dtype: str = None
    gsm8k_accuracy_threshold: float = None
    throughput_threshold: float = None

    @classmethod
    def setUpClass(cls):
        if cls is MOTest:
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
        if cls is MOTest:
            return
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        if self.gsm8k_accuracy_threshold is None:
            self.skipTest("gsm8k_accuracy_threshold not set for this test")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
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


LLAMA_FP8_MODEL = "nvidia/Llama-3.1-8B-Instruct-FP8"
LLAMA_FP4_MODEL = "nvidia/Llama-3.1-8B-Instruct-FP4"
# We only tested 1319 questions, so the results have high variance.
LLAMA_GSM8K_ACC_THRESHOLD_FP8 = 0.74
LLAMA_THROUGHPUT_THRESHOLD_FP8 = 70
LLAMA_GSM8K_ACC_THRESHOLD_FP4 = 0.63
LLAMA_THROUGHPUT_THRESHOLD_FP4 = 90  # For B100

QWEN_FP8_MODEL = "nvidia/Qwen3-8B-FP8"
QWEN_FP4_MODEL = "nvidia/Qwen3-8B-FP4"
# We only tested 1319 questions, so the results have high variance.
QWEN_GSM8K_ACC_THRESHOLD_FP8 = 0.87
QWEN_THROUGHPUT_THRESHOLD_FP8 = 67

QWEN_GSM8K_ACC_THRESHOLD_FP4 = 0.85
QWEN_THROUGHPUT_THRESHOLD_FP4 = 90  # For B100


@unittest.skipIf(
    torch.version.hip is not None,
    "ModelOpt quantization is unsupported on ROCm/AMD GPUs",
)
@unittest.skipIf(
    not torch.cuda.is_available(), "ModelOpt FP8 tests require CUDA-enabled NVIDIA GPU"
)
@unittest.skipIf(
    _get_compute_capability() < 89,
    "ModelOpt FP8 requires compute capability 8.9+ (Hopper or newer GPUs)",
)
class TestMOLlamaFP8(MOTest):
    model = LLAMA_FP8_MODEL
    quantization = "modelopt"
    kv_cache_dtype = "fp8_e4m3"
    gsm8k_accuracy_threshold = LLAMA_GSM8K_ACC_THRESHOLD_FP8
    throughput_threshold = LLAMA_THROUGHPUT_THRESHOLD_FP8


@unittest.skipIf(
    torch.version.hip is not None,
    "ModelOpt quantization is unsupported on ROCm/AMD GPUs",
)
@unittest.skipIf(
    not torch.cuda.is_available(), "ModelOpt FP8 tests require CUDA-enabled NVIDIA GPU"
)
@unittest.skipIf(
    _get_compute_capability() < 89,
    "ModelOpt FP8 requires compute capability 8.9+ (Hopper or newer GPUs)",
)
class TestMOQwenFP8(MOTest):
    model = QWEN_FP8_MODEL
    quantization = "modelopt"
    kv_cache_dtype = "fp8_e4m3"
    gsm8k_accuracy_threshold = QWEN_GSM8K_ACC_THRESHOLD_FP8
    throughput_threshold = QWEN_THROUGHPUT_THRESHOLD_FP8


@unittest.skipIf(
    torch.version.hip is not None,
    "ModelOpt quantization is unsupported on ROCm/AMD GPUs",
)
@unittest.skipIf(
    not torch.cuda.is_available(), "ModelOpt FP8 tests require CUDA-enabled NVIDIA GPU"
)
@unittest.skipIf(
    _get_compute_capability() < 100,
    "ModelOpt FP4 requires compute capability 10.0+ (Blackwell or newer GPUs)",
)
class TestMOLlamaFP4(MOTest):
    model = LLAMA_FP4_MODEL
    quantization = "modelopt"
    kv_cache_dtype = "fp8_e4m3"
    gsm8k_accuracy_threshold = LLAMA_GSM8K_ACC_THRESHOLD_FP4
    throughput_threshold = LLAMA_THROUGHPUT_THRESHOLD_FP4


@unittest.skipIf(
    torch.version.hip is not None,
    "ModelOpt quantization is unsupported on ROCm/AMD GPUs",
)
@unittest.skipIf(
    not torch.cuda.is_available(), "ModelOpt FP8 tests require CUDA-enabled NVIDIA GPU"
)
@unittest.skipIf(
    _get_compute_capability() < 100,
    "ModelOpt FP4 requires compute capability 10.0+ (Blackwell or newer GPUs)",
)
class TestMOQwenFP4(MOTest):
    model = QWEN_FP4_MODEL
    quantization = "modelopt"
    kv_cache_dtype = "fp8_e4m3"
    gsm8k_accuracy_threshold = QWEN_GSM8K_ACC_THRESHOLD_FP4
    throughput_threshold = QWEN_THROUGHPUT_THRESHOLD_FP4


if __name__ == "__main__":
    unittest.main()
