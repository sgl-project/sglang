import unittest
import requests
import time

from sglang.test.ascend.test_ascend_utils import QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=500, suite="nightly-8-npu-a3", nightly=True)


class TestMambaCache(CustomTestCase):
    """Testcase：Verify the MambaCache with different parameters, concurrent requests, long sequence,
    Inference request successful

    [Test Category] Parameter
    [Test Target] --mamba-full-memory-ratio, --mamba-ssm-dtype, --mamba-track-interval, --mamba-track-size,
    --max-mamba-cache-size
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.model_path

    test_prompt = "The capital of France is"
    expected_output = "Paris"

    @classmethod
    def setUpClass(cls):
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def _launch_server_with_mamba_params(
        self,
        max_mamba_cache_size=None,
        mamba_ssm_dtye=None,
        mamba_full_memory_ratio=0.9,
        mamba_scheduler_strategy="auto",
        mamba_track_interval=256,
    ):
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.5",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mamba-full-memory-ratio",
            mamba_full_memory_ratio,
            "--max-mamba-cache-size",
            "1024",
            "--mamba-ssm-dtype",
            "float32",
            "--mamba-scheduler-strategy",
            mamba_scheduler_strategy,
            "--mamba-track-interval",
            mamba_track_interval,
            "--tp-size",
            8,
            "--disable-radix-cache",
        ]
        if max_mamba_cache_size is not None:
            other_args.extend(["--max-mamba-cache-size", max_mamba_cache_size])
        if mamba_ssm_dtye is not None:
            other_args.extend(["--mamba-ssm-dtype", mamba_ssm_dtye])
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        return process

    def _tes_basic_inference(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": self.test_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(self.expected_output, response.text)
        return response.text

    def test_mamba_long_sequence(self):
        self.process = self._launch_server_with_mamba_params(
            max_mamba_cache_size=1024
        )
        try:
            time.sleep(5)
            long_prompt = "Explain the concept of machine learning in detail." * 10
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": long_prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 128,
                    },
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.text), 0)
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_track_interval(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_track_interval=128
        )
        try:
            time.sleep(5)
            self._tes_basic_inference()
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_batch(self):
        # test use mamba in batch requests can work properly
        self.process = self._launch_server_with_mamba_params(
            max_mamba_cache_size=512,
        )
        prompts = [
            "What is AI",
            "Explain neural network",
            "How does deep learning differ from machine learning",
            "What is reinforcement learning",
            "Explain natural language processing",
            "What are neural network layers",
            "How do activation functions work",
            "Explain backpropagation",
            "What is computer vision",
            "How do LLMs work",
        ]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 64,
                },
            },
        )
        results = response.json()
        for i, result in enumerate(results):
            self.assertGreater(len(result["text"]), 0)

    def test_mamba_scheduler_no_buffer(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_scheduler_strategy="no_buffer",
        )
        try:
            time.sleep(5)
            self._tes_basic_inference()
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_ssm_dtype_float32(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_ssm_dtye="float32",
        )
        try:
            time.sleep(5)
            self._tes_basic_inference()

        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_ssm_dtype_bfloat16(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_ssm_dtye="bfloat16",
        )
        try:
            time.sleep(5)
            self._tes_basic_inference()
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_ssm_dtype_float16(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_ssm_dtye="float16",
        )
        try:
            time.sleep(5)
            self._tes_basic_inference()
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_full_memory_ratio(self):
        self.process = self._launch_server_with_mamba_params(
            mamba_full_memory_ratio=0.5,
        )
        try:
            time.sleep(5)
            self._tes_basic_inference()
        finally:
            kill_process_tree(self.process.pid)

    def test_mamba_max_mamba_cache_size_2048(self):
        self.process = self._launch_server_with_mamba_params(
            max_mamba_cache_size=2048,
        )
        try:
            time.sleep(5)
            self._tes_basic_inference()
        finally:
            kill_process_tree(self.process.pid)


if __name__ == "__main__":
    unittest.main()
