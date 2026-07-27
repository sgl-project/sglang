import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1100, suite="full-8-npu-a3", nightly=True)


class TestMambaCacheWithMemoryRatio(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Test MambaCache basic functions using GSM8K dataset.
    The inference accuracy of the Qwen3-Next-80B-A3B-Instruct model
    on the GSM8K dataset is no less than 0.92.

    [Test Category] Parameter
    [Test Target] --mamba-scheduler-strategy, --mamba-full-memory-ratio, --mamba-track-interval
    """

    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.model_path
    accuracy = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.gsm8k_accuracy
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.5",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mamba-full-memory-ratio",
        "0.9",
        "--mamba-scheduler-strategy",
        "auto",
        "--mamba-track-interval",
        "256",
        "--tp-size",
        "8",
        "--disable-radix-cache",
    ]


class TestMambaCacheWithMambaCacheSize(TestMambaCacheWithMemoryRatio):
    """Testcase: Test MambaCache basic functions using GSM8K dataset.
    The inference accuracy of the Qwen3-Next-80B-A3B-Instruct model
    on the GSM8K dataset is no less than 0.92.

    [Test Category] Parameter
    [Test Target] --mamba-scheduler-strategy, --mamba-ssm-dtype, --max-mamba-cache-size, --mamba-track-interval
    """

    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.5",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mamba-scheduler-strategy",
        "no_buffer",
        "--mamba-track-interval",
        "512",
        "--mamba-ssm-dtype",
        "float32",
        "--tp-size",
        "8",
        "--disable-radix-cache",
        "--max-mamba-cache-size",
        "1024",
    ]


class TestMambaCacheRadix(CustomTestCase):
    """Testcase: Verify Radix Cache reuse with mamba cache.

    [Test Category] Parameter
    [Test Target] Radix Cache reuse, --mamba-ssm-dtype, --mamba-full-memory-ratio
    """

    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.5",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        "8",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--mamba-full-memory-ratio",
        "0.3",
        "--mamba-scheduler-strategy",
        "extra_buffer",  # To reuse Radix Cache, this parameter must be set to extra_buffer
    ]

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.model_path,
            DEFAULT_URL_FOR_TEST,
            timeout=1200,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def test_mamba_cache_kv_cache(self):
        # test kv cache reuse with radix cache,input text should meet page size requirement( >=128 )
        input_ids_first = [1] * 200
        input_ids_second = input_ids_first + [2] * 70

        def make_request(input_ids, expected_cached_tokens):
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                    },
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["meta_info"]["cached_tokens"], expected_cached_tokens
            )

        # First request: no cache
        make_request(input_ids_first, 0)
        # Second request: cache reused, cache token is reused in multiples of 128
        make_request(input_ids_second, 128)

    def test_basic_inference(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)

    def test_mamba_long_sequence(self):
        long_text = "Explain the concept of machine learning in detail." * 4000
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": long_text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1000,
                },
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.text), 0)


if __name__ == "__main__":
    unittest.main()
