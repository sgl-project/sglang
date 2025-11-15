import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"

# Global list to collect results
TEST_RESULTS = []


class TestDeepseekV32NasBackend_flashmla(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "nsa",
            "--nsa-prefill-backend",
            "flashmla_sparse",
            "--nsa-decode-backend",
            "flashmla_kv",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            num_shots=20,
            data_path=None,
            num_questions=1400,
            parallel=1400,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            TEST_RESULTS.append(
                {
                    "variant": "flashmla",
                    "prefill_backend": "flashmla_sparse",
                    "decode_backend": "flashmla_kv",
                    "kv_cache": "fp16",
                    "accuracy": metrics["accuracy"],
                }
            )
        self.assertGreater(metrics["accuracy"], 0.935)


class TestDeepseekV32NasBackend_fa3(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "nsa",
            "--nsa-prefill-backend",
            "fa3",
            "--nsa-decode-backend",
            "fa3",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            num_shots=20,
            data_path=None,
            num_questions=1400,
            parallel=1400,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            TEST_RESULTS.append(
                {
                    "variant": "fa3",
                    "prefill_backend": "fa3",
                    "decode_backend": "fa3",
                    "kv_cache": "fp16",
                    "accuracy": metrics["accuracy"],
                }
            )
        self.assertGreater(metrics["accuracy"], 0.935)


class TestDeepseekV32NasBackend_fp8kvcache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--attention-backend",
            "nsa",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            num_shots=20,
            data_path=None,
            num_questions=1400,
            parallel=1400,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            TEST_RESULTS.append(
                {
                    "variant": "fp8kvcache",
                    "prefill_backend": "default",
                    "decode_backend": "default",
                    "kv_cache": "fp8_e4m3",
                    "accuracy": metrics["accuracy"],
                }
            )

            # Write the summary table after all tests complete
            _write_summary_table()
        self.assertGreater(metrics["accuracy"], 0.935)


def _write_summary_table():
    """Write a markdown table with all test results."""
    if not TEST_RESULTS:
        return

    gpu_config = os.getenv("GPU_CONFIG", "8-gpu-h200")

    # Build table header
    summary = f"### {DEEPSEEK_V32_MODEL_PATH} GSM8K Accuracy [{gpu_config}]\n\n"
    summary += "| Variant | Prefill Backend | Decode Backend | KV Cache | Accuracy |\n"
    summary += "|---------|-----------------|----------------|----------|----------|\n"

    # Add each result as a row
    for result in TEST_RESULTS:
        summary += (
            f"| {result['variant']} | {result['prefill_backend']} | "
            f"{result['decode_backend']} | {result['kv_cache']} | "
            f"{result['accuracy']:.3f} |\n"
        )

    write_github_step_summary(summary)


if __name__ == "__main__":
    unittest.main()
