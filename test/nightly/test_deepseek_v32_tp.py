import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=900, suite="nightly-8-gpu-h200", nightly=True)

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"

# Global list to collect results
TEST_RESULTS = []


class TestDeepseekV32_TP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Pure TP configuration without --dp and --enable-dp-attention
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
                    "variant": "pure_tp",
                    "prefill_backend": "flashmla_sparse",
                    "decode_backend": "flashmla_kv",
                    "kv_cache": "fp16",
                    "accuracy": metrics["accuracy"],
                }
            )
        self.assertGreater(metrics["accuracy"], 0.935)


class TestDeepseekV32_Partial_TP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Partial TP configuration with dp=4 and dp-attention enabled
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
            "4",
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
                    "variant": "partial_tp",
                    "prefill_backend": "flashmla_sparse",
                    "decode_backend": "flashmla_kv",
                    "kv_cache": "fp16",
                    "accuracy": metrics["accuracy"],
                }
            )
        self.assertGreater(metrics["accuracy"], 0.935)


class TestDeepseekV32_TP_MTP(CustomTestCase):
    """Test DeepSeek V3.2 with pure TP + MTP (EAGLE speculative decoding)."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-frac",
            "0.7",
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

    def test_a_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")
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

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            TEST_RESULTS.append(
                {
                    "variant": "tp_mtp",
                    "prefill_backend": "default",
                    "decode_backend": "EAGLE",
                    "kv_cache": "fp16",
                    "accuracy": metrics["accuracy"],
                    "avg_spec_accept_length": avg_spec_accept_length,
                }
            )
        self.assertGreater(metrics["accuracy"], 0.935)
        self.assertGreater(avg_spec_accept_length, 2.5)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            # Update last result with speed data
            if TEST_RESULTS and TEST_RESULTS[-1]["variant"] == "tp_mtp":
                TEST_RESULTS[-1]["speed"] = speed

            # Write the summary table after all tests complete
            _write_summary_table()

        self.assertGreater(acc_length, 2.5)
        self.assertGreater(speed, 70)


def _format_optional_metric(value, fmt=".2f", suffix=""):
    """Format an optional metric value, returning '-' if not available."""
    if value is None:
        return "-"
    return f"{value:{fmt}}{suffix}"


def _write_summary_table():
    """Write a markdown table with all test results."""
    if not TEST_RESULTS:
        return

    gpu_config = os.getenv("GPU_CONFIG", "8-gpu-h200")

    # Build table header - keep original columns + add MTP-specific ones
    summary = (
        f"### {DEEPSEEK_V32_MODEL_PATH} GSM8K Accuracy (TP Tests) [{gpu_config}]\n\n"
    )
    summary += "| Variant | Prefill Backend | Decode Backend | KV Cache | Accuracy | Spec Acc Len | Speed |\n"
    summary += "|---------|-----------------|----------------|----------|----------|--------------|-------|\n"

    # Add each result as a row
    for result in TEST_RESULTS:
        summary += (
            f"| {result['variant']} | {result['prefill_backend']} | "
            f"{result['decode_backend']} | {result['kv_cache']} | "
            f"{result['accuracy']:.3f} | "
            f"{_format_optional_metric(result.get('avg_spec_accept_length'))} | "
            f"{_format_optional_metric(result.get('speed'), '.1f', ' tok/s')} |\n"
        )

    write_github_step_summary(summary)


if __name__ == "__main__":
    unittest.main()
