"""
Test suite for Qwen3-235B-A22B MoE models on 8-GPU B200.

Qwen3-235B-A22B is a mixture-of-experts model with 235B total parameters
and 22B activated parameters per token. Requires 8 GPUs for optimal performance.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    MetricReport,
    MetricThreshold,
    is_in_ci,
    popen_launch_server,
    write_metric_report,
)


class TestQwen3_235B_A22B(CustomTestCase):
    """Test Qwen3-235B-A22B MoE model with 8-GPU configuration"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "8",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy for Qwen3-235B MoE model"""
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
        print(f"{metrics=}")

        if is_in_ci():
            report = MetricReport(
                test_name="test_qwen3_235b_a22b_gsm8k",
                test_class=self.__class__.__name__,
                metadata={
                    "model": self.model,
                    "model_type": "MoE",
                    "total_params": "235B",
                    "activated_params": "22B",
                    "tp_size": 8,
                    "gpu_type": "B200",
                    "benchmark": "GSM8K",
                },
            )
            report.add_metric(
                name="accuracy",
                value=metrics["accuracy"],
                unit="ratio",
                higher_is_better=True,
                threshold=MetricThreshold(min=0.90),
            )
            write_metric_report(report)

        # Qwen3-235B should achieve very high accuracy on GSM8K
        self.assertGreater(metrics["accuracy"], 0.90)


class TestQwen3_235B_A22B_FP8(CustomTestCase):
    """Test Qwen3-235B-A22B with FP8 quantization"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-235B-A22B-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "8",
                "--trust-remote-code",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy_fp8(self):
        """Test GSM8K accuracy with FP8 quantization"""
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
        print(f"{metrics=}")

        if is_in_ci():
            report = MetricReport(
                test_name="test_qwen3_235b_a22b_fp8_gsm8k",
                test_class=self.__class__.__name__,
                metadata={
                    "model": self.model,
                    "model_type": "MoE",
                    "quantization": "FP8",
                    "tp_size": 8,
                    "gpu_type": "B200",
                },
            )
            report.add_metric(
                name="accuracy",
                value=metrics["accuracy"],
                unit="ratio",
                higher_is_better=True,
                threshold=MetricThreshold(min=0.88),
            )
            write_metric_report(report)

        # FP8 quantization should maintain high accuracy
        self.assertGreater(metrics["accuracy"], 0.88)


if __name__ == "__main__":
    unittest.main()
