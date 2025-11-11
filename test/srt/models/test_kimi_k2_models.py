"""
Test suite for Kimi-K2 models on 8-GPU B200.

Kimi-K2 is a MoE model from Moonshot AI with 32B activated parameters
and 1T total parameters. Includes Kimi-K2-Instruct and Kimi-K2-Thinking variants.
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


class TestKimiK2Instruct(CustomTestCase):
    """Test Kimi-K2-Instruct model with 8-GPU configuration"""

    @classmethod
    def setUpClass(cls):
        cls.model = "moonshotai/Kimi-K2-Instruct"
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
        """Test GSM8K accuracy for Kimi-K2-Instruct"""
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
                test_name="test_kimi_k2_instruct_gsm8k",
                test_class=self.__class__.__name__,
                metadata={
                    "model": self.model,
                    "model_type": "MoE",
                    "total_params": "1T",
                    "activated_params": "32B",
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

        # Kimi-K2 should achieve high accuracy on GSM8K
        self.assertGreater(metrics["accuracy"], 0.90)


class TestKimiK2Thinking(CustomTestCase):
    """Test Kimi-K2-Thinking model with reasoning capabilities"""

    @classmethod
    def setUpClass(cls):
        cls.model = "moonshotai/Kimi-K2-Thinking"
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
        """Test GSM8K accuracy for Kimi-K2-Thinking (reasoning model)"""
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
                test_name="test_kimi_k2_thinking_gsm8k",
                test_class=self.__class__.__name__,
                metadata={
                    "model": self.model,
                    "model_type": "MoE-Reasoning",
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
                threshold=MetricThreshold(min=0.92),
            )
            write_metric_report(report)

        # Thinking model should achieve even higher accuracy
        self.assertGreater(metrics["accuracy"], 0.92)


if __name__ == "__main__":
    unittest.main()
