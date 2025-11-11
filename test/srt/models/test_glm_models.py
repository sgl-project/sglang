"""
Test suite for GLM-4.5 and GLM-4.6 models on 8-GPU B200.

GLM-4.5 has 355B total parameters with 32B active parameters.
GLM-4.6 improves on GLM-4.5 with 200K context window (vs 128K).
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


class TestGLM45(CustomTestCase):
    """Test GLM-4.5 model with 8-GPU configuration"""

    @classmethod
    def setUpClass(cls):
        cls.model = "zai-org/GLM-4.5"
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
        """Test GSM8K accuracy for GLM-4.5"""
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
                test_name="test_glm_45_gsm8k",
                test_class=self.__class__.__name__,
                metadata={
                    "model": self.model,
                    "model_type": "MoE",
                    "total_params": "355B",
                    "activated_params": "32B",
                    "context_window": "128K",
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
                threshold=MetricThreshold(min=0.88),
            )
            write_metric_report(report)

        # GLM-4.5 should achieve high accuracy on GSM8K
        self.assertGreater(metrics["accuracy"], 0.88)


class TestGLM46(CustomTestCase):
    """Test GLM-4.6 model with improved context window"""

    @classmethod
    def setUpClass(cls):
        cls.model = "zai-org/GLM-4.6"
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
        """Test GSM8K accuracy for GLM-4.6"""
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
                test_name="test_glm_46_gsm8k",
                test_class=self.__class__.__name__,
                metadata={
                    "model": self.model,
                    "model_type": "MoE",
                    "context_window": "200K",
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
                threshold=MetricThreshold(min=0.89),
            )
            write_metric_report(report)

        # GLM-4.6 should improve over GLM-4.5
        self.assertGreater(metrics["accuracy"], 0.89)


if __name__ == "__main__":
    unittest.main()
