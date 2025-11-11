"""
Test suite for MiniMax-M2 model on 8-GPU B200.

MiniMax-M2 is a MoE model with 10B activated parameters and 230B total parameters.
It's an interleaved thinking model optimized for coding and agentic workflows.
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


class TestMiniMaxM2(CustomTestCase):
    """Test MiniMax-M2 model with 8-GPU configuration"""

    @classmethod
    def setUpClass(cls):
        cls.model = "MiniMaxAI/MiniMax-M2"
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
        """Test GSM8K accuracy for MiniMax-M2"""
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
                test_name="test_minimax_m2_gsm8k",
                test_class=self.__class__.__name__,
                metadata={
                    "model": self.model,
                    "model_type": "MoE-Thinking",
                    "total_params": "230B",
                    "activated_params": "10B",
                    "tp_size": 8,
                    "gpu_type": "B200",
                    "benchmark": "GSM8K",
                    "thinking_format": "<think>...</think>",
                },
            )
            report.add_metric(
                name="accuracy",
                value=metrics["accuracy"],
                unit="ratio",
                higher_is_better=True,
                threshold=MetricThreshold(min=0.87),
            )
            write_metric_report(report)

        # MiniMax-M2 should achieve good accuracy on GSM8K
        self.assertGreater(metrics["accuracy"], 0.87)


if __name__ == "__main__":
    unittest.main()
