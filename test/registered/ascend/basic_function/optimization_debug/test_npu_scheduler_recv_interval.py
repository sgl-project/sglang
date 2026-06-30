import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_npu_ci(est_time=800, suite="nightly-1-npu-a3", nightly=True)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH


class TestSchedulerRecvIntervalConsistency(unittest.TestCase):
    """
    Test Case: Verify that when --scheduler-recv-interval > 1, inference latency increases but model score is not affected.

    [Test Category] Parameter Validation
    [Test Target] --scheduler-recv-interval
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.server_process = None

    @classmethod
    def tearDownClass(cls):
        if cls.server_process:
            kill_process_tree(cls.server_process.pid)

    def _run_gsm8k_evaluation(self, scheduler_recv_interval: int):
        self.server_process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--scheduler-recv-interval",
                str(scheduler_recv_interval),
                "--attention-backend",
                "ascend",
                "--disable-radix-cache",
            ],
        )

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=10,
        )
        metrics = run_eval(args)

        kill_process_tree(self.server_process.pid)
        self.server_process = None

        return metrics

    def test_scheduler_recv_interval_consistency(self):
        baseline_metrics = self._run_gsm8k_evaluation(scheduler_recv_interval=1)
        test_metrics = self._run_gsm8k_evaluation(scheduler_recv_interval=90000)

        self.assertGreaterEqual(
            baseline_metrics["score"], 0.38, msg="Baseline score is too low."
        )

        self.assertGreaterEqual(
            test_metrics["score"],
            baseline_metrics["score"] - 0.02,
            msg=f"Score dropped by more than 2%! Baseline: {baseline_metrics['score']:.2%}, Test: {test_metrics['score']:.2%}",
        )

        # When --scheduler-recv-interval is set to 50000, the inference latency increases compared to 1
        self.assertGreaterEqual(
            test_metrics["latency"],
            baseline_metrics["latency"],
            msg="Test latency should be >= baseline latency.",
        )


if __name__ == "__main__":
    unittest.main()
