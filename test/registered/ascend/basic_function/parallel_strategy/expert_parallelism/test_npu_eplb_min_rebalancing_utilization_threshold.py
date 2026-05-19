import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestEplbMinRebalancingUtilizationThresholdBase(CustomTestCase):
    """
    Testcase：Validates that rebalancing operations are triggered or skipped based on the configured
    --eplb-min-rebalancing-utilization-threshold value and current load balance.

    [Test Category] Parameter
    [Test Target] --eplb-min-rebalancing-utilization-threshold, --eplb-rebalance-layers-per-chunk
    """

    model = QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
    accuracy = 0.86
    common_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--trust-remote-code",
        "--chunked-prefill-size",
        "1024",
        "--tp-size",
        "8",
        "--quantization",
        "modelslim",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        16,
        "--eplb-rebalance-num-iterations",
        50,
        "--expert-distribution-recorder-buffer-size",
        50,
        "--enable-expert-distribution-metrics",
        "--eplb-rebalance-layers-per-chunk",
        "1",
    ]

    out_file_path = "./rebalance_out_log.txt"
    err_file_path = "./rebalance_err_log.txt"
    log_info = "Skipped ep rebalancing: current GPU utilization"
    test_args = ["--eplb-min-rebalancing-utilization-threshold", 0.05]

    @classmethod
    def setUpClass(cls):
        if hasattr(cls, "out_file_path"):
            cls.out_file = open(cls.out_file_path, "w+", encoding="utf-8")
        if hasattr(cls, "err_file_path"):
            cls.err_file = open(cls.err_file_path, "w+", encoding="utf-8")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.common_args + cls.test_args,
            env={
                "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
                "SGLANG_EXPERT_LOCATION_UPDATER_CANARY": "1",
                "HCCL_BUFFSIZE": "1024",
                "SGLANG_DEEPEP_BF16_DISPATCH": "1",
                "TRANSFORMERS_VERBOSITY": "error",
                **os.environ,
            },
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_file.close()
        cls.err_file.close()
        os.remove("./rebalance_out_log.txt")
        os.remove("./rebalance_err_log.txt")

    def test_gsm8k(self):
        args = SimpleNamespace(
            max_tokens=512,
            base_url=DEFAULT_URL_FOR_TEST,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
            num_shots=5,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(
            metrics["score"],
            self.accuracy,
            f'Accuracy of {self.model} is {str(metrics["score"])}, is lower than {self.accuracy}',
        )

        """
        Testcase：When the configuration --eplb-min-rebalancing-utilization-threshold is set to 0.05, if the load balance
        exceeds this threshold, rebalancing operations are skipped.
        """
        self.err_file.seek(0)
        content = self.err_file.read()
        self.assertIn(self.log_info, content)
        self.assertIn("[EPLBManager] rebalance start", content)


class TestEplbMinRebalancingUtilizationThreshold095(
    TestEplbMinRebalancingUtilizationThresholdBase
):
    """
    Testcase：When the configuration --eplb-min-rebalancing-utilization-threshold is set to 0.95, if load balancing
    is less than or equal to this threshold, rebalancing operations are triggered.
    """

    log_info = "rebalance end"
    test_args = ["--eplb-min-rebalancing-utilization-threshold", 0.95]

    @classmethod
    def setUpClass(cls):
        if hasattr(cls, "out_file_path"):
            cls.out_file = open(cls.out_file_path, "w+", encoding="utf-8")
        if hasattr(cls, "err_file_path"):
            cls.err_file = open(cls.err_file_path, "w+", encoding="utf-8")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.common_args + cls.test_args,
            env={
                "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
                "SGLANG_EXPERT_LOCATION_UPDATER_CANARY": "1",
                "HCCL_BUFFSIZE": "1024",
                "SGLANG_DEEPEP_BF16_DISPATCH": "1",
                "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1",
                "TRANSFORMERS_VERBOSITY": "error",
                **os.environ,
            },
            return_stdout_stderr=(cls.out_file, cls.err_file),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
