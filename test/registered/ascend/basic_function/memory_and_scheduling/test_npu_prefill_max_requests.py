import os
import re
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestModeImpl(CustomTestCase):
    """Testcase: Verify that the number of requests processed in a single batch by the service does not exceed
    the limit configured by --prefill-max-requests.

    [Test Category] Parameter
    [Test Target] --prefill-max-requests
    """

    model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
    PREFILL_MAX_REQUESTS = 5
    OUT_LOG_PATH = "./out_log.txt"
    ERR_LOG_PATH = "./err_log.txt"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.gsm8k_lower_bound = 0.65

        cls.out_log_file = open(cls.OUT_LOG_PATH, "w+", encoding="utf-8")
        cls.err_log_file = open(cls.ERR_LOG_PATH, "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--model-impl",
                "transformers",
                "--prefill-max-requests",
                str(cls.PREFILL_MAX_REQUESTS),
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove(cls.OUT_LOG_PATH)
        os.remove(cls.ERR_LOG_PATH)

    def test_prefill_max_requests(self):
        """Verify the running-req in log does not exceed --prefill-max-requests."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            num_examples=64,
            num_threads=32,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], self.gsm8k_lower_bound)

        self.err_log_file.seek(0)
        logs = self.err_log_file.read()

        pattern = re.compile(r"Prefill batch, #new-seq[:\s]+(\d+)", re.I)
        matches = pattern.findall(logs)

        self.assertGreater(len(matches), 0, "Prefill batch, #new-seq not found in logs")

        # Iterate through all batches and assert that 1 ≤ num ≤ 5
        for num_str in matches:
            current_num = int(num_str)
            self.assertGreaterEqual(
                current_num, 1, f"Batch size {current_num} is invalid, must >= 1"
            )
            self.assertLessEqual(
                current_num,
                self.PREFILL_MAX_REQUESTS,
                f"Batch size {current_num} exceeds limit, must <= {self.PREFILL_MAX_REQUESTS}",
            )


if __name__ == "__main__":
    unittest.main()
