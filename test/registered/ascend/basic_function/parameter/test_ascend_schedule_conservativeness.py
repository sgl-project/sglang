import os
import unittest

import requests

from types import SimpleNamespace
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestScheduleConservativeness(CustomTestCase):
    """Testcase: Setting "--schedule-conservativeness" ensures successful request processing,
    rejects new requests when resources are scarce, prevents key-value cache overflow, and dynamically adjusts new_token_ratio.

    [Test Category] Parameter
    [Test Target] --schedule-conservativeness
    """

    @classmethod
    def setUpClass(cls):
        cls.message = ("KV cache pool is full. Retract requests.")
        cls.message1 = ("#new_token_ratio:")
        other_args = [
            "--schedule-conservativeness",
            0,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            2,
            "--mem-fraction-static",
            "0.52"
        ]
        cls.out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_32B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_schedule_conservativeness(self):
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

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

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=100,
            parallel=512,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        self.assertGreaterEqual(metrics["accuracy"], 0.86)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        # error_message information is recorded in the log
        self.assertIn(self.message, content)
        self.assertIn(self.message1, content)
        self.out_log_file.close()
        self.err_log_file.close()
        os.remove("./cache_out_log.txt")
        os.remove("./cache_err_log.txt")


if __name__ == "__main__":
    unittest.main()
