import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestLogLevel(CustomTestCase):
    """Testcase：Verify set log-level parameter, the printed log level is the same as the configured log level and the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --log-level
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    OUT_LOG_PATH = "./out_log.txt"
    ERR_LOG_PATH = "./err_log.txt"

    def _launch_server_and_run_infer(self, other_args):
        out_log_file = None
        err_log_file = None
        process = None
        try:
            out_log_file = open(self.OUT_LOG_PATH, "w+", encoding="utf-8")
            err_log_file = open(self.ERR_LOG_PATH, "w+", encoding="utf-8")
            process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )
            health_resp = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(health_resp.status_code, 200)
            gen_resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(gen_resp.status_code, 200)
            self.assertIn("Paris", gen_resp.text)
            out_log_file.seek(0)
            return out_log_file.read()
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.remove(self.OUT_LOG_PATH)
            os.remove(self.ERR_LOG_PATH)

    def test_log_level(self):
        # Verify set --log-level=warning and not set --log-level-http, logs print only warning level (no HTTP info)
        other_args = [
            "--log-level",
            "warning",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        log_content = self._launch_server_and_run_infer(other_args)
        self.assertNotIn("POST /generate HTTP/1.1", log_content)

    def test_log_http_level(self):
        # Verify set --log-level=warning and set --log-level-http=info, log level print http info
        other_args = [
            "--log-level",
            "warning",
            "--log-level-http",
            "info",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        log_content = self._launch_server_and_run_infer(other_args)
        self.assertIn("POST /generate HTTP/1.1", log_content)


if __name__ == "__main__":
    unittest.main()
