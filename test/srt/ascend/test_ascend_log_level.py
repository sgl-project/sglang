import os
import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestLogLevel(CustomTestCase):
    def test_log_level(self):
        other_args = (
            [
                "--log-level",
                "warning",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            if is_npu()
            else ["--log-level", "warning"]
        )
        out_log_file = open("./out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./err_log.txt", "w+", encoding="utf-8")
        process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
                if is_npu()
                else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Paris", response.text)
            out_log_file.seek(0)
            content = out_log_file.read()
            print(content)
            self.assertTrue(len(content) > 0)
            self.assertNotIn("POST /generate HTTP/1.1", content)
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.remove("./out_log.txt")
            os.remove("./err_log.txt")

    def test_log_http_level(self):
        other_args = (
            [
                "--log-level",
                "warning",
                "--log-level-http",
                "info",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            if is_npu()
            else [
                "--log-level",
                "warning",
                "--log-level-http",
                "info",
            ]
        )
        out_log_file = open("./out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./err_log.txt", "w+", encoding="utf-8")
        process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
                if is_npu()
                else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Paris", response.text)
            out_log_file.seek(0)
            content = out_log_file.read()
            print(content)
            self.assertTrue(len(content) > 0)
            self.assertIn("POST /generate HTTP/1.1", content)
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.remove("./out_log.txt")
            os.remove("./err_log.txt")


if __name__ == "__main__":
    unittest.main()
