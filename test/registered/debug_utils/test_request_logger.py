import glob
import io
import json
import os
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)


class BaseTestRequestLogger:
    log_requests_format = None

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = popen_launch_server(
            "Qwen/Qwen3-0.6B",
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--log-requests",
                "--log-requests-level",
                "2",
                "--log-requests-format",
                cls.log_requests_format,
                "--skip-server-warmup",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def _send_request(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)
        return self.stdout.getvalue() + self.stderr.getvalue()


class TestRequestLoggerText(BaseTestRequestLogger, CustomTestCase):
    log_requests_format = "text"

    def test_text_format_logging(self):
        combined_output = self._send_request()
        self.assertIn("Receive:", combined_output)
        self.assertIn("Finish:", combined_output)


class TestRequestLoggerJson(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = popen_launch_server(
            "Qwen/Qwen3-0.6B",
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--log-requests",
                "--log-requests-level",
                "2",
                "--log-requests-format",
                "json",
                "--log-requests-target",
                "stdout",
                cls.temp_dir,
                "--skip-server-warmup",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_json_format_logging(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(1)

        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        received_found = False
        finished_found = False
        for line in combined_output.splitlines():
            if not line.startswith("{"):
                continue
            data = json.loads(line)
            if data.get("event") == "request.received":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                received_found = True
            elif data.get("event") == "request.finished":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                self.assertIn("out", data)
                finished_found = True

        self.assertTrue(received_found, "request.received event not found in logs")
        self.assertTrue(finished_found, "request.finished event not found in logs")

        jsonl_files = glob.glob(os.path.join(self.temp_dir, "*.jsonl"))
        self.assertGreater(len(jsonl_files), 0, "No JSONL files found in temp directory")

        file_received = False
        file_finished = False
        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("event") == "request.received":
                        self.assertIn("rid", data)
                        self.assertIn("obj", data)
                        file_received = True
                    elif data.get("event") == "request.finished":
                        self.assertIn("rid", data)
                        self.assertIn("obj", data)
                        self.assertIn("out", data)
                        file_finished = True

        self.assertTrue(file_received, "request.received event not found in log files")
        self.assertTrue(file_finished, "request.finished event not found in log files")


if __name__ == "__main__":
    unittest.main()
