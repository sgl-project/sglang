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
        cls._temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls._temp_dir_obj.name
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        other_args = [
            "--log-requests",
            "--log-requests-level",
            "2",
            "--log-requests-format",
            cls.log_requests_format,
            "--skip-server-warmup",
            "--log-requests-target",
            "stdout",
            cls.temp_dir,
        ]
        cls.process = popen_launch_server(
            "Qwen/Qwen3-0.6B",
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        cls._temp_dir_obj.cleanup()

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

    def _verify_json_logs(self, lines, source_name):
        received_found = False
        finished_found = False
        for line in lines:
            if not line.strip() or not line.startswith("{"):
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

        self.assertTrue(received_found, f"request.received event not found in {source_name}")
        self.assertTrue(finished_found, f"request.finished event not found in {source_name}")


class TestRequestLoggerText(BaseTestRequestLogger, CustomTestCase):
    log_requests_format = "text"

    def test_text_format_logging(self):
        combined_output = self._send_request()
        time.sleep(1)

        self.assertIn("Receive:", combined_output)
        self.assertIn("Finish:", combined_output)

        log_files = glob.glob(os.path.join(self.temp_dir, "*.log"))
        self.assertGreater(len(log_files), 0, "No log files found in temp directory")

        file_content = ""
        for log_file in log_files:
            with open(log_file, "r") as f:
                file_content += f.read()

        self.assertIn("Receive:", file_content)
        self.assertIn("Finish:", file_content)


class TestRequestLoggerJson(BaseTestRequestLogger, CustomTestCase):
    log_requests_format = "json"

    def test_json_format_logging(self):
        combined_output = self._send_request()
        time.sleep(1)

        self._verify_json_logs(combined_output.splitlines(), "stdout")

        log_files = glob.glob(os.path.join(self.temp_dir, "*.log"))
        self.assertGreater(len(log_files), 0, "No log files found in temp directory")

        file_lines = []
        for log_file in log_files:
            with open(log_file, "r") as f:
                file_lines.extend(f.readlines())

        self._verify_json_logs(file_lines, "log files")


if __name__ == "__main__":
    unittest.main()
