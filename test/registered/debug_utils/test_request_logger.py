import io
import json
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


class TestRequestLoggerJson(BaseTestRequestLogger, CustomTestCase):
    log_requests_format = "json"

    def test_json_format_logging(self):
        combined_output = self._send_request()

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


if __name__ == "__main__":
    unittest.main()
