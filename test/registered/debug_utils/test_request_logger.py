import io
import json
import tempfile
import time
import unittest
from pathlib import Path

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

TEST_ROUTING_KEY = "test-routing-key-12345"


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

    def _verify_logs(self, content: str, source_name: str):
        raise NotImplementedError

    def test_logging(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            headers={"X-SMG-Routing-Key": TEST_ROUTING_KEY},
            timeout=30,
        )
        self.assertEqual(response.status_code, 200)
        time.sleep(1)

        stdout_content = self.stdout.getvalue() + self.stderr.getvalue()
        self._verify_logs(stdout_content, "stdout")

        log_files = list(Path(self.temp_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0, "No log files found in temp directory")

        file_content = "".join(f.read_text() for f in log_files)
        self._verify_logs(file_content, "log files")


class TestRequestLoggerText(BaseTestRequestLogger, CustomTestCase):
    log_requests_format = "text"

    def _verify_logs(self, content: str, source_name: str):
        self.assertIn("Receive:", content, f"'Receive:' not found in {source_name}")
        self.assertIn("Finish:", content, f"'Finish:' not found in {source_name}")
        self.assertIn(TEST_ROUTING_KEY, content, f"Routing key not found in {source_name}")
        self.assertIn("x-smg-routing-key", content, f"Header name not found in {source_name}")


class TestRequestLoggerJson(BaseTestRequestLogger, CustomTestCase):
    log_requests_format = "json"

    def _verify_logs(self, content: str, source_name: str):
        received_found = False
        finished_found = False
        header_in_received = False
        header_in_finished = False
        for line in content.splitlines():
            if not line.strip() or not line.startswith("{"):
                continue
            data = json.loads(line)
            if data.get("event") == "request.received":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                received_found = True
                if data.get("headers", {}).get("x-smg-routing-key") == TEST_ROUTING_KEY:
                    header_in_received = True
            elif data.get("event") == "request.finished":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                self.assertIn("out", data)
                finished_found = True
                if data.get("headers", {}).get("x-smg-routing-key") == TEST_ROUTING_KEY:
                    header_in_finished = True

        self.assertTrue(
            received_found, f"request.received event not found in {source_name}"
        )
        self.assertTrue(
            finished_found, f"request.finished event not found in {source_name}"
        )
        self.assertTrue(
            header_in_received, f"Header not in request.received for {source_name}"
        )
        self.assertTrue(
            header_in_finished, f"Header not in request.finished for {source_name}"
        )


if __name__ == "__main__":
    unittest.main()
