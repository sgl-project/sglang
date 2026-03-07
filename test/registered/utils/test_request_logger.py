import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=120, suite="nightly-amd-1-gpu", nightly=True)

TEST_ROUTING_KEY = "test-routing-key-12345"
TEST_CUSTOM_HEADER_NAME = "X-Test-Header"
TEST_CUSTOM_HEADER_VALUE = "test-header-value-67890"


class BaseTestRequestLogger:
    log_requests_format = None
    env_vars: dict[str, str] = {}  # Env vars to set before server launch
    request_headers: dict[str, str] = {"X-SMG-Routing-Key": TEST_ROUTING_KEY}

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
        # Set env vars and save old values for restoration
        cls._old_env_vars = {}
        for key, value in cls.env_vars.items():
            cls._old_env_vars[key] = os.environ.get(key)
            os.environ[key] = value

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
        # Restore env vars
        for key, old_value in cls._old_env_vars.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    def _verify_logs(self, content: str, source_name: str):
        raise NotImplementedError

    def test_logging(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            headers=self.request_headers,
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
        self.assertIn(
            TEST_ROUTING_KEY, content, f"Routing key not found in {source_name}"
        )
        self.assertIn(
            "x-smg-routing-key", content, f"Header name not found in {source_name}"
        )


class TestRequestLoggerJson(BaseTestRequestLogger, CustomTestCase):
    log_requests_format = "json"

    def _verify_logs(self, content: str, source_name: str):
        received_found = False
        finished_found = False
        for line in content.splitlines():
            if not line.strip() or not line.startswith("{"):
                continue
            data = json.loads(line)

            rid = data.get("rid", "")
            if rid.startswith("HEALTH_CHECK"):
                continue

            if data.get("event") == "request.received":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                self.assertEqual(
                    data.get("headers", {}).get("x-smg-routing-key"), TEST_ROUTING_KEY
                )
                received_found = True
            elif data.get("event") == "request.finished":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                self.assertIn("out", data)
                self.assertEqual(
                    data.get("headers", {}).get("x-smg-routing-key"), TEST_ROUTING_KEY
                )
                finished_found = True

        self.assertTrue(
            received_found, f"request.received event not found in {source_name}"
        )
        self.assertTrue(
            finished_found, f"request.finished event not found in {source_name}"
        )


class TestCustomHeaderViaEnvVar(BaseTestRequestLogger, CustomTestCase):
    """Test that custom headers can be added via SGLANG_LOG_REQUEST_HEADERS env var."""

    log_requests_format = "text"
    env_vars = {"SGLANG_LOG_REQUEST_HEADERS": TEST_CUSTOM_HEADER_NAME}
    request_headers = {
        "X-SMG-Routing-Key": TEST_ROUTING_KEY,
        TEST_CUSTOM_HEADER_NAME: TEST_CUSTOM_HEADER_VALUE,
    }

    def _verify_logs(self, content: str, source_name: str):
        # Verify custom header is logged
        self.assertIn(
            TEST_CUSTOM_HEADER_NAME.lower(),
            content,
            f"Custom header name not found in {source_name}",
        )
        self.assertIn(
            TEST_CUSTOM_HEADER_VALUE,
            content,
            f"Custom header value not found in {source_name}",
        )
        # Verify default header is still logged (env var appends, not replaces)
        self.assertIn(
            "x-smg-routing-key",
            content,
            f"Default header should still be in whitelist in {source_name}",
        )
        self.assertIn(
            TEST_ROUTING_KEY,
            content,
            f"Default header value not found in {source_name}",
        )


if __name__ == "__main__":
    unittest.main()
