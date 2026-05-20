import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DIFFUSION_TEST_MODEL_NAME = "Efficient-Large-Model/Sana_600M_512px_diffusers"
DIFFUSION_TEST_IMAGE_PROMPT = "A beautiful sunset over mountains, oil painting style"
DIFFUSION_VIDEO_TEST_MODEL_NAME = "IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers"
DIFFUSION_TEST_VIDEO_PROMPT = "A cat playing with a ball"


class BaseTestDiffusionRequestLogger:
    """Base test class for Diffusion model request logging."""

    log_requests_format = None
    model_name = None
    endpoint = None
    prompt = None
    request_body = None
    timeout = 60.0
    verify_fields = None
    env_vars: dict[str, str] = {}

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
            "--log-requests-target",
            "stdout",
            cls.temp_dir,
        ]
        cls._old_env_vars = {}
        for key, value in cls.env_vars.items():
            cls._old_env_vars[key] = os.environ.get(key)
            os.environ[key] = value

        cls.process = popen_launch_server(
            cls.model_name,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.stdout, cls.stderr),
            skip_device_arg=True,  # Diffusion models don't support --device
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        cls._temp_dir_obj.cleanup()
        for key, old_value in cls._old_env_vars.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

    def _verify_logs(self, content: str, source_name: str):
        raise NotImplementedError

    def _wait_until_verified(
        self,
        verify_fn,
        get_content_fn,
        source_name: str,
        timeout: float = 60.0,
        interval: float = 0.5,
    ):
        deadline = time.time() + timeout
        last_error = None
        while time.time() < deadline:
            content = get_content_fn()
            try:
                verify_fn(content, source_name)
                return
            except AssertionError as err:
                last_error = err
                time.sleep(interval)
        if last_error is not None:
            raise last_error

    def test_generation_logging(self):
        """Test that generation requests are logged."""
        response = requests.post(
            DEFAULT_URL_FOR_TEST + self.endpoint,
            json=self.request_body,
            timeout=self.timeout,
        )
        self.assertEqual(response.status_code, 200)
        # Verify logs
        self._wait_until_verified(
            self._verify_logs,
            lambda: self.stdout.getvalue() + self.stderr.getvalue(),
            "stdout",
            timeout=self.timeout,
        )
        self._wait_until_verified(
            self._verify_logs,
            lambda: "".join(f.read_text() for f in Path(self.temp_dir).glob("*.log")),
            "log files",
            timeout=self.timeout,
        )
        log_files = list(Path(self.temp_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0, "No log files found in temp directory")


class TestImageRequestLoggerText(BaseTestDiffusionRequestLogger, CustomTestCase):
    """Test request logging with text format for image models."""

    log_requests_format = "text"
    model_name = DIFFUSION_TEST_MODEL_NAME
    endpoint = "/v1/images/generations"
    prompt = DIFFUSION_TEST_IMAGE_PROMPT
    request_body = {"prompt": DIFFUSION_TEST_IMAGE_PROMPT, "size": "256x256"}
    timeout = 60.0

    def _verify_logs(self, content: str, source_name: str):
        # Verify Receive log contains expected content
        self.assertIn("Receive:", content, f"'Receive:' not found in {source_name}")
        self.assertIn(
            self.prompt[:30],
            content,
            f"Prompt not found in {source_name}",
        )
        # Verify Finish log is present
        self.assertIn("Finish:", content, f"'Finish:' not found in {source_name}")
        # Verify meta_info contains expected fields
        self.assertIn("e2e_latency", content, f"e2e_latency not found in {source_name}")


class TestImageRequestLoggerJson(BaseTestDiffusionRequestLogger, CustomTestCase):
    """Test request logging with JSON format for image models."""

    log_requests_format = "json"
    model_name = DIFFUSION_TEST_MODEL_NAME
    endpoint = "/v1/images/generations"
    prompt = DIFFUSION_TEST_IMAGE_PROMPT
    request_body = {"prompt": DIFFUSION_TEST_IMAGE_PROMPT, "size": "256x256"}
    timeout = 60.0

    def _verify_logs(self, content: str, source_name: str):
        received_found = False
        finished_found = False
        for line in content.splitlines():
            if not line.strip() or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = data.get("rid", "")
            if rid.startswith(HEALTH_CHECK_RID_PREFIX):
                continue
            if data.get("event") == "request.received":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                received_found = True
            elif data.get("event") == "request.finished":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                self.assertIn("out", data)
                # Verify meta_info contains expected fields
                out = data.get("out", {})
                meta_info = out.get("meta_info", {})
                self.assertIn("e2e_latency", meta_info)
                finished_found = True

        self.assertTrue(
            received_found, f"request.received event not found in {source_name}"
        )
        self.assertTrue(
            finished_found, f"request.finished event not found in {source_name}"
        )


class TestVideoRequestLoggerText(BaseTestDiffusionRequestLogger, CustomTestCase):
    """Test request logging with text format for Diffusion video models."""

    log_requests_format = "text"
    model_name = DIFFUSION_VIDEO_TEST_MODEL_NAME
    endpoint = "/v1/videos"
    prompt = DIFFUSION_TEST_VIDEO_PROMPT
    request_body = {"prompt": DIFFUSION_TEST_VIDEO_PROMPT, "size": "832x480"}
    timeout = 120.0

    def _verify_logs(self, content: str, source_name: str):
        # Verify Receive log contains expected content
        self.assertIn("Receive:", content, f"'Receive:' not found in {source_name}")
        self.assertIn(
            self.prompt[:20],
            content,
            f"Prompt not found in {source_name}",
        )
        # Verify Finish log is present
        self.assertIn("Finish:", content, f"'Finish:' not found in {source_name}")
        # Verify meta_info contains expected fields
        self.assertIn("e2e_latency", content, f"e2e_latency not found in {source_name}")
        # Video-specific fields
        self.assertIn("num_frames", content, f"num_frames not found in {source_name}")


class TestVideoRequestLoggerJson(BaseTestDiffusionRequestLogger, CustomTestCase):
    """Test request logging with JSON format for Diffusion video models."""

    log_requests_format = "json"
    model_name = DIFFUSION_VIDEO_TEST_MODEL_NAME
    endpoint = "/v1/videos"
    prompt = DIFFUSION_TEST_VIDEO_PROMPT
    request_body = {"prompt": DIFFUSION_TEST_VIDEO_PROMPT, "size": "832x480"}
    timeout = 120.0

    def _verify_logs(self, content: str, source_name: str):
        received_found = False
        finished_found = False
        for line in content.splitlines():
            if not line.strip() or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = data.get("rid", "")
            if rid.startswith(HEALTH_CHECK_RID_PREFIX):
                continue
            if data.get("event") == "request.received":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                received_found = True
            elif data.get("event") == "request.finished":
                self.assertIn("rid", data)
                self.assertIn("obj", data)
                self.assertIn("out", data)
                # Verify meta_info contains expected fields
                out = data.get("out", {})
                meta_info = out.get("meta_info", {})
                self.assertIn("e2e_latency", meta_info)
                # Video-specific fields
                self.assertIn("num_frames", meta_info)
                finished_found = True

        self.assertTrue(
            received_found, f"request.received event not found in {source_name}"
        )
        self.assertTrue(
            finished_found, f"request.finished event not found in {source_name}"
        )


if __name__ == "__main__":
    unittest.main()
