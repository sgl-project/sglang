"""Integration test: SGLang server bound to a Unix domain socket.

Verifies that --uds launches successfully, that /v1/models responds, and
that a real /v1/chat/completions request returns a non-empty completion
over the UDS transport.

Usage:
    python3 -m pytest test/registered/openai_server/basic/test_uds_server.py -v
"""

import http.client
import json
import logging
import os
import socket
import sys
import tempfile
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

logger = logging.getLogger(__name__)

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


class _UDSConnection(http.client.HTTPConnection):
    def __init__(self, uds_path: str, timeout: float = 30.0):
        super().__init__("localhost", timeout=timeout)
        self._uds_path = uds_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._uds_path)


@unittest.skipIf(sys.platform == "win32", "UDS not supported on Windows")
class TestUDSServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        # Keep the path short to stay under the SUN_PATH limit (~108 bytes).
        cls._tmpdir = tempfile.mkdtemp(prefix="sglang-uds-")
        cls.uds_path = os.path.join(cls._tmpdir, f"s{os.getpid()}.sock")

        cls.process = popen_launch_server(
            cls.model,
            base_url=None,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            uds_path=cls.uds_path,
        )

    @classmethod
    def tearDownClass(cls):
        try:
            kill_process_tree(cls.process.pid)
        except Exception:
            logger.exception("Failed to kill UDS test server process")
        try:
            os.unlink(cls.uds_path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("Failed to unlink UDS file %s", cls.uds_path)
        try:
            os.rmdir(cls._tmpdir)
        except OSError:
            logger.exception("Failed to remove UDS test tempdir %s", cls._tmpdir)

    def _request_json(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict:
        conn = _UDSConnection(self.uds_path, timeout=120.0)
        try:
            headers = {"Content-Type": "application/json"}
            body = json.dumps(payload).encode("utf-8") if payload is not None else None
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            self.assertEqual(
                resp.status,
                200,
                f"{method} {path} returned {resp.status}: {raw!r}",
            )
            return json.loads(raw.decode("utf-8"))
        finally:
            conn.close()

    def test_v1_models(self):
        body = self._request_json("GET", "/v1/models")
        self.assertIn("data", body)
        self.assertTrue(len(body["data"]) >= 1)

    def test_chat_completion(self):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say hi in one word."}],
            "max_tokens": 8,
            "temperature": 0.0,
        }
        body = self._request_json("POST", "/v1/chat/completions", payload)
        self.assertIn("choices", body)
        text = body["choices"][0]["message"]["content"]
        self.assertTrue(len(text.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
