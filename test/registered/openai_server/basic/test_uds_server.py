"""Integration test: SGLang server bound to a Unix domain socket.

Verifies that --uds launches successfully, that /v1/models responds, and
that a real /v1/chat/completions request returns a non-empty completion
over the UDS transport.

Usage:
    python3 -m pytest test/registered/openai_server/basic/test_uds_server.py -v
"""

import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_with_error_check,
)

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


class _UDSConnection(http.client.HTTPConnection):
    """http.client connection that talks over a Unix domain socket."""

    def __init__(self, uds_path: str, timeout: float = 30.0):
        super().__init__("localhost", timeout=timeout)
        self._uds_path = uds_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._uds_path)


def _wait_for_uds_health(
    uds_path: str, process: subprocess.Popen, timeout: float
) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Server process exited with code {return_code} before becoming "
                f"healthy on {uds_path}; last probe error: {last_err}"
            )
        conn = _UDSConnection(uds_path, timeout=5.0)
        try:
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            if resp.status == 200:
                return
        except OSError as e:
            last_err = e
        finally:
            conn.close()
        time.sleep(1.0)
    raise TimeoutError(
        f"Server did not become healthy on {uds_path} within {timeout}s: " f"{last_err}"
    )


@unittest.skipIf(sys.platform == "win32", "UDS not supported on Windows")
class TestUDSServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        # Keep the path short to stay under the SUN_PATH limit (~108 bytes).
        tmpdir = tempfile.mkdtemp(prefix="sglang-uds-")
        cls.uds_path = os.path.join(tmpdir, f"s{os.getpid()}.sock")
        cls._tmpdir = tmpdir

        command = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            "--uds",
            cls.uds_path,
        ]

        # Mirror the offline-mode optimization that popen_launch_server applies:
        # on CI runners with the model cache pre-populated, set HF_HUB_OFFLINE=1
        # so the subprocess does not hit the HuggingFace Hub network. We mutate
        # os.environ (which Popen inherits) and remember which keys to restore.
        cls._restore_env: dict[str, str | None] = {}
        try:
            from sglang.utils import is_in_ci

            if is_in_ci():
                from sglang.test.test_utils import (
                    _try_enable_offline_mode_if_cache_complete,
                )

                proxy_env: dict[str, str] = {}
                _try_enable_offline_mode_if_cache_complete(
                    cls.model, proxy_env, command[1:]
                )
                for key, value in proxy_env.items():
                    cls._restore_env[key] = os.environ.get(key)
                    os.environ[key] = value
        except Exception as e:
            # Non-fatal: live HF fetch will still work; we just lose the offline
            # optimization for this test run.
            print(f"UDS test CI cache validation failed (non-fatal): {e}")

        cls.process = popen_with_error_check(command)
        _wait_for_uds_health(
            cls.uds_path, cls.process, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        # Restore environment variables we mutated in setUpClass.
        for key, original in cls._restore_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        try:
            os.unlink(cls.uds_path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(cls._tmpdir)
        except OSError:
            pass

    def _request_json(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict:
        conn = _UDSConnection(self.uds_path, timeout=120.0)
        headers = {"Content-Type": "application/json"}
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        self.assertEqual(
            resp.status,
            200,
            f"{method} {path} returned {resp.status}: {raw!r}",
        )
        return json.loads(raw.decode("utf-8"))

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
