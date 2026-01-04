"""
E2E smoke test for HiCache storage runtime attach/detach.

This test launches an SGLang server with hierarchical cache enabled but WITHOUT
any storage backend at startup, then attaches/detaches a storage backend via the
HTTP endpoints.

Usage:
    python3 -m pytest test/srt/hicache/test_hicache_storage_runtime_attach_detach.py -v
"""

import json
import os
import tempfile
import time
import unittest
from urllib import error, request

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestHiCacheStorageRuntimeAttachDetach(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Launch server with hierarchical cache enabled but storage backend disabled.
        other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--hicache-size",
            "100",
            "--page-size",
            "64",
            "--enable-cache-report",
            # NOTE: do NOT pass --hicache-storage-backend* here
        ]

        env = {
            **os.environ,
            # File backend uses this env var to decide where to store cache pages.
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            # Make runs less flaky for CI/dev.
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
        }

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )
        cls._wait_for_server_ready()

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(2)

        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _wait_for_server_ready(cls, timeout: int = 60) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                code, _body = cls._http_get(f"{cls.base_url}/health", timeout=5)
                if code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError("Server failed to start within timeout")

    @staticmethod
    def _http_get(url: str, timeout: int = 10):
        try:
            with request.urlopen(url, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    @staticmethod
    def _http_post_json(url: str, payload: dict | None = None, timeout: int = 30):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    def _get_backend_status(self):
        code, body = self._http_get(
            f"{self.base_url}/info_from_hicache_storage_backend", timeout=10
        )
        self.assertEqual(code, 200, body)
        return json.loads(body)

    def _attach_file_backend(self, extra_cfg: dict):
        payload = {
            "hicache_storage_backend": "file",
            "hicache_storage_backend_extra_config_json": json.dumps(extra_cfg),
            "hicache_storage_prefetch_policy": "timeout",
        }
        return self._http_post_json(
            f"{self.base_url}/attach_hicache_storage_backend", payload, timeout=30
        )

    def _detach_backend(self):
        return self._http_post_json(
            f"{self.base_url}/detach_hicache_storage_backend", None, timeout=30
        )

    def test_runtime_attach_detach(self):
        # 1) Initially disabled
        status0 = self._get_backend_status()
        self.assertIsNone(status0.get("hicache_storage_backend"))

        # 2) Attach should succeed when idle
        extra_cfg = {
            "hicache_storage_pass_prefix_keys": True,
            # keep knobs small and stable
            "prefetch_threshold": 256,
            "prefetch_timeout_base": 3,
            "prefetch_timeout_per_ki_token": 0.01,
        }
        code_attach, body_attach = self._attach_file_backend(extra_cfg)
        self.assertEqual(code_attach, 200, f"{code_attach} - {body_attach}")

        status1 = self._get_backend_status()
        self.assertEqual(status1.get("hicache_storage_backend"), "file")
        self.assertEqual(
            status1.get("hicache_storage_backend_extra_config"),
            json.dumps(extra_cfg),
        )
        self.assertEqual(status1.get("hicache_storage_prefetch_policy"), "timeout")

        # 3) Attach again should be rejected (already enabled)
        code_attach_again, body_attach_again = self._attach_file_backend(extra_cfg)
        self.assertNotEqual(code_attach_again, 200, body_attach_again)

        # 4) Detach should succeed and be idempotent
        code_detach, body_detach = self._detach_backend()
        self.assertEqual(code_detach, 200, f"{code_detach} - {body_detach}")
        status2 = self._get_backend_status()
        self.assertIsNone(status2.get("hicache_storage_backend"))

        code_detach_again, body_detach_again = self._detach_backend()
        self.assertEqual(
            code_detach_again,
            200,
            f"{code_detach_again} - {body_detach_again}",
        )

        # 5) Re-attach after detach should succeed
        code_attach2, body_attach2 = self._attach_file_backend(extra_cfg)
        self.assertEqual(code_attach2, 200, f"{code_attach2} - {body_attach2}")
        status3 = self._get_backend_status()
        self.assertEqual(status3.get("hicache_storage_backend"), "file")

        # Cleanup: detach for test isolation
        code_detach2, body_detach2 = self._detach_backend()
        self.assertEqual(code_detach2, 200, f"{code_detach2} - {body_detach2}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
