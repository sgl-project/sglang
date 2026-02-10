"""
E2E tests for HiCache storage fault tolerance with fake backend.
"""

import json
import os
import tempfile
import time
from urllib import error, request

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

register_cuda_ci(est_time=200, suite="stage-b-test-large-2-gpu")


class _BaseFaultToleranceTest(CustomTestCase):
    @classmethod
    def _wait_for_server_ready(cls, base_url: str, timeout: int = 60) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                code, _body = cls._http_get(f"{base_url}/health", timeout=5)
                if code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError("Server failed to start within timeout")

    @staticmethod
    def _http_get(url: str, timeout: int = 10, headers: dict | None = None):
        try:
            req = request.Request(url, headers=headers or {}, method="GET")
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    @staticmethod
    def _http_put_json_with_headers(
        url: str,
        payload: dict | None = None,
        timeout: int = 30,
        headers: dict | None = None,
    ):
        data = None
        all_headers = dict(headers or {})
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            all_headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=all_headers, method="PUT")
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    @staticmethod
    def _attach_backend(
        base_url: str,
        backend: str,
        extra_cfg: dict,
        prefetch_policy: str = "timeout",
        write_policy: str = "write_through",
        headers: dict | None = None,
    ):
        payload = {
            "hicache_storage_backend": backend,
            "hicache_storage_backend_extra_config_json": json.dumps(extra_cfg),
            "hicache_storage_prefetch_policy": prefetch_policy,
            "hicache_write_policy": write_policy,
        }
        return _BaseFaultToleranceTest._http_put_json_with_headers(
            f"{base_url}/hicache/storage-backend",
            payload,
            timeout=30,
            headers=headers,
        )

    @staticmethod
    def _read_json(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path, "r") as fin:
            return json.load(fin)

    @staticmethod
    def _write_json(path: str, data: dict):
        with open(path, "w") as fout:
            json.dump(data, fout)

    def _call_generate(self, base_url: str):
        requests.post(
            base_url + "/generate",
            json={
                "text": "Hello world. Please repeat: hello world.",
                "sampling_params": {"max_new_tokens": 32, "temperature": 0},
            },
            timeout=30,
        )


class TestHiCacheFaultAutoDetach(_BaseFaultToleranceTest):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"
        cls.admin_key = "sglang-test-admin-key"

        cls.fault_inject_path = os.path.join(cls.temp_dir, "fault_inject.json")
        cls.stats_path = os.path.join(cls.temp_dir, "stats.json")
        cls.storage_dir = os.path.join(cls.temp_dir, "hicache_file")
        cls._write_json(cls.fault_inject_path, {"fail_mode": ""})

        cls.env = os.environ.copy()
        cls.env["SGLANG_HICACHE_FAULT_INJECT_PATH"] = cls.fault_inject_path
        cls.env["SGLANG_HICACHE_FAULT_STATS_PATH"] = cls.stats_path
        cls.env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.storage_dir

        cls.other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--hicache-size",
            "100",
            "--page-size",
            "64",
            "--admin-api-key",
            cls.admin_key,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env=cls.env,
        )
        cls._wait_for_server_ready(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(2)

    def test_auto_detach_on_fault(self):
        headers = {"Authorization": f"Bearer {self.admin_key}"}
        extra_cfg = {
            "prefetch_threshold": 1,
            "prefetch_timeout_base": 1,
            "prefetch_timeout_per_ki_token": 0.01,
            "fault_tolerance": {"level": "auto_detach"},
        }
        code, body = self._attach_backend(
            self.base_url,
            "file",
            extra_cfg,
            prefetch_policy="timeout",
            write_policy="write_through",
            headers=headers,
        )
        self.assertEqual(code, 200, body)

        self._call_generate(self.base_url)
        stats_before = self._read_json(self.stats_path)
        self.assertTrue(stats_before, f"stats missing: {stats_before}")

        self._write_json(self.fault_inject_path, {"fail_mode": "get"})
        self._call_generate(self.base_url)

        # wait for auto detach
        detached = False
        for _ in range(30):
            code_info, body_info = self._http_get(
                f"{self.base_url}/hicache/storage-backend",
                timeout=10,
                headers=headers,
            )
            if (
                code_info == 200
                and json.loads(body_info).get("hicache_storage_backend") is None
            ):
                detached = True
                break
            time.sleep(1)
        self.assertTrue(detached, "auto detach did not happen")

        stats_detached = self._read_json(self.stats_path)
        self._call_generate(self.base_url)
        stats_after = self._read_json(self.stats_path)
        self.assertEqual(stats_detached, stats_after)


class TestHiCacheFaultAutoReconnect(_BaseFaultToleranceTest):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1]) + 1
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"
        cls.admin_key = "sglang-test-admin-key"

        cls.fault_inject_path = os.path.join(cls.temp_dir, "fault_inject.json")
        cls.stats_path = os.path.join(cls.temp_dir, "stats.json")
        cls.storage_dir = os.path.join(cls.temp_dir, "hicache_file")
        cls._write_json(cls.fault_inject_path, {"fail_mode": ""})

        cls.env = os.environ.copy()
        cls.env["SGLANG_HICACHE_FAULT_INJECT_PATH"] = cls.fault_inject_path
        cls.env["SGLANG_HICACHE_FAULT_STATS_PATH"] = cls.stats_path
        cls.env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.storage_dir

        cls.other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--hicache-size",
            "100",
            "--page-size",
            "64",
            "--admin-api-key",
            cls.admin_key,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env=cls.env,
        )
        cls._wait_for_server_ready(cls.base_url)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        time.sleep(2)

    def test_auto_reconnect_after_recovery(self):
        headers = {"Authorization": f"Bearer {self.admin_key}"}
        extra_cfg = {
            "prefetch_threshold": 1,
            "prefetch_timeout_base": 1,
            "prefetch_timeout_per_ki_token": 0.01,
            "fault_tolerance": {
                "level": "auto_reconnect",
                "backoff_initial_s": 1,
                "backoff_max_s": 2,
            },
        }
        code, body = self._attach_backend(
            self.base_url,
            "file",
            extra_cfg,
            prefetch_policy="timeout",
            write_policy="write_through",
            headers=headers,
        )
        self.assertEqual(code, 200, body)

        self._write_json(self.fault_inject_path, {"fail_mode": "get"})
        self._call_generate(self.base_url)

        # wait for auto detach
        for _ in range(30):
            code_info, body_info = self._http_get(
                f"{self.base_url}/hicache/storage-backend",
                timeout=10,
                headers=headers,
            )
            if (
                code_info == 200
                and json.loads(body_info).get("hicache_storage_backend") is None
            ):
                break
            time.sleep(1)

        # recover and wait for auto reconnect
        self._write_json(self.fault_inject_path, {"fail_mode": ""})
        reattached = False
        for _ in range(30):
            code_info, body_info = self._http_get(
                f"{self.base_url}/hicache/storage-backend",
                timeout=10,
                headers=headers,
            )
            if (
                code_info == 200
                and json.loads(body_info).get("hicache_storage_backend") == "file"
            ):
                reattached = True
                break
            time.sleep(1)
        self.assertTrue(reattached, "auto reconnect did not happen")

        stats_before = self._read_json(self.stats_path)
        self._call_generate(self.base_url)
        stats_after = self._read_json(self.stats_path)
        self.assertNotEqual(stats_before, stats_after)
