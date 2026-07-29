"""
E2E check for HiCache storage runtime attach/detach on UnifiedRadixCache.

This is the UnifiedRadixCache counterpart of
test_hicache_storage_runtime_attach_detach.py (which covers HiRadixCache and
the admin auth handling). It launches a server with hierarchical cache and
the unified radix tree enabled but WITHOUT a storage backend at startup, then
drives the full runtime lifecycle via the HTTP admin endpoints:

    attach -> L3 backup/prefetch round-trip -> policy update -> detach
    (idempotent) -> re-attach -> second round-trip

The file storage backend writes one file per KV page into
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR, which the test uses as ground truth
that the runtime-attached backend actually receives traffic.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_storage_runtime_attach_detach_unified.py -v
"""

import json
import os
import random
import shutil
import tempfile
import time
import unittest

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

ADMIN_KEY = "sglang-test-admin-key"
PROMPT_TOKENS = 768


class TestUnifiedRadixCacheStorageRuntimeAttachDetach(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = get_tokenizer(cls.model)
        cls.admin_headers = {"Authorization": f"Bearer {ADMIN_KEY}"}

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                "0.5",
                "--hicache-size",
                "2",
                "--page-size",
                "64",
                "--enable-cache-report",
                # Exercises the storage metrics collector reuse on re-attach
                # (a duplicate Prometheus registration would fail the attach).
                "--enable-metrics",
                "--admin-api-key",
                ADMIN_KEY,
                # NOTE: no --hicache-storage-backend* at startup; the whole
                # point is attaching it at runtime.
            ],
            env={
                **os.environ,
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
                "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
            },
        )
        wait_for_http_ready(url=f"{cls.base_url}/health", process=cls.process)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # ---- helpers ----

    def _retry_while_not_idle(self, send_request, timeout: float = 20.0):
        """Retry an attach/detach call while the scheduler reports "not idle".

        Right after traffic the scheduler may briefly hold in-flight HiCache
        write-through/backup acks; they are drained one event-loop iteration
        later (each retry request itself pumps the loop). This mirrors the
        retry guidance in the runtime attach/detach docs.
        """
        deadline = time.monotonic() + timeout
        while True:
            resp = send_request()
            if (
                resp.status_code != 400
                or "not idle" not in resp.text
                or time.monotonic() >= deadline
            ):
                return resp
            time.sleep(0.5)

    def _attach(self, backend: str, prefetch_policy: str, write_policy: str):
        payload = {
            "hicache_storage_backend": backend,
            "hicache_storage_backend_extra_config_json": json.dumps(
                {"prefetch_threshold": 256}
            ),
            "hicache_storage_prefetch_policy": prefetch_policy,
            "hicache_write_policy": write_policy,
        }
        return self._retry_while_not_idle(
            lambda: requests.put(
                f"{self.base_url}/hicache/storage-backend",
                json=payload,
                headers=self.admin_headers,
                timeout=30,
            )
        )

    def _detach(self):
        return self._retry_while_not_idle(
            lambda: requests.delete(
                f"{self.base_url}/hicache/storage-backend",
                headers=self.admin_headers,
                timeout=30,
            )
        )

    def _status(self) -> dict:
        resp = requests.get(
            f"{self.base_url}/hicache/storage-backend",
            headers=self.admin_headers,
            timeout=10,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _generate(self, prompt: str, max_tokens: int = 64) -> dict:
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _flush_cache(self):
        resp = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 30},
            headers=self.admin_headers,
            timeout=40,
        )
        resp.raise_for_status()

    def _gen_prompt(self, token_num: int) -> str:
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def _storage_file_count(self) -> int:
        return sum(len(files) for _, _, files in os.walk(self.temp_dir))

    def _wait_for_storage_files(self, min_count: int, timeout: float = 30.0) -> int:
        """Wait until the file backend has persisted at least min_count pages.

        Storage backup runs asynchronously after the host write-through; tiny
        filler requests keep the scheduler loop pumping its ack queues.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            count = self._storage_file_count()
            if count >= min_count:
                return count
            self._generate(self._gen_prompt(1), max_tokens=8)
            time.sleep(1)
        return self._storage_file_count()

    def _assert_storage_round_trip(self, label: str):
        """Populate the cache, force everything out of GPU+host, and verify the
        prefix comes back from the runtime-attached file backend."""
        prompt = self._gen_prompt(PROMPT_TOKENS)
        self._generate(prompt)

        persisted = self._wait_for_storage_files(min_count=PROMPT_TOKENS // 64)
        self.assertGreater(persisted, 0, f"{label}: no pages persisted to storage")

        # flush_cache resets both the device tree and the host pool, so any
        # cache hit afterwards must have been prefetched from L3 storage.
        self._flush_cache()
        response = self._generate(prompt)
        cached_tokens = int(response["meta_info"].get("cached_tokens", 0))
        self.assertGreaterEqual(
            cached_tokens,
            512,
            f"{label}: expected a large L3 prefix hit, got cached_tokens={cached_tokens}",
        )

    # ---- test ----

    def test_runtime_attach_detach_lifecycle(self):
        # 0) Detach before any attach is a clean no-op. The scheduler maps
        #    tree-level failures to HTTP 200 when storage was never enabled,
        #    so also assert the message reports no teardown failure.
        resp = self._detach()
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertNotIn("Failed", resp.text)

        # 0b) Invalid policies are rejected up front, without side effects.
        resp = self._attach(
            "file", prefetch_policy="bogus_policy", write_policy="write_through"
        )
        self.assertEqual(resp.status_code, 400, resp.text)

        # 1) Initially no storage backend; generation runs on L1/L2 only and
        #    nothing is persisted to the (unattached) file backend directory.
        self.assertIsNone(self._status().get("hicache_storage_backend"))
        self._generate(self._gen_prompt(PROMPT_TOKENS))
        time.sleep(2)
        self.assertEqual(
            self._storage_file_count(),
            0,
            "storage dir must stay empty before attach",
        )

        # 2) Runtime attach succeeds and traffic reaches the backend (L3
        #    backup + prefetch round-trip).
        resp = self._attach(
            "file", prefetch_policy="wait_complete", write_policy="write_through"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        status = self._status()
        self.assertEqual(status.get("hicache_storage_backend"), "file")
        self.assertEqual(status.get("hicache_storage_prefetch_policy"), "wait_complete")
        self.assertEqual(status.get("hicache_write_policy"), "write_through")
        self._assert_storage_round_trip("first attach")

        # 3) Re-attach with the same backend only updates policies.
        resp = self._attach(
            "file", prefetch_policy="timeout", write_policy="write_through_selective"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        status = self._status()
        self.assertEqual(status.get("hicache_storage_prefetch_policy"), "timeout")
        self.assertEqual(status.get("hicache_write_policy"), "write_through_selective")

        # 4) Attaching a different backend while one is active is rejected.
        resp = self._attach(
            "mooncake", prefetch_policy="timeout", write_policy="write_through"
        )
        self.assertNotEqual(resp.status_code, 200, resp.text)

        # 5) Detach succeeds and is idempotent.
        resp = self._detach()
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIsNone(self._status().get("hicache_storage_backend"))
        resp = self._detach()
        self.assertEqual(resp.status_code, 200, resp.text)

        # 6) The server keeps serving after detach, without touching storage.
        baseline = self._storage_file_count()
        self._generate(self._gen_prompt(PROMPT_TOKENS))
        time.sleep(2)
        self.assertEqual(
            self._storage_file_count(),
            baseline,
            "storage dir must not grow after detach",
        )

        # 7) Re-attach after detach works end-to-end again (fresh storage
        #    threads, reused metrics state).
        resp = self._attach(
            "file", prefetch_policy="wait_complete", write_policy="write_through"
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        self._assert_storage_round_trip("re-attach")

        # Leave the server clean for other tests.
        resp = self._detach()
        self.assertEqual(resp.status_code, 200, resp.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
