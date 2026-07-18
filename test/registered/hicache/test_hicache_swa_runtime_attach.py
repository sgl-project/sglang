"""
E2E regression check for HiCache on a hybrid-SWA model (Gemma-style).

Hybrid-SWA models (sliding-window + full attention, e.g. Gemma-3/4, gpt-oss)
are forced onto the ``UnifiedRadixCache`` when hierarchical cache is enabled.
This test guards two regressions that shipped because no CI exercised that path:

1. Launching a hybrid-SWA model with ``--enable-hierarchical-cache`` and a fixed
   ``--hicache-size`` must succeed (the full + SWA host pools together consume
   the requested budget instead of each allocating it, which used to 2x host
   memory and hang cudaHostRegister at larger sizes).

2. Runtime attach of an L3 storage backend via ``PUT /hicache/storage-backend``
   must work. It previously failed with
   "UnifiedRadixCache does not support runtime HiCache storage attach yet."

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_swa_runtime_attach.py -v
"""

import json
import os
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

register_cuda_ci(est_time=320, stage="extra-a", runner_config="1-gpu-large")

# gpt-oss-20b is a non-gated hybrid-SWA model (sliding-window on alternating
# layers), already used by the scripted SWA CI test.
_SWA_MODEL = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE

_ADMIN_KEY = "sglang-test-admin-key"


class TestHiCacheSWARuntimeAttach(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = _SWA_MODEL
        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"

        cls.other_args = [
            "--enable-hierarchical-cache",
            # A fixed (GB) host budget exercises the full+SWA split; kept small
            # so the pool fits comfortably and would NOT fit if it doubled.
            "--hicache-size",
            "20",
            "--hicache-mem-layout",
            "page_first",
            "--hicache-io-backend",
            "kernel",
            "--page-size",
            "64",
            "--mem-fraction-static",
            "0.7",
            "--enable-cache-report",
            "--admin-api-key",
            _ADMIN_KEY,
            # NOTE: no --hicache-storage-backend* here; it is attached at runtime.
        ]
        cls.env = {
            **os.environ,
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }
        cls.admin_headers = {"Authorization": f"Bearer {_ADMIN_KEY}"}

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env=cls.env,
        )
        wait_for_http_ready(
            url=f"{cls.base_url}/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process,
        )

    @classmethod
    def tearDownClass(cls):
        import shutil

        kill_process_tree(cls.process.pid)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _get_status(self):
        resp = requests.get(
            f"{self.base_url}/hicache/storage-backend",
            headers=self.admin_headers,
            timeout=10,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _put(self, payload):
        return requests.put(
            f"{self.base_url}/hicache/storage-backend",
            json=payload,
            headers=self.admin_headers,
            timeout=30,
        )

    def _delete(self):
        return requests.delete(
            f"{self.base_url}/hicache/storage-backend",
            headers=self.admin_headers,
            timeout=30,
        )

    def _generate(self, text):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_swa_runtime_attach_detach(self):
        # The server launched with hierarchical cache + a fixed --hicache-size on
        # a hybrid-SWA model. Reaching a ready /health already guards regression
        # #1 (no double-alloc hang during host pool construction).

        # Storage backend starts detached.
        self.assertIsNone(self._get_status().get("hicache_storage_backend"))

        # Warm the cache before storage is attached.
        self._generate("The capital of France is")

        # Regression #2: runtime attach must succeed on UnifiedRadixCache.
        resp = self._put(
            {
                "hicache_storage_backend": "file",
                "hicache_storage_backend_extra_config_json": json.dumps(
                    {"prefetch_threshold": 256}
                ),
                "hicache_storage_prefetch_policy": "timeout",
                "hicache_write_policy": "write_through",
            }
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        status = self._get_status()
        self.assertEqual(status.get("hicache_storage_backend"), "file")
        self.assertEqual(status.get("hicache_write_policy"), "write_through")

        # Generation must keep working with the backend attached (backup path).
        self._generate("The capital of France is")
        self._generate("Once upon a time in a distant galaxy")

        # Detach must succeed and return the backend to a clean state.
        resp = self._delete()
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIsNone(self._get_status().get("hicache_storage_backend"))

        # Re-attach after detach must succeed (idempotent lifecycle).
        resp = self._put({"hicache_storage_backend": "file"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self._get_status().get("hicache_storage_backend"), "file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
