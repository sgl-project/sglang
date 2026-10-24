import json
import os
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

DSV4_FLASH_MODEL = os.environ.get(
    "SGLANG_LINKER_DSV4_FLASH_MODEL", "sgl-project/DeepSeek-V4-Flash-FP8"
)
ADMIN_KEY = "sglang-test-admin-key"
PAGE_SIZE = 256
EXTRA_CONFIG = json.dumps({"enable_group_semantics": True})

register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-h100")


class TestUnifiedCacheLinkerRuntimeAttach(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        cls.mooncake = MooncakeTestServices()
        cls.mooncake.start()
        cls.process = None
        try:
            cls.process = popen_launch_server(
                DSV4_FLASH_MODEL,
                cls.base_url,
                timeout=3600,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size",
                    "4",
                    "--attention-backend",
                    "compressed",
                    "--page-size",
                    str(PAGE_SIZE),
                    "--mem-fraction-static",
                    "0.92",
                    "--disable-shared-experts-fusion",
                    "--swa-full-tokens-ratio",
                    "0.25",
                    "--max-total-tokens",
                    "8192",
                    "--max-running-requests",
                    "1",
                    "--enable-cache-report",
                    "--enable-unified-cache-external-linker",
                    "--hicache-storage-backend-extra-config",
                    EXTRA_CONFIG,
                    "--admin-api-key",
                    ADMIN_KEY,
                ],
                env={
                    **os.environ,
                    **cls.mooncake.server_env(),
                    "SGLANG_DSV4_FP4_EXPERTS": "0",
                },
            )
        except Exception:
            cls.mooncake.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.process is not None:
                terminate_and_kill_process_tree(cls.process)
        finally:
            cls.mooncake.stop()

    def _request(self, method, path, payload=None):
        return requests.request(
            method,
            f"{self.base_url}{path}",
            json=payload,
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            timeout=300,
        )

    def _loaded_tokens(self, input_ids, flush=True):
        if flush:
            self._request("POST", "/flush_cache").raise_for_status()
        response = self._request(
            "POST",
            "/generate",
            {
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        response.raise_for_status()
        details = response.json()["meta_info"].get("cached_tokens_details") or {}
        return int(details.get("host", 0))

    def test_detach_and_reattach(self):
        num_tokens = 8 * PAGE_SIZE
        stored_ids = [(i * 37) % 30000 + 100 for i in range(num_tokens)]
        detached_ids = [(i * 53) % 30000 + 100 for i in range(num_tokens)]
        min_loaded = num_tokens - PAGE_SIZE

        self._loaded_tokens(stored_ids)
        self.assertGreaterEqual(self._loaded_tokens(stored_ids), min_loaded)

        response = self._request("DELETE", "/hicache/storage-backend")
        self.assertEqual(response.status_code, 200, response.text)
        status = self._request("GET", "/hicache/storage-backend").json()
        self.assertIsNone(status["unified_cache_external_linker_backend"])
        self.assertEqual(self._loaded_tokens(stored_ids), 0)
        self._loaded_tokens(detached_ids)

        attach = {
            "hicache_storage_backend": "mooncake",
            "hicache_storage_backend_extra_config_json": EXTRA_CONFIG,
        }
        response = self._request(
            "PUT",
            "/hicache/storage-backend",
            {**attach, "hicache_write_policy": "write_back"},
        )
        self.assertEqual(response.status_code, 400, response.text)
        response = self._request("PUT", "/hicache/storage-backend", attach)
        self.assertEqual(response.status_code, 200, response.text)
        status = self._request("GET", "/hicache/storage-backend").json()
        self.assertEqual(status["unified_cache_external_linker_backend"], "mooncake")

        self._loaded_tokens(detached_ids, flush=False)
        self.assertGreaterEqual(self._loaded_tokens(detached_ids), min_loaded)
        self.assertGreaterEqual(self._loaded_tokens(stored_ids), min_loaded)


if __name__ == "__main__":
    unittest.main()
