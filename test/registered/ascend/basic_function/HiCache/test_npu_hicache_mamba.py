"""Ascend NPU HiCache L3 coverage for hybrid Mamba models."""

import os
import shutil
import tempfile
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

TEST_MODEL_MATRIX = {
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B": {
        "target_token_id": 1000,
    },
}


class TestAscendMambaHiCache(CustomTestCase):
    """Exercise Mamba state write-back and L3 load-back on Ascend."""

    # A reusable Mamba checkpoint must be strictly inside the prompt. Ending the
    # prompt exactly at 1024 does not make that boundary reusable by a second
    # identical request. 3200 leaves the 3072 checkpoint inside the prompt and
    # also exercises the configured 1024-token chunked-prefill path.
    prompt_tokens = 3200

    @classmethod
    def setUpClass(cls):
        cls.models = TEST_MODEL_MATRIX.keys()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.common_args = [
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.20",
            "--chunked-prefill-size",
            "1024",
            "--page-size",
            "64",
            "--enable-hierarchical-cache",
            "--enable-cache-report",
            "--hicache-size",
            "1",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            "kernel",
            "--hicache-mem-layout",
            "page_first_direct",
        ]

    def _generate(self, token_id: int) -> dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": [token_id] * self.prompt_tokens,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    @staticmethod
    def _storage_files(storage_dir: str) -> set[str]:
        return {
            os.path.relpath(os.path.join(root, name), storage_dir)
            for root, _, names in os.walk(storage_dir)
            for name in names
            if os.path.getsize(os.path.join(root, name)) > 0
        }

    def _wait_for_mamba_storage_file(
        self, storage_dir: str, files_before: set[str], timeout: float = 30
    ) -> set[str]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            files_after = self._storage_files(storage_dir)
            new_files = files_after - files_before
            if any("mamba" in name.lower() for name in new_files):
                return files_after
            time.sleep(0.1)
        self.fail("Timed out waiting for a new Mamba HiCache storage file.")

    def test_mamba_state_restores_from_l3(self):
        for model in self.models:
            with self.subTest(model=model):
                storage_dir = tempfile.mkdtemp(prefix="npu_mamba_hicache_")
                process = None
                try:
                    process = popen_launch_server(
                        model,
                        self.base_url,
                        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                        other_args=self.common_args,
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": "0",
                            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": storage_dir,
                        },
                    )
                    target_token_id = TEST_MODEL_MATRIX[model]["target_token_id"]
                    storage_files_before = self._storage_files(storage_dir)
                    first = self._generate(target_token_id)
                    first_meta = first["meta_info"]
                    self.assertEqual(int(first_meta.get("cached_tokens", 0)), 0)

                    # The first cache hit reaches write_through's hit-count
                    # threshold and starts the asynchronous L1 -> L2 -> L3 copy.
                    warm = self._generate(target_token_id)
                    warm_details = (
                        warm["meta_info"].get("cached_tokens_details") or {}
                    )
                    self.assertGreater(
                        int(warm_details.get("device", 0) or 0),
                        0,
                        "Expected an L1 device hit; "
                        f"got meta_info={warm['meta_info']}",
                    )
                    self.assertEqual(
                        int(warm_details.get("host", 0) or 0),
                        0,
                        f"Expected no L2 contribution: {warm_details}",
                    )
                    self.assertEqual(
                        int(warm_details.get("storage", 0) or 0),
                        0,
                        f"Expected no L3 contribution: {warm_details}",
                    )

                    # Do not infer persistence from the request response: wait
                    # until the Mamba sidecar has physically reached the file
                    # backend before clearing the in-memory cache levels.
                    storage_files_after = self._wait_for_mamba_storage_file(
                        storage_dir, storage_files_before
                    )
                    self.assertTrue(storage_files_after - storage_files_before)

                    # Remove L1/L2 residency without clearing the file backend.
                    flush = requests.post(
                        f"{self.base_url}/flush_cache",
                        params={"timeout": 30},
                        timeout=40,
                    )
                    flush.raise_for_status()

                    restored = self._generate(target_token_id)
                    restored_meta = restored["meta_info"]
                    details = restored_meta.get("cached_tokens_details") or {}

                    self.assertGreater(
                        int(details.get("storage", 0) or 0),
                        0,
                        "Expected an L3 storage hit; "
                        f"got meta_info={restored_meta}",
                    )
                    self.assertEqual(
                        restored["text"],
                        first["text"],
                        "Mamba L3 restore changed deterministic generation output.",
                    )
                finally:
                    if process:
                        kill_process_tree(process.pid)
                    shutil.rmtree(storage_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
