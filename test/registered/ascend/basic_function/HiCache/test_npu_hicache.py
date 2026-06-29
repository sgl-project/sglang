import logging
import os
import shutil
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestHiCache(CustomTestCase):
    """Test Hierarchical Cache functionality on NPU environment.

    [Test Category] Functional
    [Test Target] Hierarchical Cache on NPU
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = None

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)

    def _launch_server_with_hicache(
        self,
        hicache_ratio=2.0,
        hicache_size=0,
        hicache_write_policy="write_through",
        radix_eviction_policy="lru",
        hicache_io_backend="direct",
        hicache_mem_layout="page_first_direct",
        hicache_storage_backend=None,
    ):
        """Launch server with hierarchical cache parameters."""
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            str(hicache_ratio),
            "--hicache-write-policy",
            hicache_write_policy,
            "--radix-eviction-policy",
            radix_eviction_policy,
            "--hicache-io-backend",
            hicache_io_backend,
            "--hicache-mem-layout",
            hicache_mem_layout,
        ]

        if hicache_size > 0:
            other_args.extend(
                [
                    "--hicache-size",
                    str(hicache_size),
                ]
            )

        if hicache_storage_backend is not None:
            other_args.extend(
                [
                    "--hicache-storage-backend",
                    hicache_storage_backend,
                ]
            )

        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        return process

    def test_001_combined_params(self):
        """Test Hicache with combined parameters, hicache inference request reuse successfully."""
        logging.warning("\n=== Test 001: Combined Parameters ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=1.0,
            hicache_write_policy="write_back",
            radix_eviction_policy="lru",
            hicache_io_backend="direct",
            hicache_mem_layout="page_first_kv_split",
        )

        try:
            prompt = (
                "What is The capital of France?What is The capital of France?What is The capital of France?"
                * 18
            )
            for i in range(2):
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 10,
                        },
                    },
                )
                self.assertEqual(response.status_code, 200)
                # If the same request is made, the token will be reused.
                # cached_tokens: Number of tokens cached in KV Cache
                if i == 0:
                    self.assertEqual(
                        int(response.json()["meta_info"]["cached_tokens"]), 0
                    )
                else:
                    self.assertGreater(
                        int(response.json()["meta_info"]["cached_tokens"]), 0
                    )
        finally:
            kill_process_tree(self.process.pid)
            self.process = None

    def test_002_combined_params(self):
        """Test Hicache with combined parameters, hicache_storage_backend is configured as a file, file storage is hosted under hicache."""
        logging.warning("\n=== Test 002: Combined Parameters ===")
        self.process = self._launch_server_with_hicache(
            hicache_ratio=2.0,
            hicache_write_policy="write_through",
            radix_eviction_policy="lfu",
            hicache_io_backend="direct",
            hicache_mem_layout="page_first_direct",
            hicache_storage_backend="file",
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            run_eval(args)
            self.assertTrue(
                os.path.exists("/tmp/hicache") and os.listdir("/tmp/hicache")
            )
            hicache_file_size = sum(
                os.path.getsize(os.path.join("/tmp/hicache", f))
                for f in os.listdir("/tmp/hicache")
                if os.path.isfile(os.path.join("/tmp/hicache", f))
            )
            self.assertGreater(hicache_file_size, 0)
        finally:
            kill_process_tree(self.process.pid)
            self.process = None
            shutil.rmtree("/tmp/hicache", ignore_errors=True)

    def test_003_combined_params(self):
        """Test Hicache with combined parameters, hicache with long sequence"""
        logging.warning("\n=== Test 003: Combined Parameters ===")
        self.process = self._launch_server_with_hicache(
            hicache_size=100,
            hicache_write_policy="write_through_selective",
            hicache_io_backend="kernel_ascend",
            hicache_mem_layout="page_first_kv_split",
        )

        try:
            long_prompt = "Explain the concept of machine learning in detail. " * 100
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": long_prompt,
                    "sampling_params": {
                        "temperature": 0.7,
                        "max_new_tokens": 128,
                    },
                },
                timeout=180,
            )
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.text), 50)
            logging.warning(
                f"Long sequence test passed, result length: {len(response.text)}"
            )
        finally:
            kill_process_tree(self.process.pid)
            self.process = None


if __name__ == "__main__":
    unittest.main()
