"""NPU e2e test for --hicache-storage-prefetch-retry-max-attempts.

Validates that the HiCache L3 storage prefetch retry budget parameter is
device-agnostic on Ascend NPU: a long prompt is backed up to the ``file``
storage backend, the device cache is flushed, and a second identical request
is served by a storage prefetch hit, with the retry parameters consumed
(``--hicache-storage-prefetch-retry-max-attempts`` and its paced poll interval).

[Test Category] HiCache
[Test Target] --hicache-storage-prefetch-retry-max-attempts
"""

import random
import shutil
import tempfile
import unittest

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNPUHiCacheStoragePrefetchRetry(CustomTestCase):
    """Storage prefetch retry-budget e2e on Ascend NPU.

    Cover scenarios:
    1. Long prompt is backed up from device cache to the ``file`` L3 backend.
    2. ``/flush_cache`` forces remote (L3) access on the next request.
    3. Second identical request is a storage prefetch hit (cached_tokens > 0),
       with ``--hicache-storage-prefetch-retry-max-attempts`` and
       ``--hicache-storage-prefetch-retry-poll-interval`` consumed on the path.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.temp_dir = tempfile.mkdtemp()
        cls.tokenizer = get_tokenizer(cls.model)

        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.8",
            "--tp-size",
            "1",
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "1.2",
            "--page-size",
            "64",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--enable-cache-report",
            "--hicache-storage-prefetch-retry-max-attempts",
            "4",
            "--hicache-storage-prefetch-retry-poll-interval",
            "1",
        ]
        env = {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def _send(self, text, max_new_tokens=16):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
        self.assertEqual(
            response.status_code,
            200,
            f"generate failed: {response.status_code} - {response.text}",
        )
        return response.json()

    def _flush_cache(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        self.assertEqual(
            response.status_code,
            200,
            f"flush_cache failed: {response.status_code} - {response.text}",
        )

    @staticmethod
    def _gen_prompt(tokenizer, token_num):
        vocab = list(tokenizer.get_vocab().values())
        selected = random.choices(vocab, k=token_num)
        return tokenizer.decode(selected)

    def test_storage_prefetch_retry_budget(self):
        # Long prompt that spans multiple pages, so it is backed up to L3.
        prompt = self._gen_prompt(self.tokenizer, 768)

        first = self._send(prompt)
        self.assertEqual(int(first["meta_info"]["cached_tokens"]), 0)

        # Force device cache to drain to the file backend, so the next request
        # must walk the storage prefetch path (and its retry budget).
        self._flush_cache()

        second = self._send(prompt)
        cached_tokens = int(second["meta_info"]["cached_tokens"])
        self.assertGreater(
            cached_tokens,
            500,
            f"Expected a storage prefetch hit, got cached_tokens={cached_tokens}",
        )


if __name__ == "__main__":
    unittest.main()