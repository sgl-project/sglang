"""HiCache storage must isolate LoRA and base pages across cache flushes."""

import json
import os
import random
import shutil
import tempfile
import unittest
from typing import Dict, Optional

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

LORA_NAME = "sql"
LORA_PATH = "philschmid/code-llama-3-1-8b-text-to-sql-lora"
PAGE_SIZE = 64
PROMPT_TOKENS = 768


class TestHiCacheStorageLoRAIsolation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = get_tokenizer(cls.model)

        extra_config = {"hicache_storage_pass_prefix_keys": True}
        other_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--page-size",
            str(PAGE_SIZE),
            "--enable-cache-report",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-backend-extra-config",
            json.dumps(extra_config),
            "--enable-lora",
            "--lora-paths",
            f"{LORA_NAME}={LORA_PATH}",
            "--max-loras-per-batch",
            "2",
            # Triton keeps radix caching enabled under deterministic inference.
            "--enable-deterministic-inference",
            "--attention-backend",
            "triton",
        ]
        env = {**os.environ, "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            terminate_and_kill_process_tree(cls.process)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def send_request(
        self, prompt: str, lora_path: Optional[str], max_tokens: int = 32
    ) -> Dict:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_tokens,
                "ignore_eos": True,
            },
        }
        if lora_path is not None:
            payload["lora_path"] = lora_path
        response = requests.post(f"{self.base_url}/generate", json=payload, timeout=120)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    @staticmethod
    def cached_tokens(response_json: Dict) -> int:
        return int(response_json.get("meta_info", {}).get("cached_tokens", 0))

    def flush_device_cache(self):
        # A short unrelated request first so the pages of interest get offloaded.
        self.send_request(self.gen_prompt(1), lora_path=None, max_tokens=150)
        res = requests.post(
            f"{self.base_url}/flush_cache", params={"timeout": 30}, timeout=40
        )
        res.raise_for_status()

    def gen_prompt(self, token_num: int) -> str:
        vocab = list(self.tokenizer.get_vocab().values())
        return self.tokenizer.decode(random.choices(vocab, k=token_num))

    def test_adapter_pages_are_isolated_in_storage(self):
        prompt = self.gen_prompt(PROMPT_TOKENS)
        hit_floor = PROMPT_TOKENS - 2 * PAGE_SIZE

        # Cold pass with the adapter populates host and storage.
        lora_first = self.send_request(prompt, lora_path=LORA_NAME)
        self.flush_device_cache()

        # Read adapter pages before any base request stores the same prompt.
        lora_again = self.send_request(prompt, lora_path=LORA_NAME)
        self.assertGreater(
            self.cached_tokens(lora_again),
            hit_floor,
            "the adapter's pages were not served from storage after the flush",
        )
        self.assertEqual(lora_first["text"], lora_again["text"])
        self.flush_device_cache()

        # The same prompt without the adapter must not find the adapter's pages.
        base_first = self.send_request(prompt, lora_path=None)
        self.assertLess(
            self.cached_tokens(base_first),
            PAGE_SIZE,
            "a request without the adapter hit pages written under the adapter",
        )
        self.flush_device_cache()

        # Base pages round-trip too, and the adapter still does not see them.
        base_again = self.send_request(prompt, lora_path=None)
        self.assertGreater(self.cached_tokens(base_again), hit_floor)
        self.assertEqual(base_first["text"], base_again["text"])
        self.flush_device_cache()

        lora_third = self.send_request(prompt, lora_path=LORA_NAME)
        self.assertGreater(self.cached_tokens(lora_third), hit_floor)
        self.assertEqual(
            lora_first["text"],
            lora_third["text"],
            "adapter output changed once base pages for the same prompt existed",
        )


if __name__ == "__main__":
    unittest.main()
