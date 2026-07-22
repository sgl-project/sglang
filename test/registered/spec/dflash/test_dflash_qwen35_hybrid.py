import os
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-large")

TARGET_MODEL = os.environ.get("SGLANG_TEST_QWEN35_MODEL", "Qwen/Qwen3.5-4B")
DRAFT_MODEL = os.environ.get(
    "SGLANG_TEST_QWEN35_DFLASH_MODEL", "z-lab/Qwen3.5-4B-DFlash"
)


class TestDFlashQwen35HybridRadixCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(
            TARGET_MODEL, trust_remote_code=True
        )
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--speculative-num-draft-tokens",
                "16",
                "--mamba-radix-cache-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                "256",
                "--context-length",
                "8192",
                "--max-running-requests",
                "4",
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                "0.65",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, input_ids, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
        )
        response.raise_for_status()
        return response.json()

    def test_generated_prefix_cache_matches_full_prefill(self):
        paragraph = (
            "This is a deterministic prefix-cache regression input with repeated text. "
        )
        base_ids = self.tokenizer.encode(paragraph * 400, add_special_tokens=False)[
            :4600
        ]
        self.assertEqual(len(base_ids), 4600)
        suffix_ids = self.tokenizer.encode(
            "\nContinue from this prefix.",
            add_special_tokens=False,
        )

        requests.post(self.base_url + "/flush_cache").raise_for_status()
        seed = self._generate(base_ids, max_new_tokens=320)
        self.assertGreater(seed["meta_info"]["spec_verify_ct"], 0)
        continuation_ids = base_ids + seed["output_ids"] + suffix_ids
        cached = self._generate(continuation_ids, max_new_tokens=32)

        expected_cached_tokens = (len(base_ids) + len(seed["output_ids"])) // 256 * 256
        self.assertGreater(expected_cached_tokens, len(base_ids))
        self.assertGreaterEqual(
            cached["meta_info"]["cached_tokens"], expected_cached_tokens
        )
        requests.post(self.base_url + "/flush_cache").raise_for_status()
        full_prefill = self._generate(continuation_ids, max_new_tokens=32)

        self.assertEqual(full_prefill["meta_info"]["cached_tokens"], 0)
        self.assertEqual(cached["output_ids"], full_prefill["output_ids"])


if __name__ == "__main__":
    unittest.main()
