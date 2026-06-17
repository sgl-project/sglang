"""SWA coverage for decode-side radix cache on gpt-oss-20b.

The decode worker reuses full-attention prefix KV while transferring the SWA
window fresh per request. This path requires the unified radix tree and validates
both multi-turn cache hits and two-pass GSM8K accuracy.
"""

import asyncio
import os
import shutil
import tempfile
import time
import unittest

import requests
from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer
from test_disaggregation_decode_radix_cache import (
    DisaggregationDecodeRadixCacheTestMixin,
    _has_nixl,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.cache_hit_kit import async_request_sglang_generate, gen_payload
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    is_in_ci,
    try_cached_model,
)

register_cuda_ci(est_time=600, stage="base-c", runner_config="8-gpu-h20")

SWA_SERVER_ARGS = ["--page-size", "64", "--attention-backend", "triton"]
HICACHE_FILE_ARGS = [
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    "1.2",
    "--hicache-write-policy",
    "write_through",
    "--hicache-storage-backend",
    "file",
    "--hicache-storage-prefetch-policy",
    "wait_complete",
    "--hicache-io-backend",
    "kernel",
    "--hicache-mem-layout",
    "page_first",
]


@unittest.skipUnless(
    is_in_ci() or _has_nixl(),
    "NIXL is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixCacheSWANixl(
    DisaggregationDecodeRadixCacheTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "nixl"
    model_name = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    # The 512-token eval cap truncates mxfp4 gpt-oss reasoning, so use a lower
    # absolute floor while checking that the cached second pass does not regress.
    gsm8k_min_score = 0.45
    # SWA + decode-side radix cache is gated to the unified radix tree.
    extra_prefill_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_decode_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_prefill_args = SWA_SERVER_ARGS
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        *SWA_SERVER_ARGS,
    ]


@unittest.skipUnless(
    is_in_ci() or _has_nixl(),
    "NIXL is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixCacheSWAHiCacheFileBackend(
    PDDisaggregationServerBase
):
    transfer_backend_name = "nixl"
    model_name = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    extra_prefill_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_decode_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_prefill_args = [*SWA_SERVER_ARGS, *HICACHE_FILE_ARGS]
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        *SWA_SERVER_ARGS,
        *HICACHE_FILE_ARGS,
    ]

    @classmethod
    def setUpClass(cls):
        cls.hicache_dir = tempfile.mkdtemp(prefix="sglang-hicache-swa-")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.hicache_dir

        super().setUpClass()
        cls.model = try_cached_model(cls.model_name)
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.transfer_backend_name,
        ]
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", None)
        shutil.rmtree(cls.hicache_dir, ignore_errors=True)

    def _post_ok(self, url):
        response = requests.post(url, timeout=60)
        response.raise_for_status()

    def _flush_memory_cache(self):
        self._post_ok(f"{self.prefill_url}/flush_cache?timeout=30")
        self._post_ok(f"{self.decode_url}/flush_cache?timeout=30")

    def _generate(self, input_ids, output_len):
        output = asyncio.run(
            async_request_sglang_generate(
                gen_payload(input_ids, output_len),
                f"{self.base_url}/generate",
            )
        )
        self.assertTrue(output.success, output.error)
        return output

    def _sample_token_ids(self, input_len, output_len, num_prompts=1):
        tokenizer = get_tokenizer(self.model)
        return [
            list(request.prompt)
            for request in sample_random_requests(
                input_len=input_len,
                output_len=output_len,
                num_prompts=num_prompts,
                range_ratio=1.0,
                tokenizer=tokenizer,
                dataset_path="",
                return_text=False,
            )
        ]

    def test_decode_hicache_file_backend_restores_swa_prefix_after_flush(self):
        self._post_ok(f"{self.decode_url}/hicache/storage-backend/clear")
        self._flush_memory_cache()

        num_rounds = 4
        output_len = 32
        history = self._sample_token_ids(
            input_len=192, output_len=output_len, num_prompts=1
        )[0]
        suffixes = self._sample_token_ids(
            input_len=32, output_len=output_len, num_prompts=num_rounds - 1
        )

        saw_cache_hit_after_flush = False
        prev_prompt_len = 0
        prev_output_len = 0
        for round_idx in range(num_rounds):
            output = self._generate(history, output_len)
            if round_idx == 0:
                self.assertEqual(output.cached_tokens, 0)
            else:
                self.assertGreater(output.cached_tokens, 0)
                if round_idx % 2 == 1:
                    saw_cache_hit_after_flush = True
                self.assertGreaterEqual(
                    output.cached_tokens,
                    min(prev_prompt_len + prev_output_len, len(history) - 128),
                )

            history.extend(output.output_ids)
            prev_prompt_len = output.prompt_len
            prev_output_len = len(output.output_ids)

            if round_idx < num_rounds - 1:
                history.extend(suffixes[round_idx])
                time.sleep(1)
                if round_idx % 2 == 0:
                    self._flush_memory_cache()

        self.assertTrue(saw_cache_hit_after_flush)


if __name__ == "__main__":
    unittest.main()
