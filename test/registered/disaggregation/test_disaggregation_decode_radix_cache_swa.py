"""SWA coverage for decode-side radix cache on gpt-oss-20b.

The decode worker reuses full-attention prefix KV while transferring the SWA
window fresh per request. This path requires the unified radix tree and validates
both multi-turn cache hits and two-pass GSM8K accuracy.

The HiCache variant additionally runs the decode tier's hierarchical cache
(host tier + file L3) on the SWA model: restored L2/L3 ranges are capped at
the sliding-window start so the SWA tail is still transferred fresh, and
hybrid models never promise L3 restores the KV-only hit query cannot back.
"""

import asyncio
import os
import shutil
import tempfile
import time
import unittest

import requests
from test_disaggregation_decode_radix_cache import (
    DisaggregationDecodeRadixCacheTestMixin,
    _has_mooncake,
)

from sglang.benchmark.datasets.random import sample_random_requests
from sglang.benchmark.utils import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.cache_hit_kit import async_request_sglang_generate, gen_payload
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE,
    is_in_ci,
    try_cached_model,
)

register_cuda_ci(est_time=700, stage="extra-b", runner_config="8-gpu-h200")

SWA_SERVER_ARGS = ["--page-size", "64", "--attention-backend", "triton"]
HICACHE_SERVER_ARGS = [
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


def _has_nixl():
    try:
        import nixl._api  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    is_in_ci() or _has_nixl(),
    "NIXL is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixCacheSWANixl(
    DisaggregationDecodeRadixCacheTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "nixl"
    model_name = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    # The 512-token eval cap truncates mxfp4 gpt-oss reasoning. On the fixed
    # 500-question H200 sample, the score has a roughly 2-point standard error,
    # so keep the original 0.45 absolute floor and rely on the two-pass
    # non-regression check below to catch decode-cache corruption.
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
    is_in_ci() or _has_mooncake(),
    "Mooncake is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixHiCacheSWA(
    DisaggregationDecodeRadixCacheTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "mooncake"
    model_name = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE
    gsm8k_min_score = 0.45
    extra_prefill_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_decode_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_prefill_args = [*SWA_SERVER_ARGS, *HICACHE_SERVER_ARGS]
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        *SWA_SERVER_ARGS,
        *HICACHE_SERVER_ARGS,
    ]

    @classmethod
    def setUpClass(cls):
        cls.hicache_dir = tempfile.mkdtemp(prefix="sglang-hicache-swa-")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.hicache_dir
        super().setUpClass()

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

    def test_multiturn_continuations_across_flushes(self):
        """Grow one conversation across full device/host flushes.

        Before the guards were lifted, HiCache could not be enabled on an SWA
        decode tier at all; an uncapped restore promise then tripped the
        SWA-tail allocator. Every round must succeed and both tiers stay up.
        """
        self._post_ok(f"{self.decode_url}/hicache/storage-backend/clear")
        self._flush_memory_cache()

        tokenizer = get_tokenizer(self.model)
        history = list(
            sample_random_requests(
                input_len=256,
                output_len=64,
                num_prompts=1,
                range_ratio=1.0,
                tokenizer=tokenizer,
                dataset_path="",
                return_text=False,
            )[0].prompt
        )
        for _ in range(4):
            output = self._generate(history, output_len=64)
            history.extend(output.output_ids)
            time.sleep(1)
            self._flush_memory_cache()

        self._assert_process_healthy("prefill", self.process_prefill, self.prefill_url)
        self._assert_process_healthy("decode", self.process_decode, self.decode_url)


@unittest.skipUnless(
    is_in_ci() or _has_mooncake(),
    "Mooncake is required for decode radix cache disaggregation coverage.",
)
class TestDisaggregationDecodeRadixHiCacheSWAL2Restore(PDDisaggregationServerBase):
    """Decode-side L2 (host) restores on the SWA model.

    The decode KV pool is capped so two interleaved conversations evict each
    other's prefix to the host tier. The next turn of a conversation then
    restores its full-attention prefix from L2 below the sliding-window cap
    while the SWA tail is transferred fresh -- the total_prefix_len >
    l1_prefix_len path that `TestDisaggregationDecodeRadixHiCacheSWA` never
    reaches because it flushes the host tier between turns.
    """

    transfer_backend_name = "mooncake"
    extra_prefill_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_decode_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_prefill_args = [*SWA_SERVER_ARGS, *HICACHE_SERVER_ARGS]
    # 96 pages of device KV: two ~3K-token conversations cannot both stay
    # resident, so each turn evicts the other's prefix to the host tier.
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        "--max-total-tokens",
        "6144",
        *SWA_SERVER_ARGS,
        *HICACHE_SERVER_ARGS,
        "--hicache-ratio",
        "4",
    ]

    @classmethod
    def setUpClass(cls):
        cls.hicache_dir = tempfile.mkdtemp(prefix="sglang-hicache-swa-l2-")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.hicache_dir
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE)
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

    def _generate(self, input_ids, output_len):
        output = asyncio.run(
            async_request_sglang_generate(
                gen_payload(input_ids, output_len),
                f"{self.base_url}/generate",
            )
        )
        self.assertTrue(output.success, output.error)
        return output

    def test_interleaved_conversations_restore_from_host(self):
        tokenizer = get_tokenizer(self.model)
        histories = [
            list(
                sample_random_requests(
                    input_len=2048,
                    output_len=256,
                    num_prompts=1,
                    range_ratio=1.0,
                    tokenizer=tokenizer,
                    dataset_path="",
                    return_text=False,
                )[0].prompt
            )
            for _ in range(2)
        ]
        cached_tokens = [[], []]
        for round_idx in range(3):
            for conv in range(2):
                output = self._generate(histories[conv], output_len=256)
                histories[conv].extend(output.output_ids)
                cached_tokens[conv].append(output.cached_tokens)
                assert_process_healthy(
                    self, "decode", self.process_decode, self.decode_url
                )

        # Each conversation's second and later turns must reuse its prefix
        # (from device or restored from host); the first turn is cold.
        for conv in range(2):
            self.assertGreater(
                cached_tokens[conv][1],
                0,
                f"conversation {conv} did not reuse its prefix: {cached_tokens[conv]}",
            )
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()
