"""DeepSeek V4.1 Flash + HiCache + UnifiedRadixCache.

V4.1 replaces the c4/c128 compressed layers with ratio-1/2 kv_source layers, so
the HiCache stack mirrors the low-ratio latent and fp4 index-key pools
(deepseek_v4_c1 / deepseek_v4_c2 and their indexers) next to KV and SWA.
The device pool is capped so the L2 (host) and L3 (storage) paths actually run.
"""

import os
import shutil
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import (
    AccuracyTwoPassMixin,
    UnifiedRadixTreeTestMixin,
)
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
    unified_radix_tree_server_env,
)

DSV41_FLASH_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DSV41_FLASH_LAUNCH_TIMEOUT = 3600

register_cuda_ci(est_time=3600, stage="nightly", runner_config="4-gpu-gb300")

# A cached prefix may lose up to one 256-token page at the tail.
DSV41_PAGE_SIZE = 256


def _assert_dsv41_decode_cached_tokens(result, history_len, output_len, label):
    expected = history_len + output_len
    actual = result["meta_info"]["cached_tokens"]
    lower = max(0, expected - DSV41_PAGE_SIZE)
    assert actual >= lower, f"{label}: expected cached_tokens>={lower}, got {actual}"


def _dsv41_base_args():
    return [
        "--trust-remote-code",
        "--tp-size",
        "4",
        "--ep-size",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--chunked-prefill-size",
        "8192",
        "--context-length",
        "32768",
        # Small enough that the radix tree evicts to host during the tests.
        "--max-total-tokens",
        "65536",
        "--max-running-requests",
        "8",
        "--enable-cache-report",
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "4",
        "--hicache-write-policy",
        "write_through",
    ]


class TestUnifiedDeepSeekV41FlashHiCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """DeepSeek V4.1 Flash + HiCache L2 + UnifiedRadixCache."""

    tree_core_backend = "python"
    hicache_io_backend = "kernel"
    hicache_mem_layout = "page_first"
    # The dsv4 backend is not batch-invariant (deterministic inference is refused
    # on it), so decode-vs-prefill KL has a floor with or without HiCache: the
    # same server without --enable-hierarchical-cache measured 0.0063 on the
    # decode-cache-hit helper (0.0062 with HiCache) and 0.0009 / 0.0022 on the
    # prefill-cache-hit helper, 9 samples, 512 tokens, 4xGB300 TP4.
    kl_threshold = 0.01
    sampling_temperature = 0
    decode_hit_request_batch_size = 3
    decode_hit_inter_batch_delay_s = 0.5
    decode_cache_assert = staticmethod(_assert_dsv41_decode_cached_tokens)
    gsm8k_threshold = 0.90
    num_gsm8k_questions = 100

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_multiturn_logprobs_match(self):
        pass

    @classmethod
    def _server_args(cls):
        return _dsv41_base_args() + [
            "--hicache-io-backend",
            cls.hicache_io_backend,
            "--hicache-mem-layout",
            cls.hicache_mem_layout,
        ]

    @classmethod
    def setUpClass(cls):
        cls.model = DSV41_FLASH_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DSV41_FLASH_LAUNCH_TIMEOUT,
            other_args=cls._server_args(),
            env=unified_radix_tree_server_env(cls.tree_core_backend),
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)


class TestUnifiedDeepSeekV41FlashDSparkHiCacheL3(AccuracyTwoPassMixin, CustomTestCase):
    """DeepSeek V4.1 Flash DSpark + HiCache L3 (file backend) + UnifiedRadixCache."""

    tree_core_backend = "python"
    l3_prefetch_page_size = DSV41_PAGE_SIZE
    l3_prefetch_prompt_pages = 4
    gsm8k_threshold = 0.90
    num_gsm8k_questions = 100
    gsm8k_parallel = 8

    @classmethod
    def setUpClass(cls):
        cls.model = DSV41_FLASH_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.hicache_dir = tempfile.mkdtemp(prefix="hicache_l3_dsv41_")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DSV41_FLASH_LAUNCH_TIMEOUT,
            other_args=_dsv41_base_args()
            + [
                "--hicache-io-backend",
                "kernel",
                "--hicache-mem-layout",
                "page_first",
                "--hicache-storage-backend",
                "file",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-dspark-block-size",
                "5",
            ],
            env=unified_radix_tree_server_env(
                cls.tree_core_backend,
                SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=cls.hicache_dir,
            ),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)
        hicache_dir = getattr(cls, "hicache_dir", None)
        if hicache_dir and os.path.isdir(hicache_dir):
            shutil.rmtree(hicache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
