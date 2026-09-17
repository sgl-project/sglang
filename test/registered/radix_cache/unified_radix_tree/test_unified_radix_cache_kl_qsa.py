"""Qwen3.8-Flash-Next (Qwen4-Exp) + UnifiedRadixCache + HiCache.

A page restored without its QSA compressed index-K carries valid K/V and scores
against stale keys, so only a cache-hit-vs-fresh KL comparison catches it. The
device pools below are sized to force eviction and load-back.
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
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    make_mamba_decode_assert,
    make_mamba_prefill_assert,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

# Four server launches of a 176B checkpoint dominate the budget.
register_cuda_ci(est_time=2400, stage="extra-b", runner_config="4-gpu-b200")

QSA_MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
SERVER_LAUNCH_TIMEOUT = 3600

# Forced by the Qwen4-Exp arg overrides: compressed QSA addresses a compressed
# slot as full_slot // ratio, so full-KV pages must be page-aligned at 64.
QSA_PAGE_SIZE = 64
QSA_TRACK_INTERVAL = 128
QSA_CHUNKED_PREFILL_SIZE = 8192

BASE_ARGS = [
    "--tp-size",
    "4",
    "--mem-fraction-static",
    "0.85",
    "--chunked-prefill-size",
    str(QSA_CHUNKED_PREFILL_SIZE),
    "--linear-attn-prefill-backend",
    "flashinfer",
    "--linear-attn-decode-backend",
    "flashinfer",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--mamba-track-interval",
    str(QSA_TRACK_INTERVAL),
]

HICACHE_ARGS = [
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    "4",
    "--hicache-write-policy",
    "write_through",
]

SERVER_ENV = {
    "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
    "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
}


class _QsaGsm8kNotScoredHere:
    """The mixin's GSM8K path runs the plain few-shot harness, which scores a
    thinking model well below its default floor. Accuracy for this checkpoint
    is covered with the thinking harness by test_qwen4_exp_models.py.
    """

    def test_gsm8k(self):
        raise unittest.SkipTest(
            "accuracy is covered by test_qwen4_exp_models.py with the thinking "
            "harness; this file asserts cache-hit KL"
        )


class TestUnifiedQsaRadixCache(
    _QsaGsm8kNotScoredHere, UnifiedRadixTreeTestMixin, CustomTestCase
):
    """Device-resident baseline: separates HiCache faults from QSA faults."""

    kl_threshold = 0.005
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=QSA_PAGE_SIZE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=QSA_TRACK_INTERVAL)
    )

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(QSA_MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=BASE_ARGS,
            env=SERVER_ENV,
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process, wait_timeout=60)


class TestUnifiedQsaHiCache(
    _QsaGsm8kNotScoredHere, UnifiedRadixTreeTestMixin, CustomTestCase
):
    """QSA + HiCache L2, over the kernel io backend."""

    kl_threshold = 0.005
    hicache_io_backend = "kernel"
    hicache_mem_layout = "page_first"
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=QSA_PAGE_SIZE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=QSA_TRACK_INTERVAL)
    )

    @classmethod
    def _server_args(cls):
        return (
            BASE_ARGS
            + HICACHE_ARGS
            + [
                "--hicache-io-backend",
                cls.hicache_io_backend,
                "--hicache-mem-layout",
                cls.hicache_mem_layout,
                # Small pools so the device tier overflows and pages come back
                # from host; otherwise the KL assertion passes vacuously.
                "--max-total-tokens",
                "20000",
                "--max-mamba-cache-size",
                "500",
                "--max-running-requests",
                "4",
            ]
        )

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(QSA_MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=cls._server_args(),
            env=SERVER_ENV,
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process, wait_timeout=60)


class TestUnifiedQsaHiCachePageFirstDirect(TestUnifiedQsaHiCache):
    """The other page-first layout / io-backend pair the host pools accept."""

    hicache_io_backend = "direct"
    hicache_mem_layout = "page_first_direct"


class TestUnifiedQsaHiCacheL3(AccuracyTwoPassMixin, CustomTestCase):
    """QSA + HiCache L3 (file backend): the sidecars must survive storage too.

    Only the prefetch test runs; `test_gsm8k_two_passes` asserts an absolute
    accuracy floor scored by the plain few-shot harness, which this thinking
    model does not meet. Enable it once a floor is measured for it.
    """

    l3_prefetch_page_size = QSA_PAGE_SIZE
    l3_prefetch_prompt_pages = QSA_CHUNKED_PREFILL_SIZE // QSA_PAGE_SIZE + 16
    # GDN state is only persisted at chunk boundaries, so up to a full
    # chunked_prefill_size of trailing tokens may stay uncached.
    l3_prefetch_max_uncached_tokens = QSA_CHUNKED_PREFILL_SIZE

    def test_gsm8k_two_passes(self):
        raise unittest.SkipTest(
            "needs a measured GSM8K floor for this thinking model under the "
            "plain few-shot harness; see the class docstring"
        )

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(QSA_MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.hicache_dir = tempfile.mkdtemp(prefix="hicache_l3_qsa_")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=BASE_ARGS
            + [
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--hicache-io-backend",
                "kernel",
                "--hicache-mem-layout",
                "page_first",
                "--hicache-storage-backend",
                "file",
                "--max-mamba-cache-size",
                "500",
            ],
            env=SERVER_ENV
            | {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.hicache_dir},
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process, wait_timeout=60)
        if os.path.isdir(cls.hicache_dir):
            shutil.rmtree(cls.hicache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
