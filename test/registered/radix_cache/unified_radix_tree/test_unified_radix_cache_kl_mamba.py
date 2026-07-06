import os
import shutil
import tempfile
import unittest

from test_unified_radix_cache_kl_nightly import AccuracyTwoPassMixin

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    make_mamba_decode_assert,
    make_mamba_prefill_assert,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=858, stage="extra-b", runner_config="4-gpu-h100")

MAMBA_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
MAMBA_CHUNK_SIZE = 64
MAMBA_TRACK_INTERVAL = 128
MAMBA_CHUNKED_PREFILL_SIZE = 2048


class TestUnifiedMambaRadixCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Mamba hybrid + UnifiedRadixCache."""

    # fp8 cache-hit KL is noisy and flaked at 0.003, match the HiCache variant
    kl_threshold = 0.005
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=MAMBA_CHUNK_SIZE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=MAMBA_TRACK_INTERVAL)
    )

    @classmethod
    def setUpClass(cls):
        cls.model = MAMBA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--chunked-prefill-size",
                str(MAMBA_CHUNKED_PREFILL_SIZE),
                "--mem-fraction-static",
                "0.85",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                str(MAMBA_TRACK_INTERVAL),
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


# ─── Mamba + HiCache L2 ──────────────────────────────────────────────────────


class TestUnifiedMambaHiCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Mamba hybrid + HiCache L2 + UnifiedRadixCache."""

    kl_threshold = 0.005
    prefill_cache_assert = staticmethod(
        make_mamba_prefill_assert(chunk_size=MAMBA_CHUNK_SIZE)
    )
    decode_cache_assert = staticmethod(
        make_mamba_decode_assert(track_interval=MAMBA_TRACK_INTERVAL)
    )

    @classmethod
    def setUpClass(cls):
        cls.model = MAMBA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--chunked-prefill-size",
                str(MAMBA_CHUNKED_PREFILL_SIZE),
                "--mem-fraction-static",
                "0.85",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                str(MAMBA_TRACK_INTERVAL),
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "4",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--max-total-tokens",
                "12000",
                "--max-mamba-cache-size",
                "500",
                "--max-running-requests",
                "4",
                "--weight-loader-prefetch-checkpoints",
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


# ─── Mamba + HiCache L3 (file backend) ───────────────────────────────────────


class TestUnifiedMambaHiCacheL3(AccuracyTwoPassMixin, CustomTestCase):
    """Mamba hybrid + HiCache L3 (file backend) + UnifiedRadixCache."""

    # Prompt must exceed chunked_prefill_size to exercise the multi-chunk path.
    l3_prefetch_page_size = MAMBA_CHUNK_SIZE
    l3_prefetch_prompt_pages = MAMBA_CHUNKED_PREFILL_SIZE // MAMBA_CHUNK_SIZE + 16
    # Mamba state is only persisted at chunk boundaries, so up to a full
    # chunked_prefill_size of trailing tokens may stay uncached.
    l3_prefetch_max_uncached_tokens = MAMBA_CHUNKED_PREFILL_SIZE

    @classmethod
    def setUpClass(cls):
        cls.model = MAMBA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.hicache_dir = tempfile.mkdtemp(prefix="hicache_l3_mamba_")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--chunked-prefill-size",
                str(MAMBA_CHUNKED_PREFILL_SIZE),
                "--mem-fraction-static",
                "0.85",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                str(MAMBA_TRACK_INTERVAL),
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
                "--hicache-io-backend",
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--hicache-storage-backend",
                "file",
                "--max-mamba-cache-size",
                "500",
                "--weight-loader-prefetch-checkpoints",
            ],
            env={
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.hicache_dir,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if os.path.isdir(cls.hicache_dir):
            shutil.rmtree(cls.hicache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
