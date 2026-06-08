import os
import shutil
import tempfile
import unittest

from test_unified_radix_cache_kl_nightly import AccuracyTwoPassMixin

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

DSV4_FLASH_MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
DSV4_FLASH_LAUNCH_TIMEOUT = 3600

register_cuda_ci(est_time=868, stage="base-c", runner_config="4-gpu-h100")


def _assert_dsv4_decode_cached_tokens(result, history_len, output_len, label):
    expected = history_len + output_len
    actual = result["meta_info"]["cached_tokens"]
    lower = max(0, expected - 256)
    assert actual >= lower, f"{label}: expected cached_tokens>={lower}, got {actual}"


class TestUnifiedDeepSeekV4FlashHiCache(UnifiedRadixTreeTestMixin, CustomTestCase):
    """DeepSeek V4 Flash FP8 + HiCache + UnifiedRadixCache."""

    hicache_io_backend = "direct"
    hicache_mem_layout = "page_first_direct"
    max_running_requests = 4
    kl_threshold = 0.005
    sampling_temperature = 0
    decode_hit_request_batch_size = 3
    decode_hit_inter_batch_delay_s = 0.5
    decode_cache_assert = staticmethod(_assert_dsv4_decode_cached_tokens)
    gsm8k_threshold = 0.90
    num_gsm8k_questions = 100

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_multiturn_logprobs_match(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_FLASH_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DSV4_FLASH_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--attention-backend",
                "compressed",
                "--page-size",
                "256",
                "--chunked-prefill-size",
                "8192",
                "--mem-fraction-static",
                "0.9",
                "--disable-shared-experts-fusion",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "4",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                cls.hicache_io_backend,
                "--hicache-mem-layout",
                cls.hicache_mem_layout,
                "--swa-full-tokens-ratio",
                "0.25",
                "--max-total-tokens",
                "20000",
                "--max-running-requests",
                str(cls.max_running_requests),
            ],
            env={
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            },
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestUnifiedDeepSeekV4FlashHiCachePageFirstDirect(
    TestUnifiedDeepSeekV4FlashHiCache
):
    """DeepSeek V4 Flash HiCache layout smoke: page_first_direct + direct."""

    hicache_io_backend = "kernel"
    hicache_mem_layout = "layer_first"


# ─── DeepSeek V4 Flash + HiCache L3 (file backend) ──────────────────────


class TestUnifiedDeepSeekV4FlashHiCacheL3(AccuracyTwoPassMixin, CustomTestCase):
    """DeepSeek V4 Flash FP8 + HiCache L3 (file backend) + UnifiedRadixCache."""

    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_FLASH_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.hicache_dir = tempfile.mkdtemp(prefix="hicache_l3_dsv4_")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DSV4_FLASH_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--attention-backend",
                "compressed",
                "--page-size",
                "256",
                "--chunked-prefill-size",
                "8192",
                "--mem-fraction-static",
                "0.9",
                "--disable-shared-experts-fusion",
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
                "--swa-full-tokens-ratio",
                "0.25",
            ],
            env={
                "SGLANG_DSV4_FP4_EXPERTS": "0",
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
