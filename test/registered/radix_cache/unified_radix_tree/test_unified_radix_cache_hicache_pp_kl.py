import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from test_unified_radix_cache_kl_nightly import AccuracyTwoPassMixin

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import get_input_ids
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="4-gpu-h100")

QWEN3_32B_MODEL = "Qwen/Qwen3-32B"


def _assert_pp_decode_cached_tokens(result, history_len, output_len, label):
    expected = history_len + output_len
    actual = result["meta_info"]["cached_tokens"]
    lower = max(0, expected - 1)
    assert (
        lower <= actual <= expected
    ), f"{label}: expected cached_tokens in [{lower}, {expected}], got {actual}"


class TestUnifiedQwen3HiCachePP(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Qwen3-32B + HiCache + PP + UnifiedRadixCache."""

    hicache_io_backend = "kernel"
    hicache_mem_layout = "page_first"
    max_running_requests = 2
    kl_threshold = 0.005
    gsm8k_threshold = 0.7
    num_gsm8k_questions = 50
    mmlu_threshold = 0.7
    decode_cache_assert = staticmethod(_assert_pp_decode_cached_tokens)

    def test_gsm8k(self):
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=10,
            data_path=None,
            num_questions=self.num_gsm8k_questions,
            max_new_tokens=2048,
            parallel=self.max_running_requests,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_few_shot_gsm8k(args)
        print(
            f"[{self.__class__.__name__}] GSM8K accuracy: {metrics['accuracy']:.3f} "
            f"(threshold: {self.gsm8k_threshold})"
        )
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_threshold)

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "2",
                "--pp-size",
                "2",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-max-bs-decode",
                "32",
                "--max-running-requests",
                str(cls.max_running_requests),
                "--max-total-tokens",
                "14000",
                "--disable-piecewise-cuda-graph",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "4",
                "--hicache-write-policy",
                "write_through",
                "--hicache-io-backend",
                cls.hicache_io_backend,
                "--hicache-mem-layout",
                cls.hicache_mem_layout,
            ],
            env={"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"},
        )
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestUnifiedQwen3HiCachePPL3(AccuracyTwoPassMixin, CustomTestCase):
    """Qwen3-32B + HiCache L3 (file backend) + PP + UnifiedRadixCache."""

    gsm8k_threshold = 0.8

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_32B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.hicache_dir = tempfile.mkdtemp(prefix="hicache_l3_pp_")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "2",
                "--pp-size",
                "2",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-max-bs",
                "32",
                "--max-total-tokens",
                "14000",
                "--disable-piecewise-cuda-graph",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
                "--enable-hierarchical-cache",
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
