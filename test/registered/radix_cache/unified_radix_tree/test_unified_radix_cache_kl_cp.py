import unittest

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

register_cuda_ci(est_time=950, stage="extra-b", runner_config="4-gpu-h100")

QWEN3_30B_MODEL = "Qwen/Qwen3-30B-A3B-FP8"


class TestUnifiedQwen3HiCacheCP(UnifiedRadixTreeTestMixin, CustomTestCase):
    """Qwen3-30B-A3B-FP8 + HiCache + CP + UnifiedRadixCache."""

    hicache_io_backend = "kernel"
    hicache_mem_layout = "page_first"
    max_running_requests = 32
    kl_threshold = 0.005
    gsm8k_threshold = 0.7
    mmlu_threshold = 0.7

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--moe-dp-size",
                "1",
                "--ep-size",
                "4",
                "--attn-cp-size",
                "2",
                "--enable-prefill-context-parallel",
                "--mem-fraction-static",
                "0.8",
                "--cuda-graph-max-bs",
                "32",
                "--max-running-requests",
                str(cls.max_running_requests),
                "--disable-piecewise-cuda-graph",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                "2",
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


if __name__ == "__main__":
    unittest.main()
