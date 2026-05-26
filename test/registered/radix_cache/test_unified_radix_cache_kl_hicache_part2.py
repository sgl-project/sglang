import os
import shutil
import tempfile
import unittest

from test_unified_radix_cache_kl_hicache_nightly import AccuracyTwoPassMixin

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MAMBA_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MAMBA_TRACK_INTERVAL = 128

register_cuda_ci(est_time=768, stage="base-c", runner_config="8-gpu-h200")


class TestUnifiedMambaHiCacheL3(AccuracyTwoPassMixin, CustomTestCase):
    """Mamba hybrid + HiCache L3 (file backend) + UnifiedRadixCache."""

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
                "2048",
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
