"""UnifiedRadixTree + HiCache accuracy tests for GLM-5.2.

Runs GLM-5.2-FP8 with HiCache L3 (file backend) under UnifiedRadixTree,
verifying accuracy stays stable across a cache flush.
"""

import os
import shutil
import tempfile
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.unified_radix_cache_kit import AccuracyTwoPassMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

GLM5_MODEL = "zai-org/GLM-5.2-FP8"
GLM5_LAUNCH_TIMEOUT = 3600

register_cuda_ci(est_time=900, stage="extra-b", runner_config="8-gpu-h200")


class TestGLM5UnifiedRadixCacheL3Accuracy(AccuracyTwoPassMixin, CustomTestCase):
    """GLM-5.2-FP8 + HiCache L3 (file backend), with UnifiedRadixTree."""

    @classmethod
    def setUpClass(cls):
        cls.model = GLM5_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.hicache_dir = tempfile.mkdtemp(prefix="hicache_l3_")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=GLM5_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "8",
                "--page-size",
                "64",
                "--mem-fraction-static",
                "0.8",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 64}',
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
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ],
            env={
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.hicache_dir,
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if os.path.isdir(cls.hicache_dir):
            shutil.rmtree(cls.hicache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
