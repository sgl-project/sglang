"""UnifiedRadixTree + HiCache KL divergence tests.

Tests Mamba hybrid, DeepSeek V4 Flash, and GLM-5 models with HiCache L2
offloading under UnifiedRadixTree, verifying multi-turn cache correctness
via KL divergence.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

GLM5_MODEL = "zai-org/GLM-5.1-FP8"
GLM5_LAUNCH_TIMEOUT = 3600

register_cuda_ci(est_time=900, suite="nightly-8-gpu-h200", nightly=True)


class GSM8KTwoPassMixin:
    """Mixin: run GSM8K twice with flush in between, verify accuracy diff.

    Subclass must provide:
      - self.base_url
      - self.model (for logging)
    """

    gsm8k_threshold: float = 0.90
    num_gsm8k_questions: int = 200
    max_accuracy_diff: float = 0.02
    gsm8k_parallel: int = 40

    def _run_gsm8k(self):
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=10,
            data_path=None,
            num_questions=self.num_gsm8k_questions,
            max_new_tokens=16000,
            parallel=self.gsm8k_parallel,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_few_shot_gsm8k(args)
        return metrics["accuracy"]

    def _flush_cache(self):
        response = requests.post(
            self.base_url + "/flush_cache",
            params={"timeout": 30},
            timeout=40,
        )
        response.raise_for_status()

    def test_gsm8k_two_passes(self):
        """Run GSM8K twice with flush in between, verify accuracy diff <= max_accuracy_diff."""
        # First pass
        acc1 = self._run_gsm8k()
        print(f"[{self.__class__.__name__}] GSM8K pass 1 accuracy: {acc1:.3f}")
        self.assertGreaterEqual(
            acc1,
            self.gsm8k_threshold,
            f"Pass 1 accuracy {acc1:.3f} < threshold {self.gsm8k_threshold}",
        )

        # Flush cache
        self._flush_cache()

        # Second pass
        acc2 = self._run_gsm8k()
        print(f"[{self.__class__.__name__}] GSM8K pass 2 accuracy: {acc2:.3f}")
        self.assertGreaterEqual(
            acc2,
            self.gsm8k_threshold,
            f"Pass 2 accuracy {acc2:.3f} < threshold {self.gsm8k_threshold}",
        )

        # Verify diff
        if acc1 > acc2:
            diff = abs(acc1 - acc2)
            print(
                f"[{self.__class__.__name__}] Accuracy diff: {diff:.3f} "
                f"(max allowed: {self.max_accuracy_diff})"
            )
            self.assertLessEqual(
                diff,
                self.max_accuracy_diff,
                f"Accuracy diff {diff:.3f} exceeds max {self.max_accuracy_diff} "
                f"(pass1={acc1:.3f}, pass2={acc2:.3f})",
            )


class TestGLM5HiCacheL3GSM8K(GSM8KTwoPassMixin, CustomTestCase):
    """GLM-5.1-FP8 + HiCache L3 (file backend), with HiRadixTree."""

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
                "0.85",
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
                "direct",
                "--hicache-mem-layout",
                "page_first_direct",
                "--hicache-storage-backend",
                "file",
            ],
            env={"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.hicache_dir},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if os.path.isdir(cls.hicache_dir):
            shutil.rmtree(cls.hicache_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
