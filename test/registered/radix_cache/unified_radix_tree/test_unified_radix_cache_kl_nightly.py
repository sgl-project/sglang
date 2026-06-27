"""UnifiedRadixTree + HiCache KL divergence tests.

Tests Mamba hybrid, DeepSeek V4 Flash, and GLM-5 models with HiCache L2
offloading under UnifiedRadixTree, verifying multi-turn cache correctness
via KL divergence.
"""

import os
import random
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests
import torch

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


class AccuracyTwoPassMixin:
    """Mixin: run an eval twice with flush in between, verify accuracy diff.

    Subclass must provide:
      - self.base_url
      - self.model (for logging)
    """

    gsm8k_threshold: float = 0.90
    num_gsm8k_questions: int = 200
    gsm8k_parallel: int = 40

    max_accuracy_diff: float = 0.02

    l3_prefetch_page_size: int = 64
    l3_prefetch_prompt_pages: int = 16
    # Max tokens that may stay uncached on a full-prompt re-request; the bound
    # depends on model architecture. Defaults to page_size; subclasses override.
    l3_prefetch_max_uncached_tokens: int = None

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

    def _two_pass(self, name: str, run_fn, threshold: float):
        # First pass
        acc1 = run_fn()
        print(f"[{self.__class__.__name__}] {name} pass 1 accuracy: {acc1:.3f}")
        self.assertGreaterEqual(
            acc1,
            threshold,
            f"{name} pass 1 accuracy {acc1:.3f} < threshold {threshold}",
        )

        # Flush cache
        self._flush_cache()

        # Second pass
        acc2 = run_fn()
        print(f"[{self.__class__.__name__}] {name} pass 2 accuracy: {acc2:.3f}")
        self.assertGreaterEqual(
            acc2,
            threshold,
            f"{name} pass 2 accuracy {acc2:.3f} < threshold {threshold}",
        )

        # Verify diff (only fail when 2nd pass regressed)
        if acc1 > acc2:
            diff = abs(acc1 - acc2)
            print(
                f"[{self.__class__.__name__}] {name} accuracy diff: {diff:.3f} "
                f"(max allowed: {self.max_accuracy_diff})"
            )
            self.assertLessEqual(
                diff,
                self.max_accuracy_diff,
                f"{name} accuracy diff {diff:.3f} exceeds max {self.max_accuracy_diff} "
                f"(pass1={acc1:.3f}, pass2={acc2:.3f})",
            )

    def test_gsm8k_two_passes(self):
        """Run GSM8K twice with flush in between, verify accuracy diff <= max_accuracy_diff."""
        self._two_pass("GSM8K", self._run_gsm8k, self.gsm8k_threshold)

    def test_l3_prefetch_full_prefix_hit_after_flush(self):
        from sglang.test.kl_test_utils import _flush_cache, _generate

        page = int(self.l3_prefetch_page_size)
        n_tokens = page * int(self.l3_prefetch_prompt_pages)
        max_uncached = int(
            self.l3_prefetch_max_uncached_tokens
            if self.l3_prefetch_max_uncached_tokens is not None
            else page
        )

        rng = random.Random(987)
        input_ids = [rng.randint(1, 30000) for _ in range(n_tokens)]

        _generate(self.base_url, [input_ids], max_new_tokens=4)
        _flush_cache(self.base_url)
        results = _generate(self.base_url, [input_ids], max_new_tokens=4)
        cached = int(results[0]["meta_info"]["cached_tokens"])

        expected_min = n_tokens - max_uncached
        self.assertGreaterEqual(
            cached,
            expected_min,
            f"cached_tokens={cached} < {expected_min} (= input_len - {max_uncached})",
        )


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGLM5HiRadixCacheL3Accuracy(AccuracyTwoPassMixin, CustomTestCase):
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
                "kernel",
                "--hicache-mem-layout",
                "page_first",
                "--hicache-storage-backend",
                "file",
            ],
            env={
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.hicache_dir,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        if os.path.isdir(cls.hicache_dir):
            shutil.rmtree(cls.hicache_dir, ignore_errors=True)


@unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
class TestGLM5UnifiedRadixCacheL3Accuracy(AccuracyTwoPassMixin, CustomTestCase):
    """GLM-5.1-FP8 + HiCache L3 (file backend), with UnifiedRadixTree."""

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
                "kernel",
                "--hicache-mem-layout",
                "page_first",
                "--hicache-storage-backend",
                "file",
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
