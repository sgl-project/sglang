"""MI35x DeepSeek-R1-0528 FP8 HiCache Nightly Test (8-GPU)

Regression guard: launches DeepSeek-R1-0528 (native FP8, MLA, aiter attention
backend) on MI35x with the full L1+L2+L3 HiCache hierarchy wired up
(``--enable-hierarchical-cache --hicache-storage-backend file``), then runs
GSM8K few-shot completion and asserts the accuracy still matches the
established threshold. The goal is to catch regressions where HiCache
breaks DSR1-0528 generation correctness, not to stress-test the cascade
overflow path.

Acceptance: GSM8K (1319 questions, 5-shot, completion API) score >= 0.93,
matching ``test_deepseek_r1_eval_mi35x.py`` /
``test_deepseek_r1_eval_amd.py``.

Registry: nightly-amd-8-gpu-mi35x-deepseek-r1-hicache suite.
"""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# DSR1-0528 can spend 20+ min in weight loading on MI35x before warmup.
register_amd_ci(
    est_time=5400,
    suite="nightly-amd-8-gpu-mi35x-deepseek-r1-hicache",
    nightly=True,
)

DEEPSEEK_R1_MODEL_PATH = "deepseek-ai/DeepSeek-R1-0528"
SERVER_LAUNCH_TIMEOUT = 3600

# Threshold matches the existing nightly AMD DSR1-0528 accuracy tests:
#   test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py
#   test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py
GSM8K_ACCURACY_THRESHOLD = 0.93
GSM8K_NUM_EXAMPLES = None
GSM8K_NUM_THREADS = 64


class TestDeepSeekR1HiCacheMI35x(CustomTestCase):
    """DSR1-0528 FP8 + HiCache (L1+L2+L3) GSM8K regression test for MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_R1_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.l3_storage_dir = tempfile.mkdtemp(prefix="dsr1-hicache-l3-")

        # cascade_dsr1_lite.sh on ROCm DSR1-0528 requires:
        #   SGLANG_USE_AITER=1                  -> aiter prefill/decode path
        #   ROCM_QUICK_REDUCE_QUANTIZATION=NONE -> keep allreduce fp16/bf16
        #   SGLANG_AITER_FP8_PREFILL_ATTN=0     -> disable PR #18528 FP8,
        #                                          prefill kernel flash_attn_varlen_func
        #                                          which is incompatible with
        #                                          DSR1-0528 + page_size=64
        env = {
            **os.environ,
            "SGLANG_USE_AITER": "1",
            "ROCM_QUICK_REDUCE_QUANTIZATION": "NONE",
            "SGLANG_AITER_FP8_PREFILL_ATTN": "0",
            "SAFETENSORS_FAST_GPU": "1",
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.l3_storage_dir,
        }

        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.6",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--attention-backend",
            "aiter",
            "--page-size",
            "64",
            "--context-length",
            "65536",
            "--chunked-prefill-size",
            "32768",
            "--max-prefill-tokens",
            "32768",
            "--watchdog-timeout",
            "1200",
            "--enable-metrics",
            "--enable-cache-report",
            # HiCache hierarchy: L1 (GPU radix) + L2 (host pinned) + L3 (file).
            # hicache-ratio=2 gives L2 = 2 * L1 (SGLang requires L1 <= L2).
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "2",
            "--hicache-io-backend",
            "kernel",
            # page_first + kernel io is the recommended pair; page_first_direct
            # would silently downgrade io to direct (server_args.py:3108-3125).
            "--hicache-mem-layout",
            "page_first",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "best_effort",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            env=env,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)
        if getattr(cls, "l3_storage_dir", None):
            shutil.rmtree(cls.l3_storage_dir, ignore_errors=True)

    def test_gsm8k(self):
        """GSM8K few-shot completion against the HiCache-enabled DSR1-0528."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            num_examples=GSM8K_NUM_EXAMPLES,
            num_threads=GSM8K_NUM_THREADS,
            max_tokens=512,
            temperature=0.0,
        )
        metrics = run_eval(args)
        print(f"{metrics=}", flush=True)
        score = metrics["score"]

        if is_in_ci():
            write_github_step_summary(
                "### DeepSeek-R1-0528 FP8 HiCache GSM8K (MI35x)\n\n"
                "| Model | Examples | Max Parallel | Score | Threshold | Latency |\n"
                "| ----- | --------- | ------------ | ----- | --------- | ------- |\n"
                f"| {self.model} | full | {GSM8K_NUM_THREADS} | "
                f"{score:.3f} | {GSM8K_ACCURACY_THRESHOLD:.2f} | "
                f"{metrics.get('latency', 0):.1f}s |\n"
            )

        self.assertGreaterEqual(score, GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
