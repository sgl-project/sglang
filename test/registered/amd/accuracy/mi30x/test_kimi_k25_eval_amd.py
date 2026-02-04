"""AMD Kimi-K2.5 MMMU Evaluation Test (8-GPU)

Tests moonshotai/Kimi-K2.5 with MMMU benchmark on MI325.

Kimi-K2.5 is a multimodal model (VLM) based on DeepSeek V3 architecture
with vision capabilities.

Registry: nightly-amd-accuracy-8-gpu-kimi-k25 suite
"""

import os
import unittest
from types import SimpleNamespace

import requests

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

# Register for AMD CI - Kimi K2.5 accuracy test (~60 min)
register_amd_ci(
    est_time=3600, suite="nightly-amd-accuracy-8-gpu-kimi-k25", nightly=True
)

KIMI_K25_MODEL_PATH = "moonshotai/Kimi-K2.5"
SERVER_LAUNCH_TIMEOUT = 3600
ACCURACY_THRESHOLD = 0.50  # Conservative threshold for MMMU benchmark


class TestKimiK25EvalAMD(CustomTestCase):
    """Kimi-K2.5 MMMU Evaluation Test for AMD MI325."""

    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_K25_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--decode-attention-backend",
            "triton",
            "--prefill-attention-backend",
            "aiter",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_ROCM_FUSED_DECODE_MLA"] = "0"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_kimi_k25_mmmu_accuracy(self):
        """Test Kimi-K2.5 with MMMU benchmark."""
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=KIMI_K25_MODEL_PATH,
            eval_name="mmmu",
            num_examples=100,
            num_threads=64,
            max_tokens=30,
        )
        metrics = run_eval(args)
        score = metrics["score"]

        passed = score >= ACCURACY_THRESHOLD
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  score={score:.3f} threshold={ACCURACY_THRESHOLD} {status}")

        if is_in_ci():
            summary = "### Kimi-K2.5 Model (MI325)\n\n"
            summary += "| Model | TP | Score | Threshold | Status |\n"
            summary += "| ----- | -- | ----- | --------- | ------ |\n"
            summary += f"| {KIMI_K25_MODEL_PATH} | 8 | {score:.3f} | {ACCURACY_THRESHOLD} | {status} |\n"
            write_github_step_summary(summary)

        self.assertGreaterEqual(
            score,
            ACCURACY_THRESHOLD,
            f"Kimi-K2.5 score {score:.3f} below threshold {ACCURACY_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
