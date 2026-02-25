"""AMD Kimi-K2 GSM8K Completion Evaluation Test (8-GPU)

Tests moonshotai/Kimi-K2-Instruct-0905 with GSM8K few-shot benchmark on MI325.

Registry: nightly-amd-accuracy-8-gpu-kimi-k2 suite
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Register for AMD CI - Kimi K2 accuracy test (~60 min)
register_amd_ci(est_time=3600, suite="nightly-amd-accuracy-8-gpu-kimi-k2", nightly=True)

KIMI_K2_MODEL_PATH = "moonshotai/Kimi-K2-Instruct-0905"
SERVER_LAUNCH_TIMEOUT = 3600
ACCURACY_THRESHOLD = 0.94


class TestKimiK2EvalAMD(CustomTestCase):
    """Kimi-K2 GSM8K Completion Evaluation Test for AMD MI325."""

    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_K2_MODEL_PATH
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

    def test_kimi_k2_gsm8k_accuracy(self):
        """Test Kimi-K2 with GSM8K few-shot completion benchmark."""
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        acc = metrics["accuracy"]

        passed = acc >= ACCURACY_THRESHOLD
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  accuracy={acc:.3f} threshold={ACCURACY_THRESHOLD} {status}")

        if is_in_ci():
            summary = "### Kimi-K2 Model (MI325)\n\n"
            summary += "| Model | TP | Accuracy | Threshold | Status |\n"
            summary += "| ----- | -- | -------- | --------- | ------ |\n"
            summary += f"| {KIMI_K2_MODEL_PATH} | 8 | {acc:.3f} | {ACCURACY_THRESHOLD} | {status} |\n"
            write_github_step_summary(summary)

        self.assertGreaterEqual(
            acc,
            ACCURACY_THRESHOLD,
            f"Kimi-K2 accuracy {acc:.3f} below threshold {ACCURACY_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
