"""AMD DeepSeek-V3.1 GSM8K Completion Evaluation Test (8-GPU)

Tests DeepSeek-V3.1 model using few-shot completion benchmark on MI300X.

Registry: nightly-amd-8-gpu-deepseek-v31 suite
"""

import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.gsm8k_completion_kit import run_gsm8k_benchmark
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Register for AMD CI - DeepSeek-V3.1 accuracy tests (~60 min)
register_amd_ci(
    est_time=3600, suite="nightly-amd-accuracy-8-gpu-deepseek-v31", nightly=True
)

DEEPSEEK_V31_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V31_MODEL_PATH", "deepseek-ai/DeepSeek-V3-0324"
)


class TestDeepSeekV31EvalAMD(unittest.TestCase):
    """DeepSeek-V3.1 GSM8K Completion Evaluation Test for AMD MI300X."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))
        cls.accuracy_threshold = 0.90

    def test_deepseek_v31_accuracy(self):
        """Test DeepSeek-V3.1 with GSM8K completion benchmark."""
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"

        other_args = [
            "--tp",
            "8",
            "--attention-backend",
            "aiter",
            "--chunked-prefill-size",
            "131072",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ]

        process = popen_launch_server(
            model=DEEPSEEK_V31_MODEL_PATH,
            base_url=self.base_url,
            timeout=3600,
            other_args=other_args,
            env=env,
        )

        try:
            acc, _invalid, latency = run_gsm8k_benchmark(
                self.base_url, num_questions=self.num_questions
            )
            passed = acc >= self.accuracy_threshold
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  accuracy={acc:.3f} threshold={self.accuracy_threshold} {status}")

            summary = f"### DeepSeek-V3.1 (MI300X)\n\n"
            summary += f"| Model | Accuracy | Threshold | Status |\n"
            summary += f"| ----- | -------- | --------- | ------ |\n"
            summary += f"| {DEEPSEEK_V31_MODEL_PATH} | {acc:.3f} | {self.accuracy_threshold} | {status} |\n"

            if is_in_ci():
                write_github_step_summary(summary)

            self.assertGreaterEqual(
                acc,
                self.accuracy_threshold,
                f"Accuracy {acc:.3f} below threshold {self.accuracy_threshold}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
