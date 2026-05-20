"""MI35x GROK2 GSM8K Completion Evaluation Test (8-GPU)

Tests Grok-2 model using few-shot completion benchmark on MI35x.

Registry: nightly-amd-accuracy-8-gpu-mi35x-grok2 suite
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

# Register for AMD CI - GROK2 accuracy tests on MI35x (~25 min)
register_amd_ci(
    est_time=1500, suite="nightly-amd-accuracy-8-gpu-mi35x-grok2", nightly=True
)

GROK2_MODEL_PATH = os.environ.get("GROK2_MODEL_PATH", "xai-org/grok-2")
GROK2_TOKENIZER_PATH = os.environ.get(
    "GROK2_TOKENIZER_PATH", "alvarobartt/grok-2-tokenizer"
)


class TestGrok2EvalMI35x(unittest.TestCase):
    """GROK2 GSM8K Completion Evaluation Test for MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))
        cls.accuracy_threshold = 0.90

    def test_grok2_accuracy(self):
        """Test Grok-2 with GSM8K completion benchmark."""
        env = os.environ.copy()
        env["RCCL_MSCCL_ENABLE"] = "0"
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_INT4_WEIGHT"] = "0"

        other_args = [
            "--tp",
            "8",
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--tokenizer-path",
            GROK2_TOKENIZER_PATH,
            "--trust-remote-code",
        ]

        process = popen_launch_server(
            model=GROK2_MODEL_PATH,
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

            summary = f"### GROK2 (MI35x)\n\n"
            summary += f"| Model | Accuracy | Threshold | Status |\n"
            summary += f"| ----- | -------- | --------- | ------ |\n"
            summary += f"| {GROK2_MODEL_PATH} | {acc:.3f} | {self.accuracy_threshold} | {status} |\n"

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
