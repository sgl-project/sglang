"""AMD DeepSeek-R1 GSM8K Completion Evaluation Test (8-GPU)

Tests DeepSeek-R1-0528 with multiple configurations (basic, MTP, DP, TC)
using few-shot completion benchmark on MI300X.

Registry: nightly-amd-8-gpu-deepseek-r1 suite
"""

import os
import unittest
from dataclasses import dataclass
from typing import List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.gsm8k_completion_kit import run_gsm8k_benchmark
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Register for AMD CI - DeepSeek-R1 accuracy tests (~120 min)
register_amd_ci(
    est_time=7200, suite="nightly-amd-accuracy-8-gpu-deepseek-r1", nightly=True
)


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 8
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    timeout: Optional[int] = None
    variant: Optional[str] = None

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}

    def get_display_name(self) -> str:
        if self.variant:
            return f"{self.model_path} ({self.variant})"
        return self.model_path


# DeepSeek-R1 models for MI300X
DEEPSEEK_R1_MODELS = [
    # DeepSeek-R1-0528 basic
    ModelConfig(
        model_path="deepseek-ai/DeepSeek-R1-0528",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,
        variant="basic",
        other_args=[
            "--attention-backend",
            "aiter",
            "--chunked-prefill-size",
            "131072",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={"SGLANG_USE_AITER": "1"},
    ),
    # DeepSeek-R1-0528 with MTP (EAGLE)
    ModelConfig(
        model_path="deepseek-ai/DeepSeek-R1-0528",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,
        variant="MTP",
        other_args=[
            "--chunked-prefill-size",
            "131072",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--trust-remote-code",
        ],
        env_vars={"SGLANG_USE_AITER": "1"},
    ),
    # DeepSeek-R1-0528 with DP attention
    ModelConfig(
        model_path="deepseek-ai/DeepSeek-R1-0528",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=3600,
        variant="DP",
        other_args=[
            "--chunked-prefill-size",
            "131072",
            "--dp-size",
            "8",
            "--enable-dp-attention",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "SGLANG_USE_ROCM700A": "1",
            "SGLANG_USE_AITER": "1",
        },
    ),
    # DeepSeek-R1-0528 with torch compile
    ModelConfig(
        model_path="deepseek-ai/DeepSeek-R1-0528",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=7200,
        variant="TC",
        other_args=[
            "--chunked-prefill-size",
            "131072",
            "--mem-fraction-static",
            "0.70",
            "--cuda-graph-max-bs",
            "8",
            "--enable-torch-compile",
            "--disable-cuda-graph",
            "--trust-remote-code",
        ],
        env_vars={
            "SGLANG_USE_ROCM700A": "1",
            "SGLANG_USE_AITER": "1",
        },
    ),
]


class TestDeepSeekR1EvalAMD(unittest.TestCase):
    """DeepSeek-R1 GSM8K Completion Evaluation Test for AMD MI300X."""

    @classmethod
    def setUpClass(cls):
        cls.models = DEEPSEEK_R1_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_deepseek_r1_accuracy(self):
        """Test DeepSeek-R1 models with GSM8K completion benchmark."""
        all_results = []
        summary = "### DeepSeek-R1 Models (MI300X)\n\n"
        summary += "| Model | Variant | TP | Accuracy | Threshold | Status |\n"
        summary += "| ----- | ------- | -- | -------- | --------- | ------ |\n"

        for config in self.models:
            display_name = config.get_display_name()
            with self.subTest(model=display_name):
                print(f"\n{'='*60}")
                print(f"Testing: {display_name}")
                print(f"{'='*60}")

                env = os.environ.copy()
                for key, value in config.env_vars.items():
                    env[key] = value

                other_args = list(config.other_args)
                other_args.extend(["--tp", str(config.tp_size)])
                timeout = config.timeout or DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

                try:
                    process = popen_launch_server(
                        model=config.model_path,
                        base_url=self.base_url,
                        timeout=timeout,
                        other_args=other_args,
                        env=env,
                    )

                    try:
                        acc, invalid, latency = run_gsm8k_benchmark(
                            self.base_url, num_questions=self.num_questions
                        )
                        passed = acc >= config.accuracy_threshold
                        status = "✅ PASS" if passed else "❌ FAIL"
                        print(
                            f"  accuracy={acc:.3f} threshold={config.accuracy_threshold} {status}"
                        )

                        all_results.append(
                            {
                                "model": display_name,
                                "accuracy": acc,
                                "passed": passed,
                            }
                        )
                        summary += f"| {config.model_path} | {config.variant or 'N/A'} | {config.tp_size} | {acc:.3f} | {config.accuracy_threshold} | {status} |\n"

                    finally:
                        kill_process_tree(process.pid)

                except Exception as e:
                    summary += f"| {config.model_path} | {config.variant or 'N/A'} | {config.tp_size} | N/A | {config.accuracy_threshold} | ❌ ERROR |\n"
                    all_results.append(
                        {
                            "model": display_name,
                            "accuracy": None,
                            "passed": False,
                            "error": str(e),
                        }
                    )

        if is_in_ci():
            write_github_step_summary(summary)

        failed = [r for r in all_results if not r["passed"]]
        if failed:
            raise AssertionError(f"Failed models: {[r['model'] for r in failed]}")


if __name__ == "__main__":
    unittest.main()
