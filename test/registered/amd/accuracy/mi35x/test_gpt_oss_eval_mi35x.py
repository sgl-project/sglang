"""MI35x GPT-OSS GSM8K Completion Evaluation Test (8-GPU)

Tests GPT-OSS models (openai/gpt-oss-20b, openai/gpt-oss-120b) using
few-shot completion benchmark on MI35x.

Note: MI35x uses openai/* paths, not lmsys/* paths like MI300X.

Registry: nightly-amd-8-gpu-mi35x suite
"""

import os

# Set HF cache for MI35x
os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

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

# Register for AMD CI - MI35x GPT-OSS accuracy tests (~30 min)
register_amd_ci(est_time=1800, suite="nightly-amd-8-gpu-mi35x", nightly=True)


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 8
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    timeout: Optional[int] = None

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}


# GPT-OSS models for MI35x (different paths from MI300X)
MI35X_GPT_OSS_MODELS = [
    ModelConfig(
        model_path="openai/gpt-oss-20b",
        tp_size=8,
        accuracy_threshold=0.47,
        other_args=[
            "--chunked-prefill-size",
            "130172",
            "--max-running-requests",
            "128",
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "triton",
            "--trust-remote-code",
        ],
        env_vars={"SGLANG_USE_AITER": "1"},
    ),
    ModelConfig(
        model_path="openai/gpt-oss-120b",
        tp_size=8,
        accuracy_threshold=0.79,
        timeout=900,
        other_args=[
            "--chunked-prefill-size",
            "130172",
            "--max-running-requests",
            "128",
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "triton",
            "--trust-remote-code",
        ],
        env_vars={"SGLANG_USE_AITER": "1"},
    ),
]


class TestGptOssEvalMI35x(unittest.TestCase):
    """GPT-OSS GSM8K Completion Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.models = MI35X_GPT_OSS_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_gpt_oss_accuracy(self):
        """Test GPT-OSS models with GSM8K completion benchmark."""
        all_results = []
        summary = "### GPT-OSS Models (MI35x)\n\n"
        summary += "| Model | TP | Accuracy | Threshold | Status |\n"
        summary += "| ----- | -- | -------- | --------- | ------ |\n"

        for config in self.models:
            with self.subTest(model=config.model_path):
                print(f"\n{'='*60}")
                print(f"Testing: {config.model_path}")
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
                                "model": config.model_path,
                                "accuracy": acc,
                                "passed": passed,
                            }
                        )
                        summary += f"| {config.model_path} | {config.tp_size} | {acc:.3f} | {config.accuracy_threshold} | {status} |\n"

                    finally:
                        kill_process_tree(process.pid)

                except Exception as e:
                    summary += f"| {config.model_path} | {config.tp_size} | N/A | {config.accuracy_threshold} | ❌ ERROR |\n"
                    all_results.append(
                        {
                            "model": config.model_path,
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
