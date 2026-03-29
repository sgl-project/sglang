"""AMD GROK GSM8K Completion Evaluation Test (8-GPU)

Tests GROK models (Grok-1 FP8, Grok-1 INT4, Grok-2) using
few-shot completion benchmark on MI300X.

Registry: nightly-amd-8-gpu-grok suite
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

# DISABLED: Split into individual files for each model variant
# See: test_grok1_fp8_eval_amd.py, test_grok1_int4_eval_amd.py, test_grok2_eval_amd.py
register_amd_ci(
    est_time=2700,
    suite="nightly-amd-8-gpu-grok",
    nightly=True,
    disabled="Split into test_grok1_fp8_eval_amd.py, test_grok1_int4_eval_amd.py, test_grok2_eval_amd.py",
)


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 8
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    tokenizer_path: Optional[str] = None
    timeout: Optional[int] = None

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}


# GROK models for MI300X
GROK_MODELS = [
    # GROK1-FP8
    ModelConfig(
        model_path="lmzheng/grok-1",
        tp_size=8,
        accuracy_threshold=0.80,
        timeout=3600,
        tokenizer_path="Xenova/grok-1-tokenizer",
        other_args=[
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "RCCL_MSCCL_ENABLE": "0",
            "SGLANG_USE_AITER": "1",
            "SGLANG_INT4_WEIGHT": "0",
        },
    ),
    # GROK1-INT4
    ModelConfig(
        model_path="amd/grok-1-W4A8KV8",
        tp_size=8,
        accuracy_threshold=0.80,
        timeout=3600,
        tokenizer_path="Xenova/grok-1-tokenizer",
        other_args=[
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "RCCL_MSCCL_ENABLE": "0",
            "SGLANG_USE_AITER": "1",
            "SGLANG_INT4_WEIGHT": "1",
        },
    ),
    # GROK2
    ModelConfig(
        model_path="xai-org/grok-2",
        tp_size=8,
        accuracy_threshold=0.915,
        timeout=3600,
        tokenizer_path="alvarobartt/grok-2-tokenizer",
        other_args=[
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--trust-remote-code",
        ],
        env_vars={
            "RCCL_MSCCL_ENABLE": "0",
            "SGLANG_USE_AITER": "1",
            "SGLANG_INT4_WEIGHT": "0",
        },
    ),
]


class TestGrokEvalAMD(unittest.TestCase):
    """GROK GSM8K Completion Evaluation Test for AMD MI300X."""

    @classmethod
    def setUpClass(cls):
        cls.models = GROK_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_grok_accuracy(self):
        """Test GROK models with GSM8K completion benchmark."""
        all_results = []
        summary = "### GROK Models (MI300X)\n\n"
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
                if config.tokenizer_path:
                    other_args.extend(["--tokenizer-path", config.tokenizer_path])
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
