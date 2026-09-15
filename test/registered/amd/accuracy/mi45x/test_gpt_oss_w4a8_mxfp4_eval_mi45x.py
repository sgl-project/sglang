"""MI45x GPT-OSS W4A8 MXFP4-FP8 sgl-eval GSM8K Chat Evaluation Test (1-GPU)

Tests the AMD Quark `gpt-oss-120b-w-mxfp4-a-fp8` checkpoint (MXFP4
weights + static per-tensor FP8 activations) using sgl-eval chat
benchmark on MI45x.

Registry: nightly-amd-1-gpu-mi45x suite
"""

import os
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# Register for AMD CI - MI45x GPT-OSS W4A8 MXFP4-FP8 accuracy test (~15 min)
register_amd_ci(est_time=900, suite="nightly-amd-1-gpu-mi45x", nightly=True)


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 1
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    timeout: Optional[int] = None

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}


# GPT-OSS Quark W4A8 MXFP4-FP8 models for MI45x
MI45X_GPT_OSS_W4A8_MXFP4_MODELS = [
    ModelConfig(
        model_path="amd/gpt-oss-120b-w-mxfp4-a-fp8",
        tp_size=1,
        accuracy_threshold=0.79,
        timeout=900,
        other_args=[
            "--prefill-attention-backend",
            "triton",
            "--decode-attention-backend",
            "aiter",
            "--max-running-requests",
            "128",
            "--mem-fraction-static",
            "0.9",
            "--disable-radix-cache",
            "--page-size",
            "64",
            "--trust-remote-code",
        ],
        # AITER MXFP4 fused-MoE for gpt-oss uses the SEPARATED gate/up tile
        # layout (matches `gptoss_fp4_tuned_fmoe.csv` flydsl entries and the
        # Mxfp4MoEMethod weight shuffle). Other AITER MXFP4 callers default
        # to INTERLEAVE, so opt out explicitly here.
        env_vars={
            "AITER_FORCE_A8W4": "1",
            "ENABLE_CK": "0",
            "HSA_COREDUMP_PATTERN": "/dev/null",
            "SGLANG_USE_AITER": "1",
            "SGLANG_USE_AITER_MOE_GU_ITLV": "1",
            "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
        },
    ),
]


def run_gsm8k_benchmark(
    base_url: str,
    num_questions: int = 200,
    parallel: int = 64,
) -> Tuple[float, float, float]:
    """Run the canonical sgl-eval GSM8K benchmark."""
    metrics = run_eval(
        SimpleNamespace(
            eval_name="gsm8k",
            base_url=base_url,
            num_examples=num_questions,
            num_threads=parallel,
            max_tokens=2048,
        )
    )
    return metrics["score"], metrics["invalid"], metrics["latency"]


class TestGptOssW4A8Mxfp4EvalMI45x(unittest.TestCase):
    """GPT-OSS W4A8 MXFP4-FP8 sgl-eval GSM8K Chat Evaluation Test for AMD MI45x."""

    @classmethod
    def setUpClass(cls):
        cls.models = MI45X_GPT_OSS_W4A8_MXFP4_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_gpt_oss_accuracy(self):
        """Test GPT-OSS W4A8 MXFP4-FP8 models with sgl-eval GSM8K chat benchmark."""
        all_results = []
        summary = "### GPT-OSS W4A8 MXFP4-FP8 Models (MI45x)\n\n"
        summary += "| Model | TP | Accuracy | Threshold | Status |\n"
        summary += "| ----- | -- | -------- | --------- | ------ |\n"

        for config in self.models:
            with self.subTest(model=config.model_path):
                print(f"\n{'=' * 60}")
                print(f"Testing: {config.model_path}")
                print(f"{'=' * 60}")

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
