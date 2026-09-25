"""BMG DeepSeek-V2-Lite-Chat-FP8 sgl-eval GSM8K Chat Evaluation Test (1-GPU)

Tests DeepSeek-V2-Lite-Chat-FP8 with basic configuration using sgl-eval chat
benchmark on XPU.
"""

import os
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_xpu_ci(est_time=2400, suite="nightly-xpu-1-gpu", nightly=True)


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    accuracy_threshold: float
    tp_size: int = 1
    pp_size: int = 1
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


# DeepSeek-V2-Lite-Chat-FP8 models for XPU - only basic variant for faster CI
# DP, and TC variants removed to reduce test time from ~1h to ~10min
XPU_DEEPSEEK_V2_LITE_CHAT_FP8_MODELS = [
    # DeepSeek-V2-Lite-Chat-FP8 basic
    ModelConfig(
        model_path=DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
        tp_size=1,
        pp_size=1,
        accuracy_threshold=0.62,
        timeout=300,
        variant="basic",
        other_args=[
            "--attention-backend",
            "triton",
            "--moe-runner-backend",
            "triton",
            "--mem-fraction-static",
            "0.7",
            "--trust-remote-code",
            "--watchdog-timeout",
            "120",  # 2 minutes for weight loading
        ],
        env_vars={
            "SGLANG_WARMUP_TIMEOUT": "600",
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


class TestDeepSeekV2LiteChatFP8EvalXPU(unittest.TestCase):
    """DeepSeek-V2-Lite-Chat-FP8 sgl-eval GSM8K Chat Evaluation Test for XPU BMG."""

    @classmethod
    def setUpClass(cls):
        cls.models = XPU_DEEPSEEK_V2_LITE_CHAT_FP8_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_deepseek_v2_lite_chat_fp8_accuracy(self):
        """Test DeepSeek-V2-Lite-Chat-FP8 models with sgl-eval GSM8K chat benchmark."""
        all_results = []
        summary = "### DeepSeek-V2-Lite-Chat-FP8 Models (BMG)\n\n"
        summary += "| Model | Variant | TP | Accuracy | Threshold | Status |\n"
        summary += "| ----- | ------- | -- | -------- | --------- | ------ |\n"

        for config in self.models:
            display_name = config.get_display_name()
            with self.subTest(model=display_name):
                print(f"\n{'=' * 60}")
                print(f"Testing: {display_name}")
                print(f"{'=' * 60}")

                env = os.environ.copy()
                for key, value in config.env_vars.items():
                    env[key] = value

                other_args = list(config.other_args)
                other_args.extend(["--tp", str(config.tp_size)])
                other_args.extend(["--pp-size", str(config.pp_size)])
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
                        summary += f"| {config.model_path} | {config.variant or 'N/A'} | {config.tp_size} | {config.pp_size} | {acc:.3f} | {config.accuracy_threshold} | {status} |\n"
                    except Exception as e:
                        print(e)

                    finally:
                        kill_process_tree(process.pid)

                except Exception as e:
                    print(e)
                    summary += f"| {config.model_path} | {config.variant or 'N/A'} | {config.tp_size} | {config.pp_size} | N/A | {config.accuracy_threshold} | ❌ ERROR |\n"
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
