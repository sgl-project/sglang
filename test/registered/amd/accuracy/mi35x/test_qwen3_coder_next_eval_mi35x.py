"""MI35x Qwen3-Coder-Next sgl-eval GSM8K Chat Evaluation Test (8-GPU)

Tests Qwen3-Coder-Next model with basic and MTP configurations
using sgl-eval chat benchmark on MI35x.

Registry: nightly-amd-8-gpu-mi35x-qwen3-coder-next suite
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

# Register for AMD CI - MI35x Qwen3-Coder-Next accuracy test
register_amd_ci(est_time=3600, suite="nightly-amd-8-gpu-mi35x", nightly=True)


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


def get_qwen3_coder_next_models() -> List[ModelConfig]:
    """Get Qwen3-Coder-Next model configurations for MI35x."""
    common_kwargs = {
        "model_path": "Qwen/Qwen3-Coder-Next",
        "tp_size": 8,
        "accuracy_threshold": 0.90,
        "timeout": 3600,
    }
    common_args = [
        "--attention-backend",
        "aiter",
        "--chunked-prefill-size",
        "131072",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.8",
        "--trust-remote-code",
    ]
    return [
        # Basic — matches run_qwen3-coder-next_spec.sh
        ModelConfig(
            **common_kwargs,
            variant="basic",
            other_args=common_args
            + [
                "--kv-cache-dtype",
                "fp8_e4m3",
            ],
        ),
        # MTP (speculative decoding)
        # TODO: Support MTP with fp8 kv cache on gfx950.
        # Note: no --kv-cache-dtype fp8_e4m3 because Triton extend_attention
        # used by MTP does not support fp8 kv cache on gfx950.
        ModelConfig(
            **common_kwargs,
            variant="mtp",
            other_args=common_args
            + [
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ],
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


class TestQwen3CoderNextEvalMI35x(unittest.TestCase):
    """Qwen3-Coder-Next sgl-eval GSM8K Chat Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.models = get_qwen3_coder_next_models()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_qwen3_coder_next_accuracy(self):
        """Test Qwen3-Coder-Next models with sgl-eval GSM8K chat benchmark."""
        all_results = []
        summary = "### Qwen3-Coder-Next Models (MI35x)\n\n"
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
                        status = "PASS" if passed else "FAIL"
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
                    summary += f"| {config.model_path} | {config.variant or 'N/A'} | {config.tp_size} | N/A | {config.accuracy_threshold} | ERROR |\n"
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
