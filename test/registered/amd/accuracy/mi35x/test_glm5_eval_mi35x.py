"""MI35x GLM-5 sgl-eval GSM8K Chat Evaluation Test (8-GPU)

Tests GLM-5 with DSA attention backend using sgl-eval chat
benchmark on MI35x.

Registry: nightly-amd-8-gpu-mi35x-glm5 suite
"""

import os
import unittest
from dataclasses import dataclass, field
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

# Register for AMD CI - MI35x GLM-5 accuracy test (~90 min)
register_amd_ci(
    est_time=5400,
    suite="nightly-amd-8-gpu-mi35x-glm5",
    nightly=True,
)


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 8
    accuracy_threshold: float = 0.50
    other_args: List[str] = field(default_factory=list)
    env_vars: dict = field(default_factory=dict)
    timeout: Optional[int] = None
    variant: Optional[str] = None

    def get_display_name(self) -> str:
        if self.variant:
            return f"{self.model_path} ({self.variant})"
        return self.model_path


# GLM-5 models for MI35x - DSA attention backend
MI35X_GLM5_MODELS = [
    # GLM-5 with DSA attention (TP=8)
    ModelConfig(
        model_path="zai-org/GLM-5-FP8",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=5400,
        variant="dsa",
        other_args=[
            "--trust-remote-code",
            "--reasoning-parser",
            "glm45",
            "--tool-call-parser",
            "glm47",
            "--dsa-prefill-backend",
            "tilelang",
            "--dsa-decode-backend",
            "tilelang",
            "--chunked-prefill-size",
            "131072",
            "--mem-fraction-static",
            "0.80",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
        ],
        env_vars={},
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


class TestGLM5EvalMI35x(unittest.TestCase):
    """GLM-5 sgl-eval GSM8K Chat Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.models = MI35X_GLM5_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_glm5_accuracy(self):
        """Test GLM-5 models with sgl-eval GSM8K chat benchmark."""
        all_results = []
        summary = "### GLM-5 Models (MI35x)\n\n"
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
