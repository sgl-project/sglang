"""AMD Gemma 4 mgsm_en Evaluation Test (2-GPU)

Tests Gemma 4 instruction-tuned models on mgsm_en benchmark using chat completions
on MI325/MI300X. All Gemma 4 models require the Triton attention backend for
bidirectional image-token attention on AMD GPUs.

Ref: https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-gemma-4-on-amd-processors-and-gpus.html
Model support: https://github.com/sgl-project/sglang/pull/21952

Registry: nightly-amd-accuracy-2-gpu-gemma4 suite
"""

import os
import time
import unittest
from dataclasses import dataclass, field
from typing import List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-accuracy-2-gpu-gemma4",
    nightly=True,
)


@dataclass
class ModelConfig:
    model_path: str
    tp_size: int = 1
    accuracy_threshold: float = 0.50
    other_args: List[str] = field(default_factory=list)
    env_vars: dict = field(default_factory=dict)
    timeout: Optional[int] = None


GEMMA4_MODELS = [
    ModelConfig(
        model_path="google/gemma-4-31B-it",
        tp_size=1,
        accuracy_threshold=0.90,
        timeout=1800,
        other_args=[
            "--attention-backend",
            "triton",
            "--watchdog-timeout",
            "1200",
        ],
    ),
]


class TestGemma4EvalAMD(CustomTestCase):
    """Gemma 4 mgsm_en Evaluation Test for AMD MI325/MI300X."""

    @classmethod
    def setUpClass(cls):
        cls.models = GEMMA4_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_threads = 1024

    def test_gemma4_accuracy(self):
        """Test Gemma 4 models with mgsm_en chat completions benchmark."""
        all_results = []
        summary = "### Gemma 4 Models (MI325)\n\n"
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
                        model_start = time.time()
                        metrics = run_eval(
                            type(
                                "Args",
                                (),
                                {
                                    "base_url": self.base_url,
                                    "model": config.model_path,
                                    "eval_name": "mgsm_en",
                                    "num_examples": None,
                                    "num_threads": self.num_threads,
                                },
                            )()
                        )
                        eval_time = time.time() - model_start
                        acc = metrics["score"]
                        passed = acc >= config.accuracy_threshold
                        status = "PASS" if passed else "FAIL"
                        print(
                            f"  accuracy={acc:.3f} threshold={config.accuracy_threshold}"
                            f" time={eval_time:.0f}s {status}"
                        )

                        all_results.append(
                            {
                                "model": config.model_path,
                                "accuracy": acc,
                                "passed": passed,
                            }
                        )
                        summary += (
                            f"| {config.model_path} | {config.tp_size}"
                            f" | {acc:.3f} | {config.accuracy_threshold}"
                            f" | {'✅ PASS' if passed else '❌ FAIL'} |\n"
                        )

                    finally:
                        kill_process_tree(process.pid)

                except Exception as e:
                    summary += (
                        f"| {config.model_path} | {config.tp_size}"
                        f" | N/A | {config.accuracy_threshold} | ❌ ERROR |\n"
                    )
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
