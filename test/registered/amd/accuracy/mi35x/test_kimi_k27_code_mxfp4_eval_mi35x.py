"""MI35x Kimi-K2.7-Code-MXFP4 aiter MLA backend accuracy tests (4-GPU)

Tests Kimi-K2.7-Code-MXFP4 with the aiter unified attention backend on MI35x.
This model uses mixed quantization: mxfp4 for MoE layers and fp8 per-channel
for attention projections (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj,
o_proj). The per-channel fp8 detection fix ensures the correct kernel path is
selected for each layer type.

Registry: nightly-amd-4-gpu-mi35x-kimi-k27-code-mxfp4-aiter-mla suite
"""

import os
import unittest
from dataclasses import dataclass
from typing import List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=7200,
    suite="nightly-amd-4-gpu-mi35x-kimi-k27-code-mxfp4-aiter-mla",
    nightly=True,
)


@dataclass
class ModelConfig:
    """Configuration for a model variant to test."""

    model_path: str
    tp_size: int = 4
    accuracy_threshold: float = 0.94
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


def get_kimi_k27_code_mxfp4_models() -> List[ModelConfig]:
    """Get Kimi-K2.7-Code-MXFP4 model configurations for MI35x."""
    common_kwargs = {
        "model_path": "amd/Kimi-K2.7-Code-MXFP4",
        "tp_size": 4,
        "accuracy_threshold": 0.94,
        "timeout": 3600,
    }
    common_args = [
        "--attention-backend",
        "aiter",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.90",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--trust-remote-code",
        "--watchdog-timeout",
        "1200",
        "--enable-aiter-allreduce-fusion",
    ]

    return [
        ModelConfig(
            **common_kwargs,
            variant="default",
            other_args=common_args,
        ),
    ]


class TestKimiK27CodeMXFP4AiterMlaEvalMI35x(unittest.TestCase):
    """Kimi-K2.7-Code-MXFP4 aiter MLA backend accuracy tests on MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.models = get_kimi_k27_code_mxfp4_models()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

    def test_kimi_k27_code_mxfp4_accuracy(self):
        """Test Kimi-K2.7-Code-MXFP4 with GSM8K completion benchmark."""
        from types import SimpleNamespace

        from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k

        all_results = []
        summary = "### Kimi-K2.7-Code-MXFP4 aiter MLA (MI35x)\n\n"
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
                        args = SimpleNamespace(
                            num_shots=8,
                            data_path=None,
                            num_questions=self.num_questions,
                            parallel=self.num_questions,
                            max_new_tokens=512,
                            host="http://127.0.0.1",
                            port=int(self.base_url.split(":")[-1]),
                        )
                        metrics = run_eval_few_shot_gsm8k(args)
                        acc = metrics["accuracy"]

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
