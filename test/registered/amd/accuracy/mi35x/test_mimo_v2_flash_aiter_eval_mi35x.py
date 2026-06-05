"""MI355X MiMo-V2-Flash aiter backend accuracy test (4-GPU).

MiMo-V2-Flash uses non-square attention (qk_head_dim=192, v_head_dim=128) with a
128-token sliding window across all 48 layers.  The aiter unified-attention kernel
assumes equal K/V head dims, so this PR introduces Triton fallback paths for prefill
and decode when qk_head_dim != v_head_dim.  This test validates that those paths
produce correct outputs by measuring GSM8K accuracy.

NOTE: TP=4 is required.  MiMo-V2-Flash has 64 attention heads; TP=8 would give
64/8 = 8 heads per rank which is too few for the aiter unified-attention path.
TP=4 gives 64/4 = 16 heads per rank.

Registry: nightly-amd-accuracy-8-gpu-mi355x-mimo-v2-flash-aiter suite
"""

import os

os.environ.setdefault("HF_HOME", "/mnt/hf_hub_cache")
os.environ.setdefault("HF_HUB_CACHE", "/mnt/hf_hub_cache")

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
    suite="nightly-amd-accuracy-8-gpu-mi355x-mimo-v2-flash-aiter",
    nightly=True,
)

MIMO_V2_FLASH_LOCAL_PATH = "/mnt/hf_hub_cache/MiMo-V2-Flash"
MIMO_V2_FLASH_HF_MODEL_ID = "XiaomiMiMo/MiMo-V2-Flash"


def get_model_path() -> str:
    """Return effective model path: env var > local cache > HF model ID."""
    env_path = os.environ.get("MIMO_V2_FLASH_MODEL_PATH")
    if env_path:
        return env_path
    if os.path.exists(MIMO_V2_FLASH_LOCAL_PATH):
        return MIMO_V2_FLASH_LOCAL_PATH
    return MIMO_V2_FLASH_HF_MODEL_ID


@dataclass
class ModelConfig:
    """Configuration for a single test variant."""

    model_path: str
    tp_size: int = 4
    accuracy_threshold: float = 0.78
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


def get_mimo_v2_flash_models() -> List[ModelConfig]:
    """Return the list of model configs to test."""
    model_path = get_model_path()
    common_args = [
        "--attention-backend",
        "aiter",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.8",
        "--chunked-prefill-size",
        "131072",
        "--max-running-requests",
        "128",
        "--trust-remote-code",
        "--watchdog-timeout",
        "1200",
    ]
    return [
        ModelConfig(
            model_path=model_path,
            tp_size=4,
            accuracy_threshold=0.80,
            timeout=7200,
            variant="bf16",
            other_args=common_args,
        ),
    ]


class TestMiMoV2FlashAiterEvalMI355x(unittest.TestCase):
    """GSM8K accuracy test for MiMo-V2-Flash with the aiter backend on MI355X.

    Validates that the non-square attention fallback paths (qk_head_dim != v_head_dim)
    introduced for the aiter backend produce correct model outputs.
    """

    @classmethod
    def setUpClass(cls):
        cls.models = get_mimo_v2_flash_models()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

    def test_mimo_v2_flash_accuracy(self):
        """Test MiMo-V2-Flash on GSM8K with the aiter backend."""
        from types import SimpleNamespace

        from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k

        all_results = []
        summary = "### MiMo-V2-Flash aiter backend (MI355X)\n\n"
        summary += "| Model | Variant | TP | Accuracy | Threshold | Status |\n"
        summary += "| ----- | ------- | -- | -------- | --------- | ------ |\n"

        for config in self.models:
            display_name = config.get_display_name()
            with self.subTest(model=display_name):
                model_path = config.model_path
                is_local = model_path.startswith("/")
                if is_local and not os.path.exists(model_path):
                    self.skipTest(f"Local model not found at {model_path}")

                other_args = list(config.other_args) + ["--tp", str(config.tp_size)]
                timeout = config.timeout or DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

                env = os.environ.copy()
                for key, value in config.env_vars.items():
                    env[key] = value

                try:
                    process = popen_launch_server(
                        model=model_path,
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
                            f"  accuracy={acc:.4f} threshold={config.accuracy_threshold} {status}"
                        )

                        all_results.append(
                            {"model": display_name, "accuracy": acc, "passed": passed}
                        )
                        summary += (
                            f"| {model_path} | {config.variant or 'N/A'} "
                            f"| {config.tp_size} | {acc:.4f} "
                            f"| {config.accuracy_threshold} | {status} |\n"
                        )

                    finally:
                        kill_process_tree(process.pid)

                except Exception as e:
                    summary += (
                        f"| {model_path} | {config.variant or 'N/A'} "
                        f"| {config.tp_size} | N/A "
                        f"| {config.accuracy_threshold} | ERROR |\n"
                    )
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
            raise AssertionError(
                f"Failed model variants: {[r['model'] for r in failed]}"
            )


if __name__ == "__main__":
    unittest.main()
