"""MI35x Kimi-K2.5-MXFP4 aiter MLA backend accuracy tests (4-GPU)

Tests Kimi-K2.5-MXFP4 with the aiter unified attention backend on MI35x,
covering both default and FP8 KV cache configurations.

The FP8 KV cache variant validates the fix for assertion failure
`q_scale.has_value() && kv_scale.has_value()` in aiter ASM MLA decode
when layer.k_scale is None (the RadixAttention default).

NOTE: TP must be <= 4 for Kimi-K2.5 with the aiter MLA kernel.
Kimi-K2.5 has num_attention_heads=64; with tp_size=8 that gives
64/8 = 8 heads per GPU, but the aiter ASM MLA kernel requires
heads_per_gpu % 16 == 0. With tp_size=4: 64/4 = 16 heads, which
satisfies the constraint. (DeepSeek-R1/V3 has 128 heads so TP=8
yields 128/8 = 16 heads and works fine.)

Registry: nightly-amd-8-gpu-mi35x-kimi-k25-mxfp4-aiter-mla suite
"""

import os

os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

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
    suite="nightly-amd-8-gpu-mi35x-kimi-k25-mxfp4-aiter-mla",
    nightly=True,
)

KIMI_K25_MXFP4_LOCAL_PATH = "/data/models/amd/Kimi-K2.5-MXFP4"
KIMI_K25_MXFP4_HF_MODEL_ID = "moonshotai/Kimi-K2.5-MXFP4"


def get_model_path() -> str:
    """Get effective model path: env var > local path > HF model ID."""
    env_path = os.environ.get("KIMI_K25_MXFP4_MODEL_PATH")
    if env_path:
        return env_path
    if os.path.exists(KIMI_K25_MXFP4_LOCAL_PATH):
        return KIMI_K25_MXFP4_LOCAL_PATH
    return KIMI_K25_MXFP4_HF_MODEL_ID


@dataclass
class ModelConfig:
    """Configuration for a model variant to test."""

    model_path: str
    tp_size: int = 4
    accuracy_threshold: float = 0.92
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


def get_kimi_k25_mxfp4_models() -> List[ModelConfig]:
    """Get Kimi-K2.5-MXFP4 model configurations for MI35x."""
    model_path = get_model_path()
    common_kwargs = {
        "model_path": model_path,
        # TP=4 required: Kimi-K2.5 has 64 attn heads; aiter ASM MLA needs
        # heads_per_gpu % 16 == 0 -> 64/4=16 works, 64/8=8 does not.
        "tp_size": 4,
        "accuracy_threshold": 0.92,
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
        "--max-running-requests",
        "64",
        "--trust-remote-code",
        "--watchdog-timeout",
        "1200",
    ]
    common_env = {"SGLANG_AITER_MLA_PERSIST": "1"}

    return [
        ModelConfig(
            **common_kwargs,
            variant="default",
            other_args=common_args,
            env_vars=common_env,
        ),
        # FP8 KV cache — validates the k_scale None fallback fix in
        # aiter ASM MLA decode (all 4 mla_decode_fwd call sites).
        ModelConfig(
            **common_kwargs,
            variant="fp8kv",
            other_args=common_args
            + [
                "--kv-cache-dtype",
                "fp8_e4m3",
            ],
            env_vars=common_env,
        ),
    ]


class TestKimiK25MXFP4AiterMlaEvalMI35x(unittest.TestCase):
    """Kimi-K2.5-MXFP4 aiter MLA backend accuracy tests on MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.models = get_kimi_k25_mxfp4_models()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

    def test_kimi_k25_mxfp4_accuracy(self):
        """Test Kimi-K2.5-MXFP4 with GSM8K completion benchmark (default & fp8kv)."""
        model_path = get_model_path()
        is_local_path = model_path.startswith("/")
        if is_local_path and not os.path.exists(model_path):
            print(f"\nSKIPPING: Local model not found at {model_path}")
            self.skipTest(f"Local model not found at {model_path}")
            return

        if is_local_path:
            print(f"Using local model: {model_path}")
        else:
            print(f"Using HuggingFace model: {model_path}")

        from types import SimpleNamespace

        from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k

        all_results = []
        summary = "### Kimi-K2.5-MXFP4 aiter MLA (MI35x)\n\n"
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
