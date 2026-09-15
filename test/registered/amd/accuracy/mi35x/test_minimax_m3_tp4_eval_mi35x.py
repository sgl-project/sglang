"""MI35x MiniMax-M3 MXFP8 GSM8K Chat+Thinking Evaluation Test (4-GPU, TP=4)

Tests MiniMax-M3 (MXFP8 checkpoint) with TP=4 on MI35x. MI35x (gfx950 / CDNA4)
has hardware MX-scaled matmul, so the MXFP8 MoE weights are served natively;
you still pass `--quantization mxfp8`. Serves with the aiter attention backend,
fp8 (e4m3) KV cache, and radix cache disabled — validated accuracy-neutral vs
the bf16-KV / triton-attn baseline (0.972 vs 0.970 on GSM8K chat+thinking).

MiniMax-M3 is a reasoning model: it must be evaluated through the chat template
with thinking enabled (its `<mm:think>` reasoning path). Raw sgl-eval chat
(no chat template) does NOT engage its reasoning and severely underscores it
(~0.87 vs ~0.96 on GSM8K), so this test uses chat + thinking to match how the
model is meant to be served and the published reference accuracy.

Registry: nightly-amd-4-gpu-mi35x-minimax-m3-tp4 suite
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

register_amd_ci(
    est_time=5400,
    suite="nightly-amd-4-gpu-mi35x-minimax-m3-tp4",
    nightly=True,
)


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 4
    accuracy_threshold: float = 0.93
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


MI35X_MINIMAX_M3_TP4_MODELS = [
    # MXFP8 + aiter attn + fp8 KV, with the block-fp8 linear path (PR #32036)
    # and custom/quick INT4 all-reduce (PR #32230) opted in. Both are opt-in
    # via env: on gfx950 block convert is not automatic
    # (mxfp8_block_convert_required() is False), and the M3 overrides otherwise
    # force --disable-custom-all-reduce.
    ModelConfig(
        model_path="MiniMaxAI/MiniMax-M3-MXFP8",
        tp_size=4,
        accuracy_threshold=0.95,
        timeout=5400,
        variant="TP4+MXFP8+aiterAttn+fp8KV+blockFP8+quickAR",
        other_args=[
            "--quantization",
            "mxfp8",
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--attention-backend",
            "aiter",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "8192",
            "--mem-fraction-static",
            "0.80",
            "--watchdog-timeout",
            "1200",
        ],
        env_vars={
            "SGLANG_USE_AITER": "1",
            # ROCm 7.0's rocBLAS/hipBLASLt rejects the bf16-input/fp32-output
            # router GEMM (torch.mm(bf16, bf16, out_dtype=float32)); force the
            # fp32 router path. Also gives more precise expert routing.
            "SGLANG_OPT_USE_BF16_ROUTER_GEMM": "0",
            # Block-fp8 linear path (PR #32036): convert MXFP8 linear weights to
            # block-fp8 [128,128] and run them through the tuned block-scale
            # (bpreshuffle) GEMM on gfx950.
            "SGLANG_FORCE_MXFP8_BLOCK_CONVERT": "1",
            # Custom / quick all-reduce (PR #32230): keep custom all-reduce on so
            # the INT4 quick-reduce path is used for the TP all-reduce.
            "SGLANG_M3_ALLOW_CUSTOM_AR": "1",
            "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
            "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "1",
        },
    ),
]


def run_gsm8k_benchmark(
    base_url: str,
    model_path: str,
    num_questions: int = 1319,
    parallel: int = 64,
    max_tokens: int = 4096,
) -> Tuple[float, float, float]:
    """Run the canonical sgl-eval GSM8K benchmark."""
    metrics = run_eval(
        SimpleNamespace(
            eval_name="gsm8k",
            base_url=base_url,
            num_examples=num_questions,
            num_threads=parallel,
            max_tokens=max_tokens,
            model=model_path,
            chat_template_kwargs={"thinking_mode": "enabled"},
        )
    )
    return metrics["score"], metrics["invalid"], metrics["latency"]


class TestMiniMaxM3TP4EvalMI35x(unittest.TestCase):
    """MiniMax-M3 MXFP8 TP=4 GSM8K Chat+Thinking Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.models = MI35X_MINIMAX_M3_TP4_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

    def test_minimax_m3_tp4_accuracy(self):
        """Test MiniMax-M3 MXFP8 TP=4 with GSM8K chat+thinking benchmark."""
        all_results = []
        summary = "### MiniMax-M3 MXFP8 TP=4 chat+thinking (MI35x)\n\n"
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
                            self.base_url,
                            config.model_path,
                            num_questions=self.num_questions,
                        )
                        passed = acc >= config.accuracy_threshold
                        status = "PASS" if passed else "FAIL"
                        print(
                            f"  accuracy={acc:.3f} threshold={config.accuracy_threshold} {status}"
                        )
                        print(f"  invalid={invalid:.3f} latency={latency:.1f}s")

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
