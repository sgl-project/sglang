"""MI30x MiniMax-M3 MXFP8 GSM8K Chat+Thinking Evaluation Test (8-GPU, TP=8)

Tests MiniMax-M3 (MXFP8 checkpoint) with TP=8 on MI30x. gfx942 (CDNA3) has no
hardware MX-scaled matmul, so `mxfp8_block_convert_required()` is True there and
SGLang dequantizes the MXFP8 weights and requantizes them to block-fp8 [128,128]
at load, then serves them on the tuned ROCm block-fp8 kernels. That conversion —
and the e4m3fnuz renormalization it feeds on CDNA3 — has no other end-to-end
coverage: the MI35x M3 test reaches the block-fp8 kernels through
SGLANG_FORCE_MXFP8_BLOCK_CONVERT, but on gfx950 the conversion is opt-in and
fp8 stays e4m3fn.

Serving flags are the cookbook's verified MI300X recipe: aiter attention with the
Triton MoE runner, bf16 KV, and mem-fraction 0.80. `--watchdog-timeout 3600` plus
`--skip-server-warmup` ride out the cold-start AITER JIT, which on a first
generation can compile past the default warmup window. MiniMax's MSA sparse
kernel is SM100-only, so the sparse attention step runs on the built-in Triton
path here, as it does on every ROCm target.

MiniMax-M3 is a reasoning model: it must be evaluated through the chat template
with thinking enabled (its `<mm:think>` reasoning path), the same way the MI35x
test does. Raw few-shot completion does not engage its reasoning and severely
underscores it.

Registry: nightly-amd-accuracy-8-gpu-minimax-m3 suite
"""

import json
import os
import re
import time
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

register_amd_ci(
    est_time=9000,
    suite="nightly-amd-accuracy-8-gpu-minimax-m3",
    nightly=True,
)

INVALID = -9999999


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 8
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


MI30X_MINIMAX_M3_MODELS = [
    # The threshold is the floor the block-fp8 path is documented to clear, not a
    # tight bound: the cookbook's MI300X card reports GSM8K 0.917-0.929 on the
    # legacy few-shot harness (aiter ~0.929), and chat+thinking scores well above
    # its own few-shot number (0.972 vs ~0.87 on MI35x). Tighten once a nightly
    # run establishes what this configuration actually returns.
    ModelConfig(
        model_path="MiniMaxAI/MiniMax-M3-MXFP8",
        tp_size=8,
        accuracy_threshold=0.93,
        timeout=7200,
        variant="TP8+MXFP8->blockFP8+aiterAttn+tritonMoE",
        other_args=[
            "--quantization",
            "mxfp8",
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--attention-backend",
            "aiter",
            "--moe-runner-backend",
            "triton",
            "--chunked-prefill-size",
            "8192",
            # M3 is multimodal, so adjust_mem_fraction_for_vlm scales this down
            # for the vision tower; 0.80 lands at ~0.68 effective.
            "--mem-fraction-static",
            "0.80",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            # Cold-start AITER JIT can outrun the default warmup window; the
            # cookbook recipe pairs these two for exactly that reason.
            "--watchdog-timeout",
            "3600",
            "--skip-server-warmup",
        ],
        env_vars={
            "SGLANG_USE_AITER": "1",
            # ROCm's rocBLAS/hipBLASLt rejects the bf16-input/fp32-output router
            # GEMM (torch.mm(bf16, bf16, out_dtype=float32)); force the fp32
            # router path. Also gives more precise expert routing.
            "SGLANG_OPT_USE_BF16_ROUTER_GEMM": "0",
        },
    ),
]


def get_answer_value(answer_str):
    """Extract numerical answer from response (last integer)."""
    if not isinstance(answer_str, str):
        return INVALID
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+", answer_str)
    if not numbers:
        return INVALID
    try:
        return int(numbers[-1])
    except ValueError:
        return INVALID


def run_gsm8k_benchmark(
    base_url: str,
    model_path: str,
    num_questions: int = 1319,
    parallel: int = 64,
    max_tokens: int = 4096,
) -> Tuple[float, float, float]:
    """Run GSM8K in chat + thinking mode (M3's intended reasoning path).

    Uses the OpenAI-compatible /v1/chat/completions endpoint so the chat
    template is applied, and forces thinking via chat_template_kwargs
    (M3's template reads ``thinking_mode``). The final answer is the last
    integer in the response; the reasoning trace lives in ``<mm:think>`` tags
    inside ``content`` (or in ``reasoning_content`` if a reasoning parser is on),
    so both channels are scanned.
    """
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    instruction = (
        "\n\nPlease reason step by step, and give the final answer as a single "
        "integer on the last line."
    )
    n = len(lines[:num_questions])
    questions = [lines[i]["question"] for i in range(n)]
    labels = [get_answer_value(lines[i]["answer"]) for i in range(n)]
    assert all(l != INVALID for l in labels)

    def query(question: str) -> str:
        body = {
            "model": model_path,
            "messages": [{"role": "user", "content": question + instruction}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking_mode": "enabled"},
        }
        req = urllib.request.Request(
            base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        # The server skips warmup, so the first batch pays for the AITER JIT.
        with urllib.request.urlopen(req, timeout=3600) as resp:
            msg = json.loads(resp.read())["choices"][0]["message"]
        return (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")

    tic = time.perf_counter()
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        outputs = list(ex.map(query, questions))
    latency = time.perf_counter() - tic

    preds = [get_answer_value(o) for o in outputs]
    acc = float(np.mean(np.array(preds) == np.array(labels)))
    invalid = float(np.mean(np.array(preds) == INVALID))

    return acc, invalid, latency


class TestMiniMaxM3EvalAMD(unittest.TestCase):
    """MiniMax-M3 MXFP8 TP=8 GSM8K Chat+Thinking Evaluation Test for AMD MI30x."""

    @classmethod
    def setUpClass(cls):
        cls.models = MI30X_MINIMAX_M3_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

    def test_minimax_m3_accuracy(self):
        """Test MiniMax-M3 MXFP8 TP=8 with GSM8K chat+thinking benchmark."""
        all_results = []
        summary = "### MiniMax-M3 MXFP8 TP=8 chat+thinking (MI30x)\n\n"
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
