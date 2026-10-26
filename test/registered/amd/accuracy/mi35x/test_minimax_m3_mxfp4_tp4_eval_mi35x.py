"""MI35x MiniMax-M3 MXFP4 GSM8K Chat+Thinking Evaluation Test (4-GPU, TP=4)

Tests the AMD MXFP4 checkpoint of MiniMax-M3 (`amd/MiniMax-M3-MXFP4`, Quark
w4a4 with per-group-32 scales) with TP=4 on MI35x (gfx950 / CDNA4). SGLang
auto-detects the checkpoint as `quantization=quark`, so no `--quantization`
flag is needed; the MoE runs on aiter's native MXFP4 FlyDSL kernels
(`flydsl_moe1_afp4_wfp4_*`, QuantType.per_1x32).

Measured on MI355X against the MXFP8 checkpoint on the same box: GSM8K 0.925 vs
0.928 (within noise) with 45% smaller weights (58.1 vs 106.2 GB per GPU), which
buys a ~60% larger KV cache. Throughput is 5-11% lower than MXFP8.

Gluon paged sparse prefill must be disabled on this path — see the env note
below.

MiniMax-M3 is a reasoning model: it must be evaluated through the chat template
with thinking enabled (its `<mm:think>` reasoning path). Raw few-shot completion
(no chat template) does NOT engage its reasoning and severely underscores it,
so this test uses chat + thinking to match how the model is meant to be served
and the published reference accuracy.

Registry: nightly-amd-4-gpu-mi35x-minimax-m3-mxfp4-tp4 suite
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
    est_time=5400,
    suite="nightly-amd-4-gpu-mi35x-minimax-m3-mxfp4-tp4",
    nightly=True,
)

INVALID = -9999999


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


MI35X_MINIMAX_M3_MXFP4_TP4_MODELS = [
    ModelConfig(
        model_path="amd/MiniMax-M3-MXFP4",
        tp_size=4,
        accuracy_threshold=0.90,
        timeout=5400,
        variant="TP4+MXFP4+aiterAttn+aiterMoE+quickAR",
        other_args=[
            # No --quantization: the Quark MXFP4 checkpoint is auto-detected
            # as quant_method=quark.
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--attention-backend",
            "aiter",
            "--moe-runner-backend",
            "aiter",
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "8192",
            "--mem-fraction-static",
            "0.79",
            "--watchdog-timeout",
            "1200",
        ],
        env_vars={
            "SGLANG_USE_AITER": "1",
            # Gluon paged sparse prefill faults the GPU on long prompts with
            # this checkpoint. Chunking a long prompt into 8192-token prefills
            # aborts all ranks with:
            #   Memory access fault by GPU node-N ... Reason: Unknown.
            # from _build_gluon_prefill_meta (gluon_prefill.py). The MXFP8
            # checkpoint on the same build is unaffected, so this looks
            # specific to the quark/MXFP4 path. Fall back to Triton sparse
            # prefill.
            "SGLANG_OPT_USE_MINIMAX_GLUON_PREFILL": "0",
            # Keep custom all-reduce on so the INT4 quick-reduce path is used
            # for the TP all-reduce (the M3 overrides otherwise force
            # --disable-custom-all-reduce). Note the ROCm images already
            # default ROCM_QUICK_REDUCE_QUANTIZATION=INT8, so this selects
            # INT4 rather than enabling quick-reduce.
            "SGLANG_M3_ALLOW_CUSTOM_AR": "1",
            "ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
            "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16": "1",
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
        with urllib.request.urlopen(req, timeout=1800) as resp:
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


class TestMiniMaxM3MXFP4TP4EvalMI35x(unittest.TestCase):
    """MiniMax-M3 MXFP4 TP=4 GSM8K Chat+Thinking Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.models = MI35X_MINIMAX_M3_MXFP4_TP4_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

    def test_minimax_m3_mxfp4_tp4_accuracy(self):
        """Test MiniMax-M3 MXFP4 TP=4 with GSM8K chat+thinking benchmark."""
        all_results = []
        summary = "### MiniMax-M3 MXFP4 TP=4 chat+thinking (MI35x)\n\n"
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
