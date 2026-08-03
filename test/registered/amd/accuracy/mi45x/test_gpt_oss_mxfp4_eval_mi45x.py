"""MI455X GPT-OSS W4A8 MXFP4-FP8 GSM8K Completion Evaluation Test (1-GPU).

Bring-up gate for MI455X (gfx1250): runs the AMD Quark
`gpt-oss-120b-w-mxfp4-a-fp8` checkpoint (MXFP4 weights + static per-tensor FP8
activations) as a few-shot GSM8K completion benchmark on a single GPU.

The same checkpoint is already covered at TP=8 on MI35x by
`test/registered/amd/accuracy/mi35x/test_gpt_oss_w4a8_mxfp4_eval_mi35x.py`;
this test is deliberately the TP=1 single-GPU variant so MI455X can be gated
on one card.

Registry: nightly-amd-1-gpu-mi45x suite
"""

import ast
import os
import re
import time
import unittest
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

# Register for AMD CI - MI455X GPT-OSS MXFP4 accuracy test (~20 min)
register_amd_ci(est_time=1200, suite="nightly-amd-1-gpu-mi45x", nightly=True)

INVALID = -9999999


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 1
    accuracy_threshold: float = 0.79
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    timeout: Optional[int] = None

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}


MI45X_GPT_OSS_MXFP4_MODELS = [
    ModelConfig(
        model_path="amd/gpt-oss-120b-w-mxfp4-a-fp8",
        tp_size=1,
        # Mirrors the TP=8 MI35x peer's floor (0.79). MI455X has no measured
        # baseline yet — confirm against the first green nightly and tighten or
        # loosen here rather than letting the gate pass vacuously.
        # Override for local runs with GSM8K_ACCURACY_THRESHOLD.
        accuracy_threshold=float(os.environ.get("GSM8K_ACCURACY_THRESHOLD", "0.79")),
        timeout=1800,
        other_args=[
            "--trust-remote-code",
            # gfx1250 runs prefill on Triton and decode on the AITER unified
            # attention kernel; neither backend covers both phases well yet.
            "--prefill-attention-backend",
            "triton",
            "--decode-attention-backend",
            "aiter",
            "--max-running-requests",
            "128",
            "--mem-fraction-static",
            "0.9",
            "--disable-radix-cache",
            "--page-size",
            "64",
        ],
        env_vars={
            "SGLANG_USE_AITER": "1",
            # Selects GateMode.INTERLEAVE (this is also the env default).
            #
            # UNVERIFIED ON gfx1250 — check this first if accuracy is bad.
            # The matching weight interleave in QuarkW4A8MXFp4MoE is gated on
            # is_gfx95_supported(), which matches only "gfx95" and so is False
            # on gfx1250. The kernel is therefore told INTERLEAVE while the
            # weights are left in the checkpoint's layout. If GSM8K comes back
            # near-random on real hardware, try "0" (SEPARATED) before touching
            # anything else, and extend is_gfx95_supported() if the shuffle is
            # what's actually needed.
            "SGLANG_USE_AITER_MOE_GU_ITLV": "1",
            "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
            # Route MXFP4 weights through the A8W4 GEMM path.
            "AITER_FORCE_A8W4": "1",
            # gfx1250 has no CK kernels yet.
            "ENABLE_CK": "0",
            # Coredumps on this image are multi-GB and fill the runner disk.
            "HSA_COREDUMP_PATTERN": "/dev/null",
        },
    ),
]


def get_one_example(lines, i, include_answer):
    """Format a single GSM8K example."""
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    """Get k few-shot examples for prompting."""
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    """Extract numerical answer from response."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def run_gsm8k_benchmark(
    base_url: str,
    num_questions: int = 200,
    num_shots: int = 5,
    parallel: int = 64,
) -> Tuple[float, float, float]:
    """Run GSM8K few-shot completion benchmark."""
    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    data_path = download_and_cache_file(url)
    lines = list(read_jsonl(data_path))

    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer", max_tokens=512, stop=["Question", "Assistant:", "<|separator|>"]
        )

    backend = RuntimeEndpoint(base_url)
    sgl.set_default_backend(backend)

    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments, temperature=0, num_threads=parallel, progress_bar=True
    )
    latency = time.perf_counter() - tic

    preds = [get_answer_value(states[i]["answer"]) for i in range(len(states))]
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    return float(acc), float(invalid), float(latency)


class TestGptOssMxfp4EvalMI45x(unittest.TestCase):
    """GPT-OSS MXFP4 GSM8K Completion Evaluation Test for AMD MI455X (1-GPU)."""

    @classmethod
    def setUpClass(cls):
        cls.models = MI45X_GPT_OSS_MXFP4_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_gpt_oss_accuracy(self):
        """Test GPT-OSS MXFP4 models with GSM8K completion benchmark."""
        all_results = []
        summary = "### GPT-OSS MXFP4 Models (MI455X, 1-GPU)\n\n"
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
                                "model": config.model_path,
                                "accuracy": acc,
                                "passed": passed,
                            }
                        )
                        summary += f"| {config.model_path} | {config.tp_size} | {acc:.3f} | {config.accuracy_threshold} | {status} |\n"

                    finally:
                        kill_process_tree(process.pid)

                except Exception as e:
                    summary += f"| {config.model_path} | {config.tp_size} | N/A | {config.accuracy_threshold} | ❌ ERROR |\n"
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
