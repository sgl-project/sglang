"""MI35x PR-CI accuracy coverage for Qwen3.5-FP8 aiter AR-fusion.

Sequentially launches one TP8 server per fusion variant on the 8-GPU MI35x
stage-c suite and runs GSM8K in-process (mirroring the DeepSeek-R1-MXFP4
AR-fusion eval) to compare accuracy:

* fused AR+RMSNorm+per-group FP8 quant enabled (default), and
* the same launch with SGLANG_DISABLE_FUSED_AR_QUANT=1 fallback.

Running the variants one at a time (instead of two parallel TP4 servers) keeps
the signal stable and avoids loading two 397B servers at once. The nightly
throughput/latency perf benchmark lives separately in
test_qwen35_fp8_perf_mi35x.py.
"""

import ast
import os
import re
import time
import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

register_amd_ci(est_time=4800, suite="stage-c-test-large-8-gpu-amd-mi35x")

QWEN35_FP8_MODEL_PATH = os.environ.get(
    "QWEN35_FP8_MODEL_PATH",
    "Qwen/Qwen3.5-397B-A17B-FP8",
)
SERVER_LAUNCH_TIMEOUT = 4800
GSM8K_NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
ACCURACY_THRESHOLD = 0.94
INVALID = -9999999
GSM8K_DATA_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)

COMMON_ARGS: List[str] = [
    "--tensor-parallel-size",
    "8",
    "--trust-remote-code",
    "--attention-backend",
    "aiter",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--page-size",
    "16",
    "--chunked-prefill-size",
    "8192",
    "--mem-fraction-static",
    "0.8",
    "--disable-radix-cache",
    "--enable-aiter-allreduce-fusion",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--watchdog-timeout",
    "1200",
]


@dataclass
class FusionVariant:
    """A Qwen3.5-FP8 AR-fusion configuration to validate."""

    variant: str
    env_vars: Dict[str, str] = field(default_factory=dict)


def get_fusion_variants() -> List[FusionVariant]:
    return [
        FusionVariant(
            variant="fused-ar-rms-per-group-quant",
            env_vars={
                "SGLANG_USE_AITER": "1",
                "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
            },
        ),
        FusionVariant(
            variant="disable-fused-ar-quant-opt-out",
            env_vars={
                "SGLANG_USE_AITER": "1",
                "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
                "SGLANG_DISABLE_FUSED_AR_QUANT": "1",
            },
        ),
    ]


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
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
    num_questions: int,
    num_shots: int = 5,
    parallel: int = 64,
) -> Tuple[float, float, float]:
    """Run GSM8K few-shot completion benchmark in-process against base_url."""
    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    data_path = download_and_cache_file(GSM8K_DATA_URL)
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


class TestQwen35Fp8ArFusionMI35x(CustomTestCase):
    """Validate Qwen3.5-FP8 AR-fusion accuracy on MI35x (sequential TP8)."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.variants = get_fusion_variants()
        cls.num_questions = GSM8K_NUM_QUESTIONS

    def test_qwen35_fp8_ar_fusion_accuracy(self):
        summary = "### Qwen3.5-FP8 aiter AR-fusion (MI35x, sequential TP8)\n\n"
        summary += (
            "| Variant | Accuracy | Invalid | Latency (s) | Threshold | Status |\n"
        )
        summary += (
            "| ------- | -------- | ------- | ----------- | --------- | ------ |\n"
        )

        failures = []
        for variant in self.variants:
            with self.subTest(variant=variant.variant):
                env = os.environ.copy()
                env.update(variant.env_vars)

                process = popen_launch_server(
                    self.model,
                    self.base_url,
                    timeout=SERVER_LAUNCH_TIMEOUT,
                    other_args=list(COMMON_ARGS),
                    env=env,
                )
                try:
                    acc, invalid, latency = run_gsm8k_benchmark(
                        self.base_url,
                        num_questions=self.num_questions,
                        parallel=self.num_questions,
                    )
                finally:
                    kill_process_tree(process.pid)

                passed = acc >= ACCURACY_THRESHOLD
                status = "PASS" if passed else "FAIL"
                summary += (
                    f"| {variant.variant} | {acc:.3f} | {invalid:.3f} | "
                    f"{latency:.2f} | {ACCURACY_THRESHOLD} | {status} |\n"
                )
                print(
                    f"[{variant.variant}] accuracy={acc:.3f} invalid={invalid:.3f} "
                    f"latency={latency:.2f}s {status}"
                )
                if not passed:
                    failures.append((variant.variant, acc))

        if is_in_ci():
            write_github_step_summary(summary)
        print(summary)

        self.assertEqual(
            failures,
            [],
            f"Qwen3.5-FP8 AR-fusion accuracy below {ACCURACY_THRESHOLD}: {failures}",
        )


if __name__ == "__main__":
    unittest.main()
