"""MI35x Qwen3.5-397B-A17B-MXFP4-AttnFP8 fused AR+RMSNorm+per-token-quant GSM8K gate.

Exercises the AMD per-token fused all-reduce path (PR #29723): with
`--enable-aiter-allreduce-fusion` the TP all-reduce, the following RMSNorm, and
the per-token activation quant feeding the next GEMM are fused into one aiter
kernel. The MXFP4-AttnFP8 checkpoint drives both fused-quant formats in one run:
`mxfp4` (uint8 MoE/GDN weights) and `fp8_per_token` (FP8 attention/GDN input
projections, needs `SGLANG_USE_AITER_FP8_PER_TOKEN=1`).

The feature is default-off and gated behind `--enable-aiter-allreduce-fusion`
plus `SGLANG_USE_AITER=1` and `SGLANG_USE_AITER_FP8_PER_TOKEN=1`, so a plain
green CI run never executes it. This test sets those and asserts GSM8K accuracy.
The fused epilogue is numerically equivalent to the unfused path (~0.93-0.94 on
this harness), so we gate at 0.92.

Registry: nightly-amd-accuracy-8-gpu-mi35x-qwen35 suite.
"""

import ast
import os

import re
import time
import unittest
from typing import Tuple

import numpy as np

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)
from sglang.utils import download_and_cache_file, read_jsonl

register_amd_ci(
    est_time=3600,
    suite="nightly-amd-accuracy-8-gpu-mi35x-qwen35",
    nightly=True,
)

INVALID = -9999999

QWEN35_ATTNFP8_HF_MODEL_ID = "amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8"
SERVER_LAUNCH_TIMEOUT = 3600
# TP=4: TP=8 shards shared_expert.down_proj (K=1024) to K=128, which aiter's fp8
# GEMM has no tuned config for and crashes cuda-graph capture. Override for local runs.
TP_SIZE = int(os.environ.get("QWEN35_TP_SIZE", "4"))
# Large cap so the reasoning <think> block reaches the answer (see PR #29264).
GSM8K_MAX_NEW_TOKENS = int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "8192"))
GSM8K_ACCURACY_THRESHOLD = 0.92


def get_model_path() -> str:
    # CI resolves the checkpoint from the HF id; local runs can point at a
    # pre-downloaded checkpoint by exporting QWEN35_ATTNFP8_MODEL_PATH.
    return os.environ.get("QWEN35_ATTNFP8_MODEL_PATH", QWEN35_ATTNFP8_HF_MODEL_ID)


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
    num_questions: int = 1319,
    num_shots: int = 5,
    parallel: int = 128,
) -> Tuple[float, float, float]:
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
            "answer",
            max_tokens=GSM8K_MAX_NEW_TOKENS,
            stop=["\n\nQuestion", "Assistant:", "<|im_end|>"],
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


class TestQwen35MXFP4AttnFp8ARFusionMI35x(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

        # Turn the fused path on; leave the SGLANG_DISABLE_FUSED_AR_* opt-outs unset.
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_USE_AITER_FP8_PER_TOKEN"] = "1"

        cls.process = popen_launch_server(
            model=cls.model,
            base_url=cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--attention-backend",
                "aiter",
                "--tp",
                str(TP_SIZE),
                "--enable-aiter-allreduce-fusion",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.85",
                "--trust-remote-code",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        acc, invalid, latency = run_gsm8k_benchmark(
            self.base_url, num_questions=self.num_questions
        )
        print(f"accuracy={acc:.3f} invalid={invalid:.3f} latency={latency:.1f}s")

        if is_in_ci():
            write_github_step_summary(
                "### Qwen3.5-397B-A17B-MXFP4-AttnFP8 fused AR+RMSNorm+per-token-quant GSM8K (MI35x)\n\n"
                "| Model | TP | Examples | Accuracy | Invalid | Threshold | Latency |\n"
                "| ----- | -- | -------- | -------- | ------- | --------- | ------- |\n"
                f"| {self.model} | {TP_SIZE} | {self.num_questions} | {acc:.3f} | "
                f"{invalid:.3f} | {GSM8K_ACCURACY_THRESHOLD:.2f} | {latency:.1f}s |\n"
            )

        self.assertGreaterEqual(acc, GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
