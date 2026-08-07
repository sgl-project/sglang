"""MI35x Qwen3.5-397B-A17B-MXFP4 `--enable-dense-fp8` GSM8K accuracy gate.

Exercises the AMD dense-FP8 path (PR #28932): with `--enable-dense-fp8` the
Quark MXFP4 checkpoint's otherwise-bf16 `shared_expert.down_proj` is promoted to
online w8a8 FP8 (per-token), and the preceding `SiluAndMul` + activation quant is
fused into a single `aiter.silu_and_mul_quant` kernel that feeds the `(fp8, scale)`
tuple straight into `down_proj`.

The feature is default-off and gated behind `--enable-dense-fp8` plus
`SGLANG_USE_AITER=1` and `SGLANG_USE_AITER_FP8_PER_TOKEN=1`, so a plain green CI
run never executes it. This test launches the checkpoint with the flag + env vars
set and asserts GSM8K accuracy, giving CI a path that actually runs the code.

Model: https://huggingface.co/amd/Qwen3.5-397B-A17B-MXFP4 (GSM8K flexible-extract
94.54 per the model card). Promoting `down_proj` to FP8 is accuracy-neutral, so we
gate at a comfortable 0.92.

Registry: nightly-amd-accuracy-8-gpu-mi35x-qwen35 suite (shared with the bf16
Qwen3.5 lm-eval gate so the Qwen3.5 accuracy tests stay together).
"""

import ast
import os

# Set HF cache for MI35x.
os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

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

QWEN35_MXFP4_LOCAL_PATH = "/data2/models/amd-Qwen3.5-397B-A17B-MXFP4"
QWEN35_MXFP4_HF_MODEL_ID = "amd/Qwen3.5-397B-A17B-MXFP4"
SERVER_LAUNCH_TIMEOUT = 3600
# TP=4 matches the AMD model-card reproduction recipe and keeps model-load + full
# GSM8K inside the 3600s per-file budget on this 8-GPU MI35x runner. Avoid TP=8 here:
# shared_expert.down_proj is row-parallel with a 1024 contraction dim, so TP=8 shards
# it to K=1024/8=128. aiter's fp8 bpreshuffle GEMM has no tuned config for that shape
# at decode-graph batch sizes and its untuned CK fallback rejects K<=192, so cuda-graph
# capture crashes -- an aiter FP8-GEMM dispatch/tuning gap, not a dense-FP8 issue.
# Overridable via QWEN35_TP_SIZE for local runs.
TP_SIZE = int(os.environ.get("QWEN35_TP_SIZE", "4"))
# Qwen3.5 is a reasoning model: it emits a long <think> block before the final
# answer, so the generation cap must be large enough to reach the "#### <answer>"
# line (see PR #29264, which reproduces at 16384). A small cap truncates valid
# answers mid-reasoning and tanks accuracy.
GSM8K_MAX_NEW_TOKENS = int(os.environ.get("GSM8K_MAX_NEW_TOKENS", "8192"))
# Promoting shared_expert.down_proj to FP8 is accuracy-neutral, and Qwen3.5-MXFP4
# scores ~0.94-0.97 on this harness, so 0.92 gates real regressions with margin.
GSM8K_ACCURACY_THRESHOLD = 0.92


def get_model_path() -> str:
    env_path = os.environ.get("QWEN35_MXFP4_MODEL_PATH")
    if env_path:
        return env_path
    if os.path.exists(QWEN35_MXFP4_LOCAL_PATH):
        return QWEN35_MXFP4_LOCAL_PATH
    return QWEN35_MXFP4_HF_MODEL_ID


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
        # Stop only at the next few-shot boundary ("\n\nQuestion"), not the bare
        # word "Question", which appears inside reasoning text and would truncate
        # valid answers early (fix from PR #29264).
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


class TestQwen35MXFP4DenseFp8MI35x(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_model_path()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

        # --enable-dense-fp8 is AITER-only and requires the per-token FP8 path.
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
                "--enable-dense-fp8",
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
                "### Qwen3.5-397B-A17B-MXFP4 --enable-dense-fp8 GSM8K (MI35x)\n\n"
                "| Model | TP | Examples | Accuracy | Invalid | Threshold | Latency |\n"
                "| ----- | -- | -------- | -------- | ------- | --------- | ------- |\n"
                f"| {self.model} | {TP_SIZE} | {self.num_questions} | {acc:.3f} | "
                f"{invalid:.3f} | {GSM8K_ACCURACY_THRESHOLD:.2f} | {latency:.1f}s |\n"
            )

        self.assertGreaterEqual(acc, GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
