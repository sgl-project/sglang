"""MI35x DeepSeek-R1-MXFP4 TP=4 EAGLE GSM8K regression.

This mirrors the production-style TP=4 launch recipe with EAGLE speculative
decoding, overlap plan stream, FP8 KV cache, long context, and full GSM8K
client pressure enabled.

Registry: nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4-tp4 suite
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
    suite="nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4-tp4",
    nightly=True,
)

INVALID = -9999999

SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_MTP_ACCURACY_THRESHOLD = 0.944


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
    num_questions: int = 200,
    num_shots: int = 5,
    parallel: int = 64,
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


class TestDeepSeekR1MXFP4TP4MTPMI35x(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get(
            "DEEPSEEK_R1_MXFP4_MODEL_PATH", "amd/DeepSeek-R1-MXFP4-Preview"
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
        cls.parallel = int(os.environ.get("GSM8K_PARALLEL", "1319"))

        env = os.environ.copy()
        env.update(
            {
                "SGLANG_USE_AITER": "1",
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                # Retired on current main, but kept to mirror the reported launch.
                "ROCM_QUICK_REDUCE_QUANTIZATION": "NONE",
                "SGLANG_AITER_FP8_PREFILL_ATTN": "1",
                "SGLANG_AITER_MLA_PERSIST": "1",
                "AITER_MXFP4_MOE_SF": "1",
                "SGLANG_INT4_WEIGHT": "0",
                "SGLANG_MOE_PADDING": "1",
                "SGLANG_SET_CPU_AFFINITY": "1",
                "SGLANG_ROCM_FUSED_DECODE_MLA": "1",
                "SGLANG_USE_ROCM700A": "1",
            }
        )

        cls.process = popen_launch_server(
            model=cls.model,
            base_url=cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--tensor-parallel-size",
                "4",
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.9",
                "--chunked-prefill-size",
                "131072",
                "--attention-backend",
                "aiter",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--max-running-requests",
                "32",
                "--context-length",
                "200000",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true, "num_threads": 8}',
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        acc, invalid, latency = run_gsm8k_benchmark(
            self.base_url,
            num_questions=self.num_questions,
            num_shots=5,
            parallel=self.parallel,
        )
        print(f"accuracy={acc:.3f} invalid={invalid:.3f} latency={latency:.1f}s")

        if is_in_ci():
            write_github_step_summary(
                "### DeepSeek-R1-MXFP4 TP=4 MTP GSM8K (MI35x)\n\n"
                "| Model | TP | Examples | Parallel | Accuracy | Invalid | Threshold | Latency |\n"
                "| ----- | -- | -------- | -------- | -------- | ------- | --------- | ------- |\n"
                f"| {self.model} | 4 | {self.num_questions} | {self.parallel} | "
                f"{acc:.3f} | {invalid:.3f} | {GSM8K_MTP_ACCURACY_THRESHOLD:.3f} | "
                f"{latency:.1f}s |\n"
            )

        self.assertGreaterEqual(acc, GSM8K_MTP_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
