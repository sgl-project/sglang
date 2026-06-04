"""MI35x DeepSeek-R1-MXFP4 TP=2 GSM8K AITER MLA regression.

DeepSeek-R1 has 128 attention heads, so TP=2 gives 64 heads per rank. This
covers the AITER persistent MLA decode metadata path that previously crashed
with GPU memory access faults.

Registry: nightly-amd-2-gpu-mi35x-deepseek-r1-mxfp4-tp2 suite
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
    est_time=1800,
    suite="nightly-amd-2-gpu-mi35x-deepseek-r1-mxfp4-tp2",
    nightly=True,
)

INVALID = -9999999

SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_ACCURACY_THRESHOLD = 0.93


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


class TestDeepSeekR1MXFP4TP2MI35x(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("DEEPSEEK_R1_MXFP4_MODEL_PATH", "amd/DeepSeek-R1-MXFP4-Preview")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))

        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_AITER_MLA_PERSIST"] = "1"

        cls.process = popen_launch_server(
            model=cls.model,
            base_url=cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--attention-backend",
                "aiter",
                "--tp",
                "2",
                "--chunked-prefill-size",
                "131072",
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.85",
                "--trust-remote-code",
                "--kv-cache-dtype",
                "fp8_e4m3",
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
                "### DeepSeek-R1-MXFP4 TP=2 GSM8K (MI35x)\n\n"
                "| Model | TP | Examples | Accuracy | Invalid | Threshold | Latency |\n"
                "| ----- | -- | -------- | -------- | ------- | --------- | ------- |\n"
                f"| {self.model} | 2 | {self.num_questions} | {acc:.3f} | "
                f"{invalid:.3f} | {GSM8K_ACCURACY_THRESHOLD:.2f} | {latency:.1f}s |\n"
            )

        self.assertGreaterEqual(acc, GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
