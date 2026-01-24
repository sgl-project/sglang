"""MI35x GROK1-INT4 GSM8K Completion Evaluation Test (8-GPU)

Tests Grok-1 INT4 (W4A8KV8) model using few-shot completion benchmark on MI35x.

Registry: nightly-amd-accuracy-8-gpu-mi35x-grok1-int4 suite
"""

import ast
import os
import re
import time
import unittest

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

# Register for AMD CI - GROK1-INT4 accuracy tests on MI35x (~25 min)
register_amd_ci(
    est_time=1500, suite="nightly-amd-accuracy-8-gpu-mi35x-grok1-int4", nightly=True
)

INVALID = -9999999

GROK1_INT4_MODEL_PATH = os.environ.get("GROK1_INT4_MODEL_PATH", "amd/grok-1-W4A8KV8")
GROK1_TOKENIZER_PATH = os.environ.get("GROK1_TOKENIZER_PATH", "Xenova/grok-1-tokenizer")


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


def run_gsm8k_benchmark(base_url, num_questions=200, num_shots=5, parallel=64):
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
    return float(acc), float(latency)


class TestGrok1INT4EvalMI35x(unittest.TestCase):
    """GROK1-INT4 GSM8K Completion Evaluation Test for MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))
        cls.accuracy_threshold = 0.80

    def test_grok1_int4_accuracy(self):
        """Test Grok-1 INT4 with GSM8K completion benchmark."""
        env = os.environ.copy()
        env["RCCL_MSCCL_ENABLE"] = "0"
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_INT4_WEIGHT"] = "1"

        other_args = [
            "--tp",
            "8",
            "--quantization",
            "fp8",
            "--attention-backend",
            "aiter",
            "--mem-fraction-static",
            "0.85",
            "--tokenizer-path",
            GROK1_TOKENIZER_PATH,
            "--trust-remote-code",
        ]

        process = popen_launch_server(
            model=GROK1_INT4_MODEL_PATH,
            base_url=self.base_url,
            timeout=3600,
            other_args=other_args,
            env=env,
        )

        try:
            acc, latency = run_gsm8k_benchmark(
                self.base_url, num_questions=self.num_questions
            )
            passed = acc >= self.accuracy_threshold
            status = "✅ PASS" if passed else "❌ FAIL"

            summary = f"### GROK1-INT4 (MI35x)\n\n"
            summary += f"| Model | Accuracy | Threshold | Status |\n"
            summary += f"| ----- | -------- | --------- | ------ |\n"
            summary += f"| {GROK1_INT4_MODEL_PATH} | {acc:.3f} | {self.accuracy_threshold} | {status} |\n"

            if is_in_ci():
                write_github_step_summary(summary)

            self.assertGreaterEqual(
                acc,
                self.accuracy_threshold,
                f"Accuracy {acc:.3f} below threshold {self.accuracy_threshold}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
