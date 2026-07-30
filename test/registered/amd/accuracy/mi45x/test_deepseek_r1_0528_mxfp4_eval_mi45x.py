"""MI45x DeepSeek-R1-0528-MXFP4 GSM8K Completion Evaluation Test (1-GPU)

Tests amd/DeepSeek-R1-0528-MXFP4 quantized model with triton attention
using few-shot completion benchmark on MI45x.

Registry: nightly-amd-1-gpu-mi45x-deepseek-r1-0528-mxfp4 suite
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

# Register for AMD CI - MI45x DeepSeek-R1-0528-MXFP4 accuracy test (~60 min)
register_amd_ci(
    est_time=3600,
    suite="nightly-amd-1-gpu-mi45x-deepseek-r1-0528-mxfp4",
    nightly=True,
)

INVALID = -9999999


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 1
    accuracy_threshold: float = 0.50
    other_args: Optional[List[str]] = None
    env_vars: Optional[dict] = None
    timeout: Optional[int] = None

    def __post_init__(self):
        if self.other_args is None:
            self.other_args = []
        if self.env_vars is None:
            self.env_vars = {}


MI45X_DEEPSEEK_R1_0528_MXFP4_MODELS = [
    ModelConfig(
        model_path="amd/DeepSeek-R1-0528-MXFP4",
        tp_size=1,
        accuracy_threshold=0.50,
        timeout=3600,
        other_args=[
            "--host",
            "0.0.0.0",
            "--mem-fraction-static",
            "0.90",
            "--chunked-prefill-size",
            "16384",
            "--attention-backend",
            "triton",
            "--max-running-requests",
            "32",
            "--kv-cache-dtype",
            "auto",
            "--page-size",
            "64",
            "--trust-remote-code",
        ],
        env_vars={
            "HSA_ENABLE_COREDUMP": "0",
            "HSA_COREDUMP_PATTERN": "/dev/null",
            "AMD_COREDUMP": "0",
            "ENABLE_CK": "0",
            "SGLANG_USE_AITER": "1",
            "AITER_FORCE_A8W4": "1",
            "AITER_GROUPED_FORCE_SPLIT_K1": "1",
            "SGLANG_MOE_SHUFFLE_GFX1250": "1",
            "ROCM_QUICK_REDUCE_QUANTIZATION": "NONE",
            "SGLANG_AITER_FP8_PREFILL_ATTN": "0",
            "SGLANG_AITER_MLA_PERSIST": "0",
            "SGLANG_INT4_WEIGHT": "0",
            "SGLANG_MOE_PADDING": "1",
            "SGLANG_SET_CPU_AFFINITY": "1",
            "SGLANG_ROCM_FUSED_DECODE_MLA": "0",
            "SGLANG_USE_ROCM700A": "1",
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


class TestDeepSeekR10528MXFP4EvalMI45x(unittest.TestCase):
    """DeepSeek-R1-0528-MXFP4 GSM8K Completion Evaluation Test for AMD MI45x."""

    @classmethod
    def setUpClass(cls):
        cls.models = MI45X_DEEPSEEK_R1_0528_MXFP4_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_deepseek_r1_0528_mxfp4_accuracy(self):
        """Test DeepSeek-R1-0528-MXFP4 with GSM8K completion benchmark."""
        all_results = []
        summary = "### DeepSeek-R1-0528-MXFP4 Models (MI45x)\n\n"
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
                other_args.extend(["--tensor-parallel-size", str(config.tp_size)])
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
