"""MI35x Qwen3-Coder-Next GSM8K Completion Evaluation Test (8-GPU)

Tests Qwen3-Coder-Next model with basic and MTP configurations
using few-shot completion benchmark on MI35x.

Registry: nightly-amd-8-gpu-mi35x-qwen3-coder-next suite
"""

import ast
import os

# Set HF cache for MI35x
os.environ.setdefault("HF_HOME", "/data2/models/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/data2/models/huggingface/hub")

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

# Register for AMD CI - MI35x Qwen3-Coder-Next accuracy test
register_amd_ci(est_time=3600, suite="nightly-amd-8-gpu-mi35x", nightly=True)

INVALID = -9999999

# Model path configuration for MI35x Qwen3-Coder-Next
# Priority: 1) env var, 2) local path
QWEN3_CODER_NEXT_LOCAL_PATH = "/data/Qwen/Qwen3-Coder-Next/"
QWEN3_CODER_NEXT_HF_MODEL_ID = "Qwen/Qwen3-Coder-Next"


def get_model_path() -> str:
    """Get effective model path: env var > local path > HF model ID."""
    env_path = os.environ.get("QWEN3_CODER_NEXT_MODEL_PATH")
    if env_path:
        return env_path
    if os.path.exists(QWEN3_CODER_NEXT_LOCAL_PATH):
        return QWEN3_CODER_NEXT_LOCAL_PATH
    return QWEN3_CODER_NEXT_HF_MODEL_ID


@dataclass
class ModelConfig:
    """Configuration for a model to test."""

    model_path: str
    tp_size: int = 8
    accuracy_threshold: float = 0.50
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


def get_qwen3_coder_next_models() -> List[ModelConfig]:
    """Get Qwen3-Coder-Next model configurations for MI35x."""
    model_path = get_model_path()
    return [
        # Basic — matches run_qwen3-coder-next_spec.sh
        ModelConfig(
            model_path=model_path,
            tp_size=8,
            accuracy_threshold=0.90,
            timeout=3600,
            variant="basic",
            other_args=[
                "--attention-backend",
                "aiter",
                "--chunked-prefill-size",
                "131072",
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.8",
                "--trust-remote-code",
                "--kv-cache-dtype",
                "fp8_e4m3",
            ],
        ),
        # MTP (speculative decoding)
        # TODO: Support MTP with fp8 kv cache on gfx950.
        # Note: no --kv-cache-dtype fp8_e4m3 because Triton extend_attention
        # used by MTP does not support fp8 kv cache on gfx950.
        ModelConfig(
            model_path=model_path,
            tp_size=8,
            accuracy_threshold=0.90,
            timeout=3600,
            variant="mtp",
            other_args=[
                "--attention-backend",
                "aiter",
                "--chunked-prefill-size",
                "131072",
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.8",
                "--trust-remote-code",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ],
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


class TestQwen3CoderNextEvalMI35x(unittest.TestCase):
    """Qwen3-Coder-Next GSM8K Completion Evaluation Test for AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.models = get_qwen3_coder_next_models()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_qwen3_coder_next_accuracy(self):
        """Test Qwen3-Coder-Next models with GSM8K completion benchmark."""
        # Check if model exists
        model_path = get_model_path()
        is_local_path = model_path.startswith("/")
        if is_local_path and not os.path.exists(model_path):
            print(f"\nSKIPPING: Local model not found at {model_path}")
            self.skipTest(f"Local model not found at {model_path}")
            return

        if is_local_path:
            print(f"Using local model: {model_path}")
        else:
            print(f"Using HuggingFace model: {model_path}")

        all_results = []
        summary = "### Qwen3-Coder-Next Models (MI35x)\n\n"
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
                            self.base_url, num_questions=self.num_questions
                        )
                        passed = acc >= config.accuracy_threshold
                        status = "PASS" if passed else "FAIL"
                        print(
                            f"  accuracy={acc:.3f} threshold={config.accuracy_threshold} {status}"
                        )

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
