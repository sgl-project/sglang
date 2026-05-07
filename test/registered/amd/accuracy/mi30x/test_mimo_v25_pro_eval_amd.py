"""AMD MiMo-V2.5-Pro GSM8K Completion Evaluation Test (8-GPU)

Tests XiaomiMiMo/MiMo-V2.5-Pro (1.02T MoE, 42B active, FP8) with TP=8 using
the few-shot completion benchmark on MI325/MI300X.

The model uses the native SGLang ``MiMoV2ForCausalLM`` architecture with hybrid
attention (interleaved sliding-window + full-attention layers, head_dim=192,
GQA ratio 16). Server args follow AMD's published Day-0 SGLang deployment
recipe for MiMo-V2.5-Pro on Instinct GPUs:

  https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-xiaomi-mimo-v2-5-pro-on-amd-instinct-gpus-.html

Key choices from that recipe: ``--attention-backend triton`` (the AMD-validated
backend for this model on Instinct), no expert parallelism, ``--disable-radix-cache``,
and large chunked prefill (128 K).

Note on EAGLE: the AMD recipe also enables multi-layer EAGLE speculative
decoding via ``--enable-multi-layer-eagle`` + ``SGLANG_ENABLE_SPEC_V2=1``. On
the current AMD CI image that path hits an upstream Triton compilation error
inside ``extend_attention.py`` when ``SLIDING_WINDOW_SIZE > 0`` is combined
with the multi-layer EAGLE draft-extend cuda graph capture (the MiMo MTP head
goes through the hybrid SWA attention). Base inference passes that capture
fine. We omit the EAGLE flags here so the nightly is green; once the upstream
issue is resolved the EAGLE flags can be added back.

Registry: nightly-amd-accuracy-8-gpu-mimo-v25-pro suite
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

register_amd_ci(
    est_time=5400,
    suite="nightly-amd-accuracy-8-gpu-mimo-v25-pro",
    nightly=True,
)

INVALID = -9999999


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


MIMO_V25_PRO_MODELS = [
    ModelConfig(
        model_path="XiaomiMiMo/MiMo-V2.5-Pro",
        tp_size=8,
        accuracy_threshold=0.90,
        timeout=5400,
        variant="TP8",
        # Server args mirror AMD's published Day-0 SGLang recipe for
        # MiMo-V2.5-Pro on Instinct GPUs (triton attention, no expert
        # parallelism, large chunked prefill).
        # Source: https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-xiaomi-mimo-v2-5-pro-on-amd-instinct-gpus-.html
        # EAGLE / multi-layer EAGLE flags are intentionally omitted -- see
        # module docstring for the upstream Triton issue that path hits on
        # the current AMD CI image.
        other_args=[
            "--trust-remote-code",
            "--attention-backend",
            "triton",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.8",
            "--chunked-prefill-size",
            "131072",
            "--max-running-requests",
            "64",
            "--watchdog-timeout",
            "1200",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ],
        env_vars={
            "SGLANG_USE_AITER": "1",
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
            "answer", max_tokens=4096, stop=["Question", "Assistant:", "<|separator|>"]
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


class TestMiMoV25ProEvalAMD(unittest.TestCase):
    """MiMo-V2.5-Pro GSM8K Completion Evaluation Test for AMD MI325/MI300X."""

    @classmethod
    def setUpClass(cls):
        cls.models = MIMO_V25_PRO_MODELS
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.num_questions = int(os.environ.get("GSM8K_NUM_QUESTIONS", "200"))

    def test_mimo_v25_pro_accuracy(self):
        """Test MiMo-V2.5-Pro with GSM8K completion benchmark."""
        all_results = []
        summary = "### MiMo-V2.5-Pro Models (MI325)\n\n"
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
