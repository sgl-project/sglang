"""GSM8K few-shot completion benchmark kit.

Provides a reusable ``run_gsm8k_benchmark`` function for AMD accuracy tests
that launch their own SGLang server and evaluate it with the GSM8K few-shot
completion benchmark.

The three helper functions (``INVALID``, ``get_one_example``,
``get_few_shot_examples``, ``get_answer_value``) are re-exported from
``sglang.test.few_shot_gsm8k`` to avoid duplicating logic.

Usage::

    from sglang.test.kits.gsm8k_completion_kit import run_gsm8k_benchmark

    acc, invalid, latency = run_gsm8k_benchmark(base_url, num_questions=200)
"""

import time
from typing import Tuple

from sglang.test.few_shot_gsm8k import (
    INVALID,
    get_answer_value,
    get_few_shot_examples,
    get_one_example,
)
from sglang.utils import download_and_cache_file, read_jsonl

__all__ = [
    "INVALID",
    "get_answer_value",
    "get_few_shot_examples",
    "get_one_example",
    "run_gsm8k_benchmark",
]


def run_gsm8k_benchmark(
    base_url: str,
    num_questions: int = 200,
    num_shots: int = 5,
    parallel: int = 64,
) -> Tuple[float, float, float]:
    """Run the GSM8K few-shot completion benchmark against a running SGLang server.

    Args:
        base_url: Base URL of the SGLang server (e.g. ``"http://127.0.0.1:30000"``).
        num_questions: Number of test questions to evaluate.
        num_shots: Number of few-shot examples to prepend.
        parallel: Number of parallel requests.

    Returns:
        A 3-tuple ``(accuracy, invalid_rate, latency)`` where:

        - ``accuracy``     — fraction of questions answered correctly.
        - ``invalid_rate`` — fraction of predictions that could not be parsed
          (equal to ``INVALID``).
        - ``latency``      — wall-clock seconds for the full batch.
    """
    import numpy as np

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
    assert all(label != INVALID for label in labels)
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
