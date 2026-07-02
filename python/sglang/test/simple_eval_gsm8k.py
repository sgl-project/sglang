# Adapted from https://github.com/openai/simple-evals/

import ast
import re
from typing import Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)
from sglang.utils import download_and_cache_file, read_jsonl

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
INVALID = -9999999


def get_one_example(lines, i, include_answer):
    ret = f"Question: {lines[i]['question']}\nAnswer:"
    if include_answer:
        ret += f" {lines[i]['answer']}"
    return ret


def get_few_shot_examples(lines, k):
    return "".join(get_one_example(lines, i, True) + "\n\n" for i in range(k))


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    # GSM8K answers (both the ground-truth and the few-shot format the model
    # imitates) end with a `#### <number>` marker. Prefer the number after the
    # FIRST such marker: it is the answer to the question actually asked.
    # Instruction-tuned models that don't reliably stop after `####` (e.g.
    # Mistral/Mixtral) run on and emit further `Question:/#### N` pairs; taking
    # the last number in the whole string then scores the run-on continuation
    # instead of the real answer. Fall back to the last number when there is no
    # `####` marker (models that answer in prose).
    marker = re.search(r"####\s*(-?\d+\.?\d*)", answer_str)
    if marker is not None:
        try:
            return ast.literal_eval(marker.group(1))
        except (SyntaxError, ValueError):
            return INVALID
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


class GSM8KEval(Eval):
    def __init__(
        self,
        num_examples: Optional[int] = None,
        num_threads: int = 64,
        num_shots: int = 5,
        data_path: Optional[str] = None,
    ):
        self._num_threads = num_threads
        self._num_shots = num_shots

        if data_path:
            filename = data_path
        else:
            filename = download_and_cache_file(GSM8K_URL)

        all_lines = list(read_jsonl(filename))
        pool_size = self._setup_prefix_pool(all_lines, num_shots)
        # The evaluation data should not include the few-shot examples to prevent data leakage.
        self._lines = all_lines[pool_size:]
        if num_examples is not None:
            # Slice caps silently when num_examples exceeds the available lines,
            # matching upstream: callers like test_basic_sanity_eagle3 pass a
            # num_examples larger than the dataset on purpose.
            self._lines = self._lines[:num_examples]

    def _setup_prefix_pool(self, all_lines: list, num_shots: int) -> int:
        self._few_shot_prompt = get_few_shot_examples(all_lines, num_shots)
        return num_shots

    def _build_prefix(self, idx: int) -> str:
        return self._few_shot_prompt

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(idx: int) -> SingleEvalResult:
            question = get_one_example(self._lines, idx, include_answer=False)
            correct_answer = get_answer_value(self._lines[idx]["answer"])

            prompt_content = self._build_prefix(idx) + question
            prompt_messages = [
                sampler._pack_message(content=prompt_content, role="user")
            ]

            try:
                response_text = sampler(prompt_messages)
            except Exception:
                response_text = ""

            extracted_answer = get_answer_value(response_text)
            score = float(extracted_answer == correct_answer)

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            return SingleEvalResult(html=html, score=score, convo=convo)

        results = common.map_with_progress(
            fn, list(range(len(self._lines))), num_threads=self._num_threads
        )
        return common.aggregate_results(results, default_stats=("mean", "std"))
