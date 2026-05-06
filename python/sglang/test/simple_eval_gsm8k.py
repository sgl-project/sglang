# Adapted from https://github.com/openai/simple-evals/

import ast
import re
from typing import Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN,
    HTML_JINJA,
    ChatCompletionSampler,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)
from sglang.utils import download_and_cache_file, read_jsonl

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
INVALID = -9999999

# Prefix used for the chat-mode wrapper. The instruction is intentionally short
# and mirrors what the (now-removed) MGSM-EN eval used to send, since that
# format was empirically known to work across instruction-tuned model families
# whose chat templates strictly wrap user input (e.g. Mistral [INST]...[/INST]).
CHAT_MODE_INSTRUCTION = (
    "Solve the following math problems. Show your reasoning, then write the "
    "final answer on the last line in the exact format `Answer: <integer>` "
    "with nothing after the integer.\n\n"
    "Here are some worked examples:\n\n"
)


def get_one_example(lines, i, include_answer):
    ret = f"Question: {lines[i]['question']}\nAnswer:"
    if include_answer:
        ret += f" {lines[i]['answer']}"
    return ret


def get_few_shot_examples(lines, k):
    return "".join(get_one_example(lines, i, True) + "\n\n" for i in range(k))


def get_answer_value(answer_str):
    """Fallback extractor: take the last number in the response."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def extract_answer(response_text: str):
    """Extract the model's answer.

    First try the explicit `Answer: <number>` form (what we ask the model to
    produce in chat mode). Fall back to the last number in the response, which
    is what the original few-shot completion-mode behaviour relied on.
    """
    match = re.search(ANSWER_PATTERN, response_text)
    if match:
        candidate = get_answer_value(match.group(1))
        if candidate != INVALID:
            return candidate
    return get_answer_value(response_text)


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

        self._lines = list(read_jsonl(filename))
        self._few_shot_prompt = get_few_shot_examples(self._lines, num_shots)

        # The evaluation data should not include the few-shot examples to prevent data leakage.
        self._lines = self._lines[num_shots:]
        if num_examples is not None:
            self._lines = self._lines[:num_examples]

    def _build_prompt(self, question: str, sampler: SamplerBase) -> str:
        """Build a sampler-appropriate prompt.

        Completion mode keeps the original raw `Question/Answer` few-shot form
        (matches OpenAI simple-evals reference). Chat mode wraps the few-shot
        block with a short instruction asking the model to end with
        `Answer: <integer>`, which:
          (1) gives instruction-tuned models (esp. Mistral / Mixtral whose
              chat templates strictly wrap the prompt in [INST]...[/INST]) a
              clear task description instead of forcing them to "complete" a
              `Answer:` token in chat mode, and
          (2) lets the response be parsed via `Answer:` regex first, falling
              back to last-number extraction for models that ignore the format
              hint.
        """
        if isinstance(sampler, ChatCompletionSampler):
            return (
                CHAT_MODE_INSTRUCTION
                + self._few_shot_prompt
                + "Now solve this problem:\n\n"
                + question
            )
        return self._few_shot_prompt + question

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(idx: int) -> SingleEvalResult:
            question = get_one_example(self._lines, idx, include_answer=False)
            correct_answer = get_answer_value(self._lines[idx]["answer"])

            prompt_content = self._build_prompt(question, sampler)
            prompt_messages = [
                sampler._pack_message(content=prompt_content, role="user")
            ]

            try:
                response_text = sampler(prompt_messages)
            except Exception:
                response_text = ""

            extracted_answer = extract_answer(response_text)
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
