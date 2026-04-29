# Adapted from https://github.com/openai/simple-evals/

"""
AIME 2026 - American Invitational Mathematics Examination 2026
Dataset: MathArena/aime_2026
https://huggingface.co/datasets/MathArena/aime_2026

Prompt, dataset, and answer-extraction follow the matharena evaluation
(https://github.com/eth-sri/matharena), which reproduces the published
reasoning-model AIME numbers. Reasoning models are trained to emit \\boxed{N},
so we extract the last \\boxed{...} or \\fbox{...}.
"""

from typing import Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_aime25 import extract_boxed_answer, normalize_aime_answer
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

QUERY_TEMPLATE = """Put your final answer within \\boxed{{}}.
The answer is an integer between 0 and 999 inclusive.

{question}"""


class AIME26Eval(Eval):
    def __init__(
        self,
        num_examples: Optional[int],
        num_threads: int,
    ):
        from datasets import load_dataset

        dataset = load_dataset("MathArena/aime_2026", split="train")
        examples = [
            {"question": row["problem"], "answer": str(row["answer"])}
            for row in dataset
        ]

        if num_examples:
            examples = examples[: min(num_examples, len(examples))]

        self.examples = examples
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(**row), role="user")
            ]
            response_text = sampler(prompt_messages)
            response_text = response_text or ""

            extracted_answer = extract_boxed_answer(response_text)
            extracted_answer = extracted_answer.strip() if extracted_answer else None

            normalized_extracted = normalize_aime_answer(extracted_answer)
            normalized_correct = normalize_aime_answer(row["answer"])

            score = 1.0 if normalized_extracted == normalized_correct else 0.0

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={"chars": len(response_text)},
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
