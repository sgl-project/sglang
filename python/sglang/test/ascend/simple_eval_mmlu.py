# Adapted from https://github.com/openai/simple-evals/

"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import re
from typing import Optional

import pandas

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    format_multichoice_question,
)
from sglang.test.simple_eval_mmlu import subject2category


def format_multichoice_question_example(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

TEMPLATE_MULTICHOICE_EXAMPLE_BEGIN = """
Answer the multiple-choice questions following the examples below. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.

"""

TEMPLATE_MULTICHOICE_EXAMPLE = """
Example question:
{Question}

A {A}
B {B}
C {C}
D {D}

The last line of your response should be
Answer: {Answer}
""".strip()


class MMLUEval(Eval):
    def __init__(
        self,
        filename: str,
        num_examples: Optional[int],
        num_threads: int,
        num_shots: int,
    ):
        if "://" in filename:
            df = pandas.read_csv(filename, storage_options={"timeout": 30})
        else:
            df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_shots:
            example_questions = "".join(
                format_multichoice_question_example(row) + "\n\n"
                for row in examples[:num_shots]
            )
            self.template = (
                TEMPLATE_MULTICHOICE_EXAMPLE_BEGIN
                + example_questions
                + QUERY_TEMPLATE_MULTICHOICE
            )
            examples = examples[num_shots:]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.num_threads = num_threads
        self.num_shots = num_shots

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            if self.num_shots:
                prompt_messages = [
                    sampler._pack_message(
                        content=self.template.format(**row), role="user"
                    )
                ]
            else:
                prompt_messages = [
                    sampler._pack_message(
                        content=format_multichoice_question(row), role="user"
                    )
                ]
            response_text = sampler(prompt_messages)
            response_text = response_text or ""
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == row["Answer"] else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            category = subject2category.get(row["Subject"], "other")
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
