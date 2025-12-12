# Adapted from https://github.com/openai/simple-evals/

"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import random
import re
from typing import Optional

import pandas

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    check_equality,
)

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


class MathEval(Eval):
    def __init__(
        self,
        filename: str,
        equality_checker: SamplerBase,
        num_examples: Optional[int],
        num_threads: int,
    ):
        df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.equality_checker = equality_checker
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(**row), role="user")
            ]
            response_text = sampler(prompt_messages)
            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None
            score = float(
                check_equality(self.equality_checker, row["Answer"], extracted_answer)
            )
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
