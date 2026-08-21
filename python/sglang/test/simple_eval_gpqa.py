# Adapted from https://github.com/openai/simple-evals/

"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import hashlib
import json
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

GPQA_SOURCE_FIELDS = (
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)


class GPQAEval(Eval):
    def __init__(
        self,
        filename: str,
        num_examples: Optional[int],
        num_threads: int,
        n_repeats: int = 1,
    ):
        if "://" in filename:
            df = pandas.read_csv(filename, storage_options={"timeout": 30})
        else:
            df = pandas.read_csv(filename)
        missing_columns = set(GPQA_SOURCE_FIELDS) - set(df.columns)
        if missing_columns:
            raise ValueError(f"GPQA data is missing columns: {sorted(missing_columns)}")
        invalid_fields = [
            field
            for field in GPQA_SOURCE_FIELDS
            if any(
                not isinstance(value, str) or not value.strip() for value in df[field]
            )
        ]
        if invalid_fields:
            raise ValueError(
                "GPQA semantic fields must contain non-empty strings: "
                f"{invalid_fields}"
            )
        examples = [
            {field: row[field] for field in GPQA_SOURCE_FIELDS}
            | {"source_row_index": source_row_index}
            for source_row_index, (_, row) in enumerate(df.iterrows())
        ]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [
            example | {"permutation": rng.sample(range(4), 4)} for example in examples
        ]
        self.examples = examples
        self.n_repeats = n_repeats
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0],
                B=choices[1],
                C=choices[2],
                D=choices[3],
                Question=row["Question"],
            )
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
                )
            ]
            response_text = sampler(prompt_messages)
            if response_text is None:
                response_text = ""
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == correct_answer else 0.0
            question_sha256 = hashlib.sha256(row["Question"].encode()).hexdigest()
            source_row = {field: row[field] for field in GPQA_SOURCE_FIELDS}
            source_row_sha256 = hashlib.sha256(
                json.dumps(
                    source_row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            prompt_sha256 = hashlib.sha256(
                json.dumps(
                    prompt_messages,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={"chars": len(response_text)},
                example={
                    "question_id": (
                        f"gpqa-{row['source_row_index']:03d}-{question_sha256[:16]}"
                    ),
                    "source_row_index": row["source_row_index"],
                    "source_row_sha256": source_row_sha256,
                    "question_sha256": question_sha256,
                    "prompt_sha256": prompt_sha256,
                    "permutation": row["permutation"],
                    "expected_answer": correct_answer,
                    "parsed_answer": extracted_answer,
                    "correct": bool(score),
                    "response_sha256": hashlib.sha256(
                        response_text.encode()
                    ).hexdigest(),
                    "response_chars": len(response_text),
                },
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
