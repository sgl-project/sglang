# Adapted from https://github.com/openai/simple-evals/ and simple_eval_aime25.py
"""AIME competition math evaluation (AIME 2024鈥?026)."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

QUERY_TEMPLATE = """
Solve the following AIME (American Invitational Mathematics Examination) problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

Note: AIME answers are always integers from 000 to 999 (inclusive). If you get a non-integer answer, you likely made a computational error.

{question}

Remember to put your answer on its own line after "Answer:", and express your answer as an integer from 000 to 999.
""".strip()


@dataclass(frozen=True)
class AimeSource:
    hf_id: str
    config: Optional[str] = None
    splits: Sequence[str] = ("test", "train")
    q_field: str = "question"
    a_field: str = "answer"


_AIME_SOURCES: Dict[int, List[AimeSource]] = {
    24: [
        AimeSource("HuggingFaceH4/aime_2024", q_field="problem"),
        AimeSource("Maxwell-Jia/AIME_2024", q_field="Problem", a_field="Answer"),
    ],
    25: [
        AimeSource("opencompass/AIME2025", "AIME2025-I"),
        AimeSource("opencompass/AIME2025", "AIME2025-II"),
    ],
    26: [
        AimeSource("MathArena/aime_2026", q_field="problem"),
        AimeSource("96kevinli29/aime2026-en", q_field="problem", a_field="ground_truth"),
    ],
}


def normalize_aime_answer(answer: Optional[str]) -> Optional[str]:
    """Normalize AIME answer to a canonical integer string (0鈥?99)."""
    if answer is None:
        return None
    answer = str(answer).strip()
    try:
        num = int(float(answer))
        if 0 <= num <= 999:
            return str(num)
    except (ValueError, TypeError):
        pass
    return answer


def _row_question_answer(row: Dict[str, Any], q_field: str, a_field: str) -> Dict[str, str]:
    question = row.get(q_field)
    answer = row.get(a_field)
    if question is None or answer is None:
        raise KeyError(f"missing {q_field}/{a_field} in dataset row")
    return {"question": str(question), "answer": str(answer)}


def _load_source(source: AimeSource) -> List[Dict[str, str]]:
    from datasets import load_dataset

    for split in source.splits:
        try:
            if source.config:
                dataset = load_dataset(source.hf_id, source.config, split=split)
            else:
                dataset = load_dataset(source.hf_id, split=split)
            return [
                _row_question_answer(row, source.q_field, source.a_field)
                for row in dataset
            ]
        except Exception:
            continue
    return []


def load_aime_examples(year: int) -> List[Dict[str, str]]:
    if year not in _AIME_SOURCES:
        raise ValueError(f"Unsupported AIME year: {year}")

    examples: List[Dict[str, str]] = []
    for source in _AIME_SOURCES[year]:
        examples.extend(_load_source(source))

    if not examples:
        raise ValueError(
            f"No AIME {year} examples loaded from HuggingFace "
            f"(set HF_ENDPOINT=https://hf-mirror.com if hub access is blocked)"
        )

    return examples


class AIMEEval(Eval):
    def __init__(
        self,
        year: int,
        num_examples: Optional[int],
        num_threads: int,
    ):
        self.examples = load_aime_examples(year)
        if num_examples is not None:
            self.examples = self.examples[: min(num_examples, len(self.examples))]
        self.num_threads = num_threads
        self.year = year

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=QUERY_TEMPLATE.format(**row), role="user"
                )
            ]
            response_text = sampler(prompt_messages) or ""
            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1).strip() if match else None
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
