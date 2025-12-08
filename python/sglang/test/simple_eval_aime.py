# Adapted from https://github.com/openai/simple-evals/

"""
AIME 2025: American Invitational Mathematics Examination
A high school mathematics competition testing olympiad-level mathematical reasoning.
Problems require multi-step logical deductions with integer answers from 000-999.

Dataset: https://huggingface.co/datasets/opencompass/AIME2025
"""

import re
from typing import Optional

from datasets import load_dataset

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

QUERY_TEMPLATE = """
Solve the following math problem step by step. Put your final answer within \\boxed{{}}.

{problem}

Please reason step by step, and put your final answer within \\boxed{{}}.
""".strip()


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{answer} format.
    Returns the numeric answer as a string, or None if not found.
    """
    # Try to find \\boxed{...} pattern
    patterns = [
        r"\\boxed\{([^}]+)\}",  # Standard LaTeX boxed
        r"\\boxed\s*\{([^}]+)\}",  # With optional whitespace
        r"boxed\{([^}]+)\}",  # Without backslash (some models do this)
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Take the last occurrence (usually the final answer)
            answer = matches[-1].strip()
            # Extract just the number
            num_match = re.search(r"\d+", answer)
            if num_match:
                return num_match.group(0)

    # Fallback: try to find any 3-digit number at the end
    last_num = re.findall(r"\b\d{1,3}\b", text)
    if last_num:
        return last_num[-1]

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize AIME answer to 3-digit format (000-999).
    """
    if answer is None:
        return ""
    try:
        num = int(answer)
        # AIME answers are 000-999
        if 0 <= num <= 999:
            return f"{num:03d}"
    except (ValueError, TypeError):
        pass
    return ""


class AIMEEval(Eval):
    def __init__(
        self,
        num_examples: Optional[int],
        num_threads: int,
        split: str = "test",
        config: str = "AIME2025-I",
    ):
        """
        Initialize AIME evaluation.

        Args:
            num_examples: Number of examples to evaluate (None for all)
            num_threads: Number of threads for parallel evaluation
            split: Dataset split to use (default: "test")
            config: AIME configuration - "AIME2025-I" or "AIME2025-II"
        """
        # Load dataset from Hugging Face
        dataset = load_dataset("opencompass/AIME2025", name=config, split=split)

        examples = []
        for item in dataset:
            examples.append(
                {"problem": item["problem"], "answer": str(item["answer"])}
            )

        if num_examples:
            # Take first num_examples for consistent evaluation
            examples = examples[:num_examples]

        self.examples = examples
        self.num_threads = num_threads
        self.config = config

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=QUERY_TEMPLATE.format(**row), role="user"
                )
            ]
            response_text = sampler(prompt_messages)
            response_text = response_text or ""

            # Extract answer from response
            extracted_answer = extract_boxed_answer(response_text)

            # Normalize both answers for comparison
            correct_answer_norm = normalize_answer(row["answer"])
            extracted_answer_norm = normalize_answer(extracted_answer)

            # Score: 1.0 if match, 0.0 otherwise
            score = 1.0 if extracted_answer_norm == correct_answer_norm else 0.0

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer_norm,
                extracted_answer=extracted_answer_norm,
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
