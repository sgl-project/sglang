# Adapted from https://github.com/openai/simple-evals/

"""
LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-Context Multitasks
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li
https://arxiv.org/abs/2412.15204
"""

import csv
import json
import os
import re
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

# LongBench-v2 task categories
TASK_CATEGORIES = {
    "single_document_qa",
    "multi_document_qa",
    "long_in_context_learning",
    "long_dialogue_history",
    "code_repo_understanding",
    "long_structured_data",
}

DEFAULT_DATASET = "THUDM/LongBench-v2"
DEFAULT_DATASET_SPLIT = "train"


def format_longbench_v2_question(row: dict) -> str:
    """Format a LongBench-v2 question using the official template."""
    context = row.get("context", "")
    question = row.get("question", "")

    # Handle both standard format (A, B, C, D) and alternative format (choices list)
    if "choices" in row:
        choices = row["choices"]
        choice_A = choices[0] if len(choices) > 0 else ""
        choice_B = choices[1] if len(choices) > 1 else ""
        choice_C = choices[2] if len(choices) > 2 else ""
        choice_D = choices[3] if len(choices) > 3 else ""
    else:
        choice_A = row.get("A", row.get("choice_A", ""))
        choice_B = row.get("B", row.get("choice_B", ""))
        choice_C = row.get("C", row.get("choice_C", ""))
        choice_D = row.get("D", row.get("choice_D", ""))

    # Official LongBench-v2 template
    prompt = f"""
Please read the following text and answer the question below.
<text>
{context.strip()}
</text>

What is the correct answer to this question: {question.strip()}
Choices:
(A) {choice_A.strip()}
(B) {choice_B.strip()}
(C) {choice_C.strip()}
(D) {choice_D.strip()}

Format your response as follows: "The correct answer is (insert answer here)"."""

    return prompt


def extract_longbench_v2_answer(response: str) -> Optional[str]:
    """Extract answer from model response using official LongBench-v2 method."""
    response = response.replace("*", "")

    # First try: "The correct answer is (A)"
    match = re.search(r"The correct answer is \(([A-D])\)", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Second try: "The correct answer is A"
    match = re.search(r"The correct answer is ([A-D])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: Standard SGLang multichoice pattern
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
    if match:
        return match.group(1).upper()

    # Generic fallback when model says "answer is A"
    match = re.search(r"answer\s+is\s*\(?([A-D])\)?", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


class LongBenchV2Eval(Eval):
    """
    Evaluation utility for LongBench-v2 dataset.

    LongBench-v2 is designed to assess the ability of LLMs to handle long-context problems
    requiring deep understanding and reasoning across real-world multitasks.
    """

    def __init__(
        self,
        model: str = None,
        data_source: str = DEFAULT_DATASET,
        num_examples: Optional[int] = None,
        num_threads: int = 1,
        n_repeats: int = 1,
        categories: Optional[List[str]] = None,
        max_context_length: Optional[int] = None,
        min_context_length: Optional[int] = None,
    ):
        """
        Initialize LongBench-v2 evaluation.

        Args:
            data_source: HuggingFace dataset name, local file path (CSV/JSON)
            num_examples: Number of examples to evaluate (None for all)
            num_threads: Number of threads for parallel processing
            n_repeats: Number of times to repeat evaluation for error bars
            categories: List of task categories to include (None for all)
            max_context_length: Maximum context length in characters
            min_context_length: Minimum context length in characters
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.min_context_length = min_context_length
        self.max_context_length = max_context_length
        # Load dataset based on data source type
        examples = self._load_dataset(data_source)

        # Apply filtering
        if categories:
            examples = [ex for ex in examples if ex.get("category") in categories]

        # Sample examples if specified
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when not sampling examples"
            examples = examples[: min(num_examples, len(examples))]

        # Repeat examples for multiple runs
        examples = examples * n_repeats

        if not examples:
            raise ValueError(
                "No examples available for LongBench-v2 evaluation after filtering"
            )

        self.examples = examples
        self.n_repeats = n_repeats
        self.num_threads = num_threads

        print(f"Loaded {len(self.examples)} examples from LongBench-v2")
        if categories:
            print(f"Filtered to categories: {categories}")
        if min_context_length or max_context_length:
            print(
                f"Context length filter: {min_context_length}-{max_context_length} characters"
            )

    def _load_dataset(self, data_source: str) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace hub or local files."""

        if not data_source:
            data_source = DEFAULT_DATASET

        if os.path.exists(data_source):
            raw_examples = self._load_local_file(data_source)
        else:
            raw_examples = self._load_hf_dataset(data_source)

        return [self._normalize_example(example) for example in raw_examples]

    def _load_local_file(self, path: str) -> List[Dict[str, Any]]:
        """Load examples from a local CSV/JSON/JSONL file."""

        suffix = os.path.splitext(path)[1].lower()
        if suffix in {".json", ".jsonl"}:
            with open(path, "r", encoding="utf-8") as fh:
                if suffix == ".jsonl":
                    data = [json.loads(line) for line in fh if line.strip()]
                else:
                    data = json.load(fh)
        elif suffix == ".csv":
            with open(path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                data = list(reader)
        else:
            # Try JSON, then CSV as fallback
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except json.JSONDecodeError:
                with open(path, "r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    data = list(reader)

        if isinstance(data, dict):
            data = data.get("data", [])

        if not isinstance(data, list):
            raise ValueError("Expected list of examples from local file")

        return data

    def _load_hf_dataset(self, identifier: str) -> List[Dict[str, Any]]:
        """Load the dataset from HuggingFace Hub."""

        parts = identifier.split(":", maxsplit=1)
        dataset_name = parts[0]
        split = parts[1] if len(parts) == 2 else DEFAULT_DATASET_SPLIT

        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Please install the 'datasets' package to load LongBench-v2 from HuggingFace: pip install datasets"
            ) from exc

        dataset = load_dataset(dataset_name, split=split)
        return [dict(row) for row in dataset]

    def _normalize_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure each example exposes the expected keys."""

        normalized = dict(example)

        for letter in ["A", "B", "C", "D"]:
            choice_key = f"choice_{letter}"
            if letter not in normalized and choice_key in normalized:
                normalized[letter] = normalized[choice_key]

        if "category" not in normalized and "domain" in normalized:
            normalized["category"] = normalized["domain"]

        answer = normalized.get("answer")
        if isinstance(answer, str):
            normalized["answer"] = answer.strip().upper()
        elif isinstance(answer, int) and 0 <= answer < 4:
            normalized["answer"] = ["A", "B", "C", "D"][answer]

        return normalized

    def _check_context_length(
        self,
        formatted_question: str,
        tokenizer: AutoTokenizer,
        min_length: Optional[int],
        max_length: Optional[int],
    ) -> bool:
        """Filter examples by context length measured in characters."""
        input_ids = tokenizer.encode(formatted_question)
        context_length = len(input_ids)

        if min_length is not None and context_length < min_length:
            return False
        if max_length is not None and context_length > max_length:
            return False

        return True

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """Run the evaluation."""

        def fn(row: dict):
            # Format the question using official template
            formatted_question = format_longbench_v2_question(row)

            if self.min_context_length or self.max_context_length:
                if not self._check_context_length(
                    formatted_question,
                    self.tokenizer,
                    self.min_context_length,
                    self.max_context_length,
                ):
                    # Skip this example
                    return None

            prompt_messages = [
                sampler._pack_message(content=formatted_question, role="user")
            ]

            # Get model response
            response_text = sampler(prompt_messages)
            if response_text is None:
                response_text = ""

            # Extract answer using official method
            extracted_answer = extract_longbench_v2_answer(response_text)

            # Get correct answer
            correct_answer = row.get("answer", "")
            if isinstance(correct_answer, str):
                correct_answer = correct_answer.strip().upper()
            elif isinstance(correct_answer, int) and 0 <= correct_answer < 4:
                correct_answer = ["A", "B", "C", "D"][correct_answer]

            # Calculate score
            score = 1.0 if extracted_answer == correct_answer else 0.0

            # Generate HTML report
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )

            # Build conversation
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            # Prepare metrics
            metrics = {"chars": len(response_text)}

            # Add category-specific metrics
            category = row.get("category", row.get("domain", "unknown"))
            if category in TASK_CATEGORIES:
                metrics[category] = score

            difficulty = row.get("difficulty")
            if isinstance(difficulty, str) and difficulty:
                metrics[f"difficulty_{difficulty.lower()}"] = score

            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
            )

        # Run evaluation with progress tracking
        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
