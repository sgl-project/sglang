# Adapted from https://github.com/openai/simple-evals/

"""
LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-Context Multitasks
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li
https://arxiv.org/abs/2412.15204
"""

import random
import re
from typing import List, Optional, Union

import pandas

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
        choice_A = row.get("A", "")
        choice_B = row.get("B", "")
        choice_C = row.get("C", "")
        choice_D = row.get("D", "")

    # Official LongBench-v2 template
    prompt = f"""{context.strip()}

What is the correct answer to this question: {question.strip()}
Choices:
(A) {choice_A.strip()}
(B) {choice_B.strip()}
(C) {choice_C.strip()}
(D) {choice_D.strip()}

The correct answer is"""

    return prompt


def extract_longbench_v2_answer(response: str) -> Optional[str]:
    """Extract answer from model response using official LongBench-v2 method."""
    response = response.replace("*", "")

    # First try: "The correct answer is (A)"
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)

    # Second try: "The correct answer is A"
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)

    # Fallback: Standard SGLang multichoice pattern
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response)
    if match:
        return match.group(1)

    return None


class LongBenchV2Eval(Eval):
    """
    Evaluation utility for LongBench-v2 dataset.

    LongBench-v2 is designed to assess the ability of LLMs to handle long-context problems
    requiring deep understanding and reasoning across real-world multitasks.
    """

    def __init__(
        self,
        data_source: str = "zai-org/LongBench-v2",
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
            max_context_length: Maximum context length in tokens
            min_context_length: Minimum context length in tokens
        """
        # Load dataset based on data source type
        examples = self._load_dataset(data_source)

        # Apply filtering
        if categories:
            examples = [ex for ex in examples if ex.get("category") in categories]

        if min_context_length or max_context_length:
            examples = self._filter_by_context_length(
                examples, min_context_length, max_context_length
            )

        # Sample examples if specified
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when not sampling examples"
            examples = rng.sample(examples, min(num_examples, len(examples)))

        # Repeat examples for multiple runs
        examples = examples * n_repeats

        self.examples = examples
        self.n_repeats = n_repeats
        self.num_threads = num_threads

        print(f"Loaded {len(self.examples)} examples from LongBench-v2")
        if categories:
            print(f"Filtered to categories: {categories}")
        if min_context_length or max_context_length:
            print(
                f"Context length filter: {min_context_length}-{max_context_length} tokens"
            )

    def _load_dataset(self, data_source: str) -> List[dict]:
        """Load dataset from various sources."""
        examples = []

        if (
            data_source.startswith("http")
            or "/" in data_source
            and not data_source.endswith((".csv", ".json"))
        ):
            # HuggingFace dataset
            try:
                from datasets import load_dataset

                dataset = load_dataset(data_source, split="test")
                examples = [dict(row) for row in dataset]
            except ImportError:
                raise ImportError(
                    "Please install datasets library: pip install datasets"
                )
        elif data_source.endswith(".csv"):
            # CSV file
            df = pandas.read_csv(data_source)
            examples = [row.to_dict() for _, row in df.iterrows()]
        elif data_source.endswith(".json"):
            # JSON file
            import json

            with open(data_source, "r") as f:
                examples = json.load(f)
        else:
            # Assume it's a file path, try CSV first
            try:
                df = pandas.read_csv(data_source)
                examples = [row.to_dict() for _, row in df.iterrows()]
            except:
                # Try JSON
                import json

                with open(data_source, "r") as f:
                    examples = json.load(f)

        return examples

    def _filter_by_context_length(
        self, examples: List[dict], min_length: Optional[int], max_length: Optional[int]
    ) -> List[dict]:
        """Filter examples by context length."""
        filtered = []
        for example in examples:
            context = example.get("context", "")
            # Rough token count approximation (4 chars per token)
            token_count = len(context) // 4

            if min_length and token_count < min_length:
                continue
            if max_length and token_count > max_length:
                continue

            filtered.append(example)

        return filtered

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """Run the evaluation."""

        def fn(row: dict):
            # Format the question using official template
            formatted_question = format_longbench_v2_question(row)

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
            if isinstance(correct_answer, int):
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
            category = row.get("category", "unknown")
            if category in TASK_CATEGORIES:
                metrics[category] = score

            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
            )

        # Run evaluation with progress tracking
        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
