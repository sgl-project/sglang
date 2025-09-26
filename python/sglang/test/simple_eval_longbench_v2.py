# Adapted from https://github.com/openai/simple-evals/

"""
LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-Context Multitasks
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li
https://arxiv.org/abs/2412.15204
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
)

# LongBench-v2 domain categories
DOMAIN_CATEGORIES = {
    "single_document_qa": "single_document_qa",
    "multi_document_qa": "multi_document_qa", 
    "long_in_context_learning": "long_in_context_learning",
    "long_dialogue_history": "long_dialogue_history",
    "code_repo_understanding": "code_repo_understanding",
    "long_structured_data": "long_structured_data"
}


def format_longbench_v2_question(row: dict) -> str:
    """Format a LongBench-v2 question using the official template."""
    context = row.get("context", "")
    question = row.get("question", "")
    choice_A = row.get("choice_A", "")
    choice_B = row.get("choice_B", "")
    choice_C = row.get("choice_C", "")
    choice_D = row.get("choice_D", "")
    
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
    response = response.replace('*', '')
    
    # First try: "The correct answer is (A)"
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    
    # Second try: "The correct answer is A"
    match = re.search(r'The correct answer is ([A-D])', response)
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
        filename: str,
        num_examples: Optional[int] = None,
        num_threads: int = 1,
        n_repeats: int = 1,
    ):
        """
        Initialize LongBench-v2 evaluation.
        
        Args:
            filename: Path to CSV file containing LongBench-v2 data
            num_examples: Number of examples to evaluate (None for all)
            num_threads: Number of threads for parallel processing
            n_repeats: Number of times to repeat evaluation for error bars
        """
        # Load dataset from CSV
        df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        
        # Sample examples if specified
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when not sampling examples"
            examples = rng.sample(examples, num_examples)
        
        # Repeat examples for multiple runs
        examples = examples * n_repeats
        
        self.examples = examples
        self.n_repeats = n_repeats
        self.num_threads = num_threads
        
        print(f"Loaded {len(self.examples)} examples from LongBench-v2")
    
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
            
            # Add domain-specific metrics
            domain = row.get("domain", "unknown")
            if domain in DOMAIN_CATEGORIES:
                metrics[domain] = score
            
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
            )
        
        # Run evaluation with progress tracking
        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)