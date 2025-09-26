# Adapted from https://github.com/openai/simple-evals/

"""
LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-Context Multitasks
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, Juanzi Li
https://arxiv.org/abs/2412.15204
"""

import json
import random
import re
from typing import Dict, List, Optional, Union

import pandas as pd

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    download_dataset,
    format_multichoice_question,
)

# LongBench-v2 task categories mapping
TASK_CATEGORIES = {
    "single_document_qa": "Single-Document QA",
    "multi_document_qa": "Multi-Document QA", 
    "long_in_context_learning": "Long In-Context Learning",
    "long_dialogue_history": "Long-Dialogue History Understanding",
    "code_repo_understanding": "Code Repository Understanding",
    "long_structured_data": "Long Structured Data Understanding"
}

# LongBench-v2 specific prompt template for long contexts
LONGBENCH_V2_TEMPLATE = """
Please read the following context carefully and answer the multiple-choice question below.

Context:
{context}

Question: {question}

A) {A}
B) {B}
C) {C}
D) {D}

Answer the question by selecting the most appropriate option. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of A, B, C, D.
""".strip()

class LongBenchV2Eval(Eval):
    """
    Evaluation utility for LongBench-v2 dataset.
    
    LongBench-v2 is designed to assess the ability of LLMs to handle long-context problems 
    requiring deep understanding and reasoning across real-world multitasks. It consists of 
    503 challenging multiple-choice questions with contexts ranging from 8k to 2M words 
    across six major task categories.
    """
    
    def __init__(
        self,
        data_source: Union[str, List[Dict]],
        num_examples: Optional[int] = None,
        num_threads: int = 1,
        categories: Optional[List[str]] = None,
        max_context_length: Optional[int] = None,
        min_context_length: Optional[int] = None,
        n_repeats: int = 1,
    ):
        """
        Initialize LongBench-v2 evaluation.
        
        Args:
            data_source: Path to dataset file, HuggingFace dataset name, or list of examples
            num_examples: Number of examples to evaluate (None for all)
            num_threads: Number of threads for parallel processing
            categories: List of task categories to include (None for all)
            max_context_length: Maximum context length in tokens (None for no limit)
            min_context_length: Minimum context length in tokens (None for no limit)
            n_repeats: Number of times to repeat evaluation for error bars
        """
        self.num_threads = num_threads
        self.n_repeats = n_repeats
        
        # Load dataset
        if isinstance(data_source, str):
            self.examples = self._load_dataset(data_source)
        elif isinstance(data_source, list):
            self.examples = data_source
        else:
            raise ValueError("data_source must be a file path, dataset name, or list of examples")
        
        # Filter by categories if specified
        if categories:
            valid_categories = set(TASK_CATEGORIES.keys())
            invalid_categories = set(categories) - valid_categories
            if invalid_categories:
                raise ValueError(f"Invalid categories: {invalid_categories}. Valid categories: {valid_categories}")
            self.examples = [ex for ex in self.examples if ex.get("category") in categories]
        
        # Filter by context length if specified
        if max_context_length or min_context_length:
            self.examples = self._filter_by_length(self.examples, min_context_length, max_context_length)
        
        # Sample examples if specified
        rng = random.Random(0)
        if num_examples and num_examples < len(self.examples):
            assert n_repeats == 1, "n_repeats only supported when not sampling examples"
            self.examples = rng.sample(self.examples, num_examples)
        
        # Repeat examples for multiple runs
        self.examples = self.examples * n_repeats
        
        print(f"Loaded {len(self.examples)} examples from LongBench-v2")
        
        # Add category distribution info
        if self.examples:
            category_counts = {}
            for example in self.examples:
                category = example.get("category", "unknown")
                category_counts[category] = category_counts.get(category, 0) + 1
            print(f"Category distribution: {category_counts}")
    
    def _load_dataset(self, data_source: str) -> List[Dict]:
        """Load dataset from various sources."""
        try:
            # Try loading as HuggingFace dataset
            if not data_source.endswith(('.json', '.jsonl', '.csv')):
                try:
                    from datasets import load_dataset
                    dataset = load_dataset(data_source)
                    # Assume 'test' split exists, fallback to first available split
                    split_name = 'test' if 'test' in dataset else list(dataset.keys())[0]
                    return [dict(item) for item in dataset[split_name]]
                except ImportError:
                    raise ImportError("datasets library not installed. Install with: pip install datasets")
                except Exception as e:
                    print(f"Failed to load as HuggingFace dataset: {e}")
                    raise ValueError(f"Could not load dataset from {data_source}")
            
            # Try loading as local file
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
                return [row.to_dict() for _, row in df.iterrows()]
            elif data_source.endswith('.json'):
                with open(data_source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else [data]
            elif data_source.endswith('.jsonl'):
                examples = []
                with open(data_source, 'r', encoding='utf-8') as f:
                    for line in f:
                        examples.append(json.loads(line.strip()))
                return examples
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {data_source}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")
    
    def _filter_by_length(self, examples: List[Dict], min_length: Optional[int], max_length: Optional[int]) -> List[Dict]:
        """Filter examples by context length."""
        filtered = []
        for example in examples:
            context = example.get("context", "")
            # Rough token count estimation (4 chars per token)
            estimated_tokens = len(context) // 4
            
            if min_length and estimated_tokens < min_length:
                continue
            if max_length and estimated_tokens > max_length:
                continue
            
            filtered.append(example)
        
        print(f"Filtered from {len(examples)} to {len(filtered)} examples based on length constraints")
        return filtered
    
    def _format_longbench_question(self, example: Dict) -> str:
        """Format a LongBench-v2 example into a multiple choice question."""
        # Extract fields with fallbacks for different dataset formats
        context = example.get("context", example.get("input", ""))
        question = example.get("question", example.get("query", ""))
        
        # Handle different choice formats
        if "choices" in example:
            choices = example["choices"]
            if isinstance(choices, list) and len(choices) >= 4:
                return LONGBENCH_V2_TEMPLATE.format(
                    context=context,
                    question=question,
                    A=choices[0],
                    B=choices[1], 
                    C=choices[2],
                    D=choices[3]
                )
        
        # Fallback to individual choice fields
        choice_dict = {
            "context": context,
            "question": question,
            "A": example.get("A", example.get("option_a", "")),
            "B": example.get("B", example.get("option_b", "")),
            "C": example.get("C", example.get("option_c", "")),
            "D": example.get("D", example.get("option_d", ""))
        }
        
        return LONGBENCH_V2_TEMPLATE.format(**choice_dict)
    
    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """Run the evaluation."""
        def fn(example: Dict):
            # Format the question
            formatted_question = self._format_longbench_question(example)
            
            prompt_messages = [
                sampler._pack_message(content=formatted_question, role="user")
            ]
            
            # Get model response
            response_text = sampler(prompt_messages)
            if response_text is None:
                response_text = ""
            
            # Extract answer
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            
            # Get correct answer
            correct_answer = example.get("answer", example.get("label", example.get("correct_answer", "")))
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
            category = example.get("category", "unknown")
            if category in TASK_CATEGORIES:
                metrics[category] = score
            
            # Add context length metrics
            context = example.get("context", example.get("input", ""))
            context_length = len(context) // 4  # Rough token estimation
            if context_length < 32000:
                metrics["short_context"] = score
            elif context_length < 128000:
                metrics["medium_context"] = score
            else:
                metrics["long_context"] = score
            
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics=metrics,
            )
        
        # Run evaluation with progress tracking
        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)


def download_longbench_v2_dataset(save_path: str = "longbench_v2.json"):
    """
    Download LongBench-v2 dataset.
    
    Args:
        save_path: Path where to save the dataset
    """
    # HuggingFace dataset URL - this would need to be updated with actual URL
    url = "https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/longbench_v2.json"
    
    try:
        download_dataset(save_path, url)
        print(f"LongBench-v2 dataset downloaded to {save_path}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("You can manually download from: https://huggingface.co/datasets/zai-org/LongBench-v2")
        print("Or use: from datasets import load_dataset; dataset = load_dataset('zai-org/LongBench-v2')")