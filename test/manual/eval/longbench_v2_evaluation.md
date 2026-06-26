# LongBench-v2 Evaluation Guide

## Overview

LongBench-v2 is a benchmark designed to assess the ability of Large Language Models (LLMs) to handle long-context problems requiring deep understanding and reasoning across real-world multitasks. This guide explains how to use SGLang's LongBench-v2 evaluation utilities.

## Features

- **Context Length**: 8k to 2M words (majority under 128k)
- **Task Categories**: 6 major categories with 503 challenging multiple-choice questions
- **Difficulty**: Challenging enough that human experts achieve only 53.7% accuracy
- **Format**: All questions are multiple-choice for reliable evaluation

## Task Categories

1. **Single-Document QA**: Question answering within a single long document
2. **Multi-Document QA**: Cross-document reasoning and synthesis
3. **Long In-Context Learning**: Few-shot learning with long examples
4. **Long-Dialogue History**: Understanding long conversation histories
5. **Code Repository Understanding**: Analysis of large codebases
6. **Long Structured Data**: Comprehension of tables, JSON, and structured data

## Quick Start

### Basic Usage

```python
from sglang.test.simple_eval_longbench_v2 import LongBenchV2Eval
from sglang.test.simple_eval_common import ChatCompletionSampler

# Initialize evaluator with HuggingFace dataset
eval_obj = LongBenchV2Eval(
    data_source="THUDM/LongBench-v2",
    num_examples=10,  # Limit for testing
    num_threads=4
)

# Create sampler (pointing to your SGLang server)
sampler = ChatCompletionSampler(
    base_url="http://localhost:30000/v1",
    model="your-model-name"
)

# Run evaluation
result = eval_obj(sampler)
print(f"Overall Score: {result.score:.3f}")
print(f"Metrics: {result.metrics}")
```

### Using the Command Line

```bash
# Basic evaluation
python -m sglang.test.run_eval \
    --eval-name longbench_v2 \
    --port 30000 \
    --num-examples 50

# Evaluate specific categories
python -m sglang.test.run_eval \
    --eval-name longbench_v2 \
    --categories "single_document_qa,multi_document_qa" \
    --port 30000

# Filter by context length
python -m sglang.test.run_eval \
    --eval-name longbench_v2 \
    --max-context-length 100000 \
    --min-context-length 10000 \
    --port 30000
```

## Advanced Configuration

### Category-Specific Evaluation

```python
# Evaluate only specific task categories
eval_obj = LongBenchV2Eval(
    data_source="THUDM/LongBench-v2",
    categories=[
        "single_document_qa",
        "code_repo_understanding"
    ]
)
```

### Context Length Filtering

```python
# Focus on medium-length contexts
eval_obj = LongBenchV2Eval(
    data_source="THUDM/LongBench-v2",
    min_context_length=32000,  # characters
    max_context_length=128000  # characters
)
```

### Using Local Dataset

```python
# Load from local JSON file
eval_obj = LongBenchV2Eval(
    data_source="/path/to/longbench_v2.json",
    num_examples=100
)

# Load from local CSV file
eval_obj = LongBenchV2Eval(
    data_source="/path/to/longbench_v2.csv"
)
```

## Dataset Format

The expected format for LongBench-v2 examples:

```json
{
    "context": "Long context text...",
    "question": "Question about the context",
    "A": "First choice",
    "B": "Second choice",
    "C": "Third choice",
    "D": "Fourth choice",
    "answer": "A",
    "category": "single_document_qa"
}
```

Alternative format with choices as list:

```json
{
    "context": "Long context text...",
    "question": "Question about the context",
    "choices": ["First choice", "Second choice", "Third choice", "Fourth choice"],
    "answer": "A",
    "category": "multi_document_qa"
}
```

## Metrics and Scoring

### Overall Metrics

- **score**: Overall accuracy across all examples
- **chars**: Average response length in characters

### Category-Specific Metrics

Each task category gets its own metric:
- `single_document_qa`: Accuracy on single-document QA tasks
- `multi_document_qa`: Accuracy on multi-document QA tasks
- `long_in_context_learning`: Accuracy on in-context learning tasks
- `long_dialogue_history`: Accuracy on dialogue understanding tasks
- `code_repo_understanding`: Accuracy on code analysis tasks
- `long_structured_data`: Accuracy on structured data tasks

### Context Length Metrics

- `short_context`: Accuracy on contexts < 32k characters
- `medium_context`: Accuracy on contexts 32k-128k characters
- `long_context`: Accuracy on contexts > 128k characters
- `difficulty_easy` / `difficulty_hard`: Accuracy grouped by dataset difficulty labels

## Performance Considerations

### Memory Usage

LongBench-v2 contains very long contexts (up to 2M words). Consider:

1. **GPU Memory**: Ensure your model can handle the context lengths
2. **Batch Size**: Use smaller batch sizes for longer contexts
3. **Parallel Processing**: Adjust `num_threads` based on available resources

### Evaluation Time

- Full evaluation (503 examples) can take several hours
- Use `num_examples` parameter to limit evaluation size during development
- Consider filtering by context length to focus on specific ranges

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce context length limits or batch size
2. **Slow Evaluation**: Increase `num_threads` or reduce `num_examples`
3. **Dataset Loading**: Ensure `datasets` library is installed for HuggingFace integration

### Installation Requirements

```bash
pip install datasets  # For HuggingFace dataset support
```

## Example Results

Typical performance ranges for different model sizes:

- **Small models (7B)**: 35-45% accuracy
- **Medium models (13-30B)**: 45-55% accuracy
- **Large models (70B+)**: 55-65% accuracy
- **Human experts**: 53.7% accuracy

## Citation

If you use LongBench-v2 in your research, please cite:

```bibtex
@article{bai2024longbench,
  title={LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-Context Multitasks},
  author={Bai, Yushi and Tu, Shangqing and Zhang, Jiajie and Peng, Hao and Wang, Xiaozhi and Lv, Xin and Cao, Shulin and Xu, Jiazheng and Hou, Lei and Dong, Yuxiao and Tang, Jie and Li, Juanzi},
  journal={arXiv preprint arXiv:2412.15204},
  year={2024}
}
```
