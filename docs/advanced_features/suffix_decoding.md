# Suffix Decoding

Suffix Decoding is a training-free, CPU-only speculative decoding method that can improve throughput for structured or repetitive outputs.

## Overview

Suffix Decoding ([technical report](https://arxiv.org/abs/2411.04975)) is a novel speculation method that:

- **No draft model required** - Operates entirely on CPU using pattern matching
- **Dynamic speculation** - Adapts speculation length per request at each step
- **Frequency-based proposals** - Uses token frequency counts for better acceptance rates
- **Dual pattern matching** - Matches against both prompts and previous generations

Unlike n-gram methods, Suffix Decoding can achieve better performance for tasks with high repetition, such as code-editing, agentic loops (e.g., self-reflection, self-consistency), and RL rollouts.

## Installation

Suffix Decoding requires the Arctic Inference library:

```bash
pip install arctic-inference==0.1.1
```

## Quick Start

### Basic Usage

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 24
```

### Advanced Configuration

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-8B-Instruct \
    --speculative-algorithm SUFFIX \
    --speculative-num-draft-tokens 32 \
    --speculative-suffix-max-tree-depth 24 \
    --speculative-suffix-max-cached-requests 10000 \
    --speculative-suffix-max-spec-factor 1.5 \
    --speculative-suffix-min-token-prob 0.05
```

## Configuration Parameters

### Required Parameters

- `--speculative-algorithm SUFFIX`: Enable suffix decoding
- `--speculative-num-draft-tokens <int>`: Maximum number of speculative tokens (default: 24)
  - Suffix decoding dynamically adjusts this per request
  - Recommended: 16-32 for best performance

### Optional Parameters

- `--speculative-suffix-max-tree-depth <int>`: Maximum depth of suffix trees (default: 24)
  - Limits the sum of prefix match and speculation lengths
  - Higher values enable longer pattern matching but use more memory

- `--speculative-suffix-max-cached-requests <int>`: Maximum cached requests (default: 10000)
  - Controls global suffix tree size
  - Set to 0 to disable global cache (only use prompt trees)
  - FIFO eviction when limit exceeded

- `--speculative-suffix-max-spec-factor <float>`: Speculation factor (default: 1.0)
  - Controls speculation length: `max_spec = factor * prefix_match_length`
  - Higher values = more aggressive speculation
  - Recommended range: 0.5 - 2.0

- `--speculative-suffix-min-token-prob <float>`: Minimum token probability (default: 0.1)
  - Only speculate tokens above this probability threshold
  - Based on frequency counts in suffix tree
  - Lower values = more aggressive, higher = more conservative
  - Valid range: 0.0 - 1.0

## Use Cases

Suffix Decoding excels at tasks with high repetition:

### 1. Code Editing
```python
# Example: Refactoring code with repetitive patterns
prompt = """Refactor this function to use list comprehension:

```python
result = []
for x in data:
    if x > 0:
        result.append(x * 2)
```"""
```

### 2. Agentic Loops
```python
# Example: Self-reflection loop
prompt = "Analyze this solution, then improve it, then analyze again:\n\n..."
```

### 3. Structured Output Generation
```python
# Example: Generating JSON with repeated structure
prompt = "Generate a list of 10 users in JSON format with fields: name, email, age"
```

### 4. RL Rollouts
```python
# Example: Generating multiple trajectories
prompt = "Generate 5 different approaches to solve this problem:\n\n..."
```

## Performance Tuning

### For High Repetition Tasks

```bash
# Aggressive speculation
--speculative-suffix-max-spec-factor 2.0 \
--speculative-suffix-min-token-prob 0.05 \
--speculative-num-draft-tokens 32
```

### For Mixed Workloads

```bash
# Balanced settings (default)
--speculative-suffix-max-spec-factor 1.0 \
--speculative-suffix-min-token-prob 0.1 \
--speculative-num-draft-tokens 24
```

### For Memory-Constrained Systems

```bash
# Conservative caching
--speculative-suffix-max-cached-requests 1000 \
--speculative-suffix-max-tree-depth 16
```

### Disable Global Cache

```bash
# Use only prompt trees (no cross-request caching)
--speculative-suffix-max-cached-requests 0
```

## How It Works

1. **Prompt Tree Building**: When a request starts, build a suffix tree from the prompt
2. **Pattern Matching**: Extract last N tokens as pattern (N ≤ max_tree_depth)
3. **Frequency-Based Speculation**: Find matching suffixes and rank by frequency
4. **Dynamic Length**: Speculate `min(max_spec_tokens, spec_factor * match_length)` tokens
5. **Verification**: Target model verifies draft tokens in parallel
6. **Cache Update**: Add accepted tokens to global suffix tree for future requests

## Comparison with Other Methods

| Feature | Suffix | N-gram | EAGLE | Draft Model |
|---------|--------|--------|-------|-------------|
| Extra Model | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| GPU Memory | ❌ No | ❌ No | ✅ High | ✅ High |
| CPU Only | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| Dynamic Length | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Cross-Request Cache | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Frequency-Based | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Best For | Repetition | Prompts | General | General |

## Troubleshooting

### Import Error

```
ImportError: Arctic Inference is required for suffix decoding
```

**Solution:** Install arctic-inference:
```bash
pip install arctic-inference==0.1.1
```

### Low Acceptance Rate

If you see low acceptance rates:

1. **Increase `max_spec_factor`**: Try 1.5 or 2.0
2. **Lower `min_token_prob`**: Try 0.05 instead of 0.1
3. **Increase `max_tree_depth`**: Try 32 instead of 24
4. **Check task repetition**: Suffix works best with repetitive patterns

### Memory Issues

If running out of memory:

1. **Reduce `max_cached_requests`**: Try 1000 or 5000
2. **Reduce `max_tree_depth`**: Try 16 instead of 24
3. **Disable global cache**: Set `max_cached_requests=0`

### No Speedup

If not seeing speedup:

1. **Check task type**: Suffix works best with repetition
2. **Monitor acceptance rate**: Should be >50% for benefit
3. **Try other methods**: EAGLE or draft model for general tasks
4. **Increase batch size**: Suffix benefits from larger batches

## Monitoring

Key metrics to monitor:

- **Acceptance Rate**: Percentage of draft tokens accepted
  - Target: >50% for noticeable speedup
  - Check via server logs or metrics endpoint

- **Average Draft Length**: Tokens speculated per step
  - Should vary dynamically based on patterns
  - Higher for repetitive tasks

- **Cache Hit Rate**: Frequency of global cache matches
  - Higher = better cross-request reuse
  - Only relevant if `max_cached_requests > 0`

## Python API Example

```python
from sglang import Engine

# Initialize with suffix decoding
engine = Engine(
    model_path="meta-llama/Llama-3-8B-Instruct",
    speculative_algorithm="SUFFIX",
    speculative_num_draft_tokens=24,
    speculative_suffix_max_spec_factor=1.5,
)

# Generate with high repetition task
prompts = [
    "Generate 5 user profiles in JSON format with fields: name, email, age, city"
]

outputs = engine.generate(prompts, sampling_params={"temperature": 0.7, "max_new_tokens": 500})
for output in outputs:
    print(output["text"])
```

## References

- [Suffix Decoding Paper](https://arxiv.org/abs/2411.04975)
- [Arctic Inference GitHub](https://github.com/snowflakedb/ArcticInference)
- [vLLM Implementation](https://github.com/vllm-project/vllm/pull/25784)

## Limitations

- **CUDA Only**: Suffix decoding currently requires CUDA devices
- **No DP Attention**: Data parallel attention is not yet supported
- **CPU Overhead**: Pattern matching on CPU may add latency for small batches
- **Best for Repetition**: Performance gains primarily on repetitive tasks
