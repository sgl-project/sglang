# Deterministic Inference

## Why Deterministic Inference Matters

Deterministic inference ensures consistent LLM outputs across runs, which is critical for:
- **Reinforcement Learning**: Ensures consistent logprobs across runs, reducing stochastic noise and making RL training more stable, reproducible, and debuggable.
- **Testing & Debugging**: Enables reproducible validation
- **Production**: Improves reliability and user experience

Even with `temperature=0`, standard LLM inference can produce different outputs due to dynamic batching and varying reduction orders in GPU kernels.

## The Root Cause of Non-Determinism

The main source is **varying batch sizes**. Different batch sizes cause GPU kernels to split reduction operations differently, leading to different addition orders. Due to floating-point non-associativity (`(a + b) + c ≠ a + (b + c)`), this produces different results even for identical inputs.


## SGLang's Solution

Building on [Thinking Machines Lab's batch-invariant operators](https://github.com/thinking-machines-lab/batch_invariant_ops), SGLang achieves fully deterministic inference while maintaining compatibility with chunked prefill, CUDA graphs, radix cache, and non-greedy sampling. With CUDA graphs, SGLang delivers 2.8x acceleration and reduces performance overhead to just 34.35% (vs. TML's 61.5%). 

### Supported Backends

Deterministic inference is supported on the following attention backends:
- **FlashInfer**
- **FlashAttention 3 (FA3)**
- **Triton**

## Usage

### Basic Usage

Enable deterministic inference by adding the `--enable-deterministic-inference` flag:

```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### Server Arguments

| Argument | Type/Default | Description |
|----------|--------------|-------------|
| `--enable-deterministic-inference` | flag; default: disabled | Enable deterministic inference with batch-invariant operations |
| `--attention-backend` | string; default: fa3 | Choose attention backend (flashinfer, fa3, or triton) |

### Example Configurations

#### Qwen3-8B
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference
```

#### Llama Models
```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

#### Qwen3-30B-A3B (MoE Model)
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```


## Verification

Run deterministic tests to verify consistent outputs:

```bash
# Single test: same prompt, varying batch sizes
python3 -m sglang.test.test_deterministic --test-mode single --n-trials 50

# Prefix test: prompts with different prefix lengths
python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 50
```

Expected result: All tests should show `Unique samples: 1` (perfectly deterministic).
