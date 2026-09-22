---
title: "Deterministic Inference"
metatags:
    description: "SGLang deterministic inference: consistent outputs for RL training, testing, and production. Supports FlashInfer, FA3, Triton backends with CUDA Graph."
---
## Why Deterministic Inference Matters

Deterministic inference ensures consistent LLM outputs across runs, which is critical for:
- **Reinforcement Learning**: Ensures consistent logprobs across runs, reducing stochastic noise and making RL training more stable, reproducible, and debuggable.
- **Testing & Debugging**: Enables reproducible validation
- **Production**: Improves reliability and user experience

Even with `temperature=0`, standard LLM inference can produce different outputs due to dynamic batching and varying reduction orders in GPU kernels.

## The Root Cause of Non-Determinism

The main source is **varying batch sizes**. Different batch sizes cause GPU kernels to split reduction operations differently, leading to different addition orders. Due to floating-point non-associativity (`(a + b) + c ≠ a + (b + c)`), this produces different results even for identical inputs.


## SGLang's Solution

Building on [Thinking Machines Lab's batch-invariant operators](https://github.com/thinking-machines-lab/batch_invariant_ops), SGLang achieves fully deterministic inference while maintaining compatibility with chunked prefill, CUDA graphs, radix cache, and non-greedy sampling. The development roadmap for deterministic inference features can be found in this [issue](https://github.com/sgl-project/sglang/issues/10278).

### Supported Backends

Deterministic inference is only supported with the following three attention backends: **FlashInfer**, **FlashAttention 3 (FA3)**, and **Triton**.

The following table shows feature compatibility for deterministic inference across different attention backends:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "20%"}} />
    <col style={{width: "20%"}} />
    <col style={{width: "20%"}} />
    <col style={{width: "20%"}} />
    <col style={{width: "20%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Attention Backend</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>CUDA Graph</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Chunked Prefill</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Radix Cache</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Non-greedy Sampling (Temp > 0)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>**FlashInfer**</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>✅ Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>❌ No</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Yes</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>**FlashAttention 3 (FA3)**</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>✅ Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>✅ Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Yes</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>**Triton**</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>✅ Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>✅ Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>✅ Yes</td>
    </tr>
  </tbody>
</table>

## Usage

### Basic Usage

Enable deterministic inference by adding the `--enable-deterministic-inference` flag:

```bash Command
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### Server Arguments

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "34%"}} />
    <col style={{width: "33%"}} />
    <col style={{width: "33%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Argument</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Type/Default</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--enable-deterministic-inference`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>flag; default: disabled</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Enable deterministic inference with batch-invariant operations</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`--attention-backend`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>string; default: fa3</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Choose attention backend (flashinfer, fa3, or triton)</td>
    </tr>
  </tbody>
</table>

### Example Configurations

#### Qwen3-8B
```bash Command
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --attention-backend flashinfer \
    --enable-deterministic-inference
```

#### Llama Models
```bash Command
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

#### Qwen3-30B-A3B (MoE Model)
```bash Command
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --attention-backend fa3 \
    --enable-deterministic-inference
```

### Deterministic Inference with Non-Greedy Sampling (Temperature > 0)

SGLang supports deterministic inference even with non-greedy sampling by using sampling seeds. This is particularly useful for reinforcement learning scenarios like GRPO (Group Relative Policy Optimization) where you need multiple diverse but reproducible responses.

#### Default Behavior

By default, SGLang uses a sampling seed of `42` for reproducible sampling:

```python Example
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "Tell me a joke",
        "sampling_params": {
            "temperature": 0.8,  # Non-greedy sampling
            "max_new_tokens": 128,
        },
    },
)
print(response.json())
# This will always produce the same response across runs
```

#### Generating Multiple Reproducible Responses

To sample different responses from the same prompt while maintaining reproducibility (e.g., for GRPO training), provide different sampling seeds in your requests:

```python Example
import requests

# Prepare a list of sampling seeds for different responses
sampling_seeds = [42, 43, 44, 45, 46]

responses = []
for seed in sampling_seeds:
    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": "Tell me a joke",
            "sampling_params": {
                "temperature": 0.8,
                "max_new_tokens": 128,
                "sampling_seed": seed,  # Specify sampling seed
            },
        },
    )
    responses.append(response.json())

# Each seed will produce a different but reproducible response
# Using the same seed will always produce the same response
```

This approach ensures that:
- Different seeds produce diverse responses
- The same seed always produces the same response across different runs
- Results are reproducible for debugging and evaluation


## Verification

Run deterministic tests to verify consistent outputs:

```bash Command
# Single test: same prompt, varying batch sizes
python3 -m sglang.test.test_deterministic --test-mode single --n-trials 50

# Prefix test: prompts with different prefix lengths
python3 -m sglang.test.test_deterministic --test-mode prefix --n-trials 50

# Radix Cache Consistency mode: test radix cache determinism (cached vs uncached prefill)
python3 -m sglang.test.test_deterministic --test-mode radix_cache
```

Expected result: All tests should show `Unique samples: 1` (perfectly deterministic).
