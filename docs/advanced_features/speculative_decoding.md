# Speculative Decoding

SGLang provides several speculative decoding options, including EAGLE-2/EAGLE-3, MTP, classic draft-model decoding, and an NGRAM-based variant. Our implementation aims to maximize speed and efficiency and is considered to be among the fastest in open-source LLM engines.

## Summary

### Jump to sections

- [EAGLE Decoding](#eagle-decoding)
  - [EAGLE-2 Decoding](#eagle-2-decoding)
  - [EAGLE-2 Decoding with torch.compile](#eagle-2-decoding-with-torchcompile)
  - [EAGLE-2 Decoding via Frequency-Ranked Speculative Sampling](#eagle-2-decoding-via-frequency-ranked-speculative-sampling)
  - [EAGLE-3 Decoding](#eagle-3-decoding)
- [Multi Token Prediction](#multi-token-prediction)
- [Standalone Speculative Decoding (Small Draft Model)](#standalone-speculative-decoding-small-draft-model)
- [Speculative Decoding V2 (Overlap Scheduler)](#speculative-decoding-v2-overlap-scheduler)
- [Ngram Speculative Decoding](#ngram-speculative-decoding)
- [Full Parameter Reference](#full-parameter-reference)
- [OOM Troubleshooting](#oom-troubleshooting)
- [References](#references)

### Quick guidance

- **Best speed/quality (recommended)**: Use **EAGLE-3** with `--speculative-algorithm EAGLE3`.
- **Strong default / broad compatibility**: Use **EAGLE-2** with `--speculative-algorithm EAGLE`.
- **Lower `lm_head` overhead for EAGLE-2**: Enable **FR-Spec** with `--speculative-token-map`.
- **Model is MTP-enabled**: Use **MTP via speculative decoding** (often with small `speculative_num_steps/topk/num_draft_tokens`, see the example section).
- **You have a smaller draft LLM**: Use **STANDALONE** (`--speculative-algorithm STANDALONE`).
- **No extra model available**: Use **NGRAM** (`--speculative-algorithm NGRAM`, CUDA-only).
- **Want overlap scheduler (experimental)**: Enable **SpecV2** with `SGLANG_ENABLE_SPEC_V2=True` (requires `--speculative-eagle-topk 1`).

### Method comparison (mini table)

| Method | Draft source | Separate draft model? | How to enable | Notes / constraints |
|---|---|---:|---|---|
| EAGLE-2 | EAGLE draft model (feature drafting + tree) | Typically yes | `--speculative-algorithm EAGLE` + `--speculative-draft-model-path ...` | Tune `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens` |
| EAGLE-2 + `torch.compile` | Same as EAGLE-2 | Typically yes | Add `--enable-torch-compile` (optionally `--torch-compile-max-bs`) | Further kernel-level optimizations |
| EAGLE-2 + FR-Spec | Same as EAGLE-2 + token subset | Typically yes | Add `--speculative-token-map ...` | Reduces `lm_head` overhead with high-frequency token vocab |
| EAGLE-3 | EAGLE3 draft model | Yes | `--speculative-algorithm EAGLE3` + `--speculative-draft-model-path ...` | Best throughput in the benchmark above |
| MTP | Built-in multi-token heads (model-specific) | Often no | See **Multi Token Prediction** section | Uses speculative workflow; draft path may be auto-handled for some models |
| STANDALONE | Smaller draft LLM (token-level) | Yes | `--speculative-algorithm STANDALONE` + `--speculative-draft-model-path ...` | Does **not** support `--enable-dp-attention` |
| SpecV2 (experimental) | V2 workers + overlap scheduler | N/A | `SGLANG_ENABLE_SPEC_V2=True` | Only supports `--speculative-eagle-topk 1`; applies to `EAGLE`, `EAGLE3`, `STANDALONE` |
| NGRAM | Ngram cache from previous tokens | No | `--speculative-algorithm NGRAM` | CUDA-only; no `--enable-dp-attention`; disables overlap scheduler & mixed chunked prefill |

### Performance Highlights

Please see below for the huge improvements on throughput for LLaMA-Instruct 3.1 8B tested on MT bench that can be achieved via EAGLE3 decoding.
For further details please see the [EAGLE3 paper](https://arxiv.org/pdf/2503.01840).

| Method | Throughput (tokens/s) |
|--------|----------------|
| SGLang (w/o speculative, 1x H100) | 158.34 tokens/s |
| SGLang + EAGLE-2 (1x H100) | 244.10 tokens/s |
| SGLang + EAGLE-3 (1x H100) | 373.25 tokens/s |

---

## EAGLE Decoding

To enable EAGLE speculative decoding the following parameters are relevant:

| Parameter | Description | Default |
|---|---|---|
| `--speculative-draft-model-path` | Draft model path/weights. **Typically required** for EAGLE/EAGLE3 and STANDALONE. For some MTP-enabled models, this can be omitted. | `None` |
| `--speculative-num-steps` | Depth of autoregressive drafting. Increases speculation range but risks rejection cascades. | Auto (`5` for Llama/Grok; `3` for many other models) |
| `--speculative-eagle-topk` | Branching factor per step. Improves candidate diversity and acceptance rate, but increases memory/compute consumption. | Auto (`4` for Llama/Grok; `1` for many other models) |
| `--speculative-num-draft-tokens` | Maximum parallel verification capacity. Allows deeper tree evaluation but increases GPU memory usage. | Auto (`8` for Llama/Grok; `4` for many other models). If `topk=1`, it is adjusted to `num_steps + 1`. |
| `--speculative-accept-threshold-single` | Acceptance threshold for single-token verification. Lower values accept more aggressively. | `1.0` |
| `--speculative-accept-threshold-acc` | Accumulated acceptance threshold across steps. | `1.0` |
| `--speculative-attention-mode` | Attention mode for speculative operations (`prefill` or `decode`), affecting both target verification and draft extension. | `"prefill"` |
| `--speculative-draft-attention-backend` | Override attention backend for the draft model. | `None` (same as target) |
| `--speculative-draft-model-quantization` | Quantization method for the draft model. Use `"unquant"` to force no quantization even when the target model is quantized. | Same as target model |
| `--speculative-draft-model-revision` | Specific revision/commit of the draft model to load. | `None` (auto-set to `"main"` when `--speculative-draft-model-path` is set and revision is omitted) |
| `--speculative-draft-load-format` | Load format for the draft model weights. | `None` |

These parameters are mostly the same for EAGLE-2 and EAGLE-3. `--speculative-token-map` is ignored for EAGLE-3 models.
For `--speculative-num-steps`, `--speculative-eagle-topk`, and `--speculative-num-draft-tokens`: leave all three unset to use auto-tuning, or set all three explicitly when tuning.

You can find the best combinations of these parameters with [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py).


### EAGLE-2 decoding

You can enable EAGLE-2 decoding by setting `--speculative-algorithm EAGLE` and choosing an appropriate model.

**Launch the server:**

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

### EAGLE-2 Decoding with `torch.compile`

You can also enable `torch.compile` for further optimizations and optionally set `--torch-compile-max-bs`:

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --enable-torch-compile \
    --torch-compile-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

### EAGLE-2 Decoding via Frequency-Ranked Speculative Sampling

By employing a truncated high-frequency token vocabulary in the draft model, Eagle speculative decoding reduces `lm_head` computational overhead while accelerating the pipeline without quality degradation. For more details, checkout [the paper](https://arxiv.org/pdf/2502.14856).

In our implementation, set `--speculative-token-map` to enable the optimization. You can get the high-frequency token in FR-Spec from [this model](https://huggingface.co/thunlp/LLaMA3-Instruct-8B-FR-Spec). Or you can obtain high-frequency token by directly downloading these token from [this repo](https://github.com/thunlp/FR-Spec/tree/main?tab=readme-ov-file#prepare-fr-spec-vocabulary-subset).

Thanks for the contribution from [Weilin Zhao](https://github.com/Achazwl) and [Zhousx](https://github.com/Zhou-sx).

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-LLaMA3-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --dtype float16 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

### EAGLE-3 Decoding

You can enable EAGLE-3 decoding by setting `--speculative-algorithm EAGLE3` and choosing an appropriate model.

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --dtype float16 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## Multi Token Prediction

We support [MTP(Multi-Token Prediction)](https://arxiv.org/pdf/2404.19737) in SGLang by using speculative decoding. We use `XiaomiMiMo/MiMo-7B-RL` as an example here (for DeepSeek MTP usage, refer to [deepseek_v32 doc](../basic_usage/deepseek_v32.md#multi-token-prediction)).

```bash
python3 -m sglang.launch_server \
    --model XiaomiMiMo/MiMo-7B-RL \
    --host 0.0.0.0 \
    --trust-remote-code \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "XiaomiMiMo/MiMo-7B-RL",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

response = requests.post(url, json=data)
print(response.json())
```

---

## Standalone Speculative Decoding (Small Draft Model)

Besides EAGLE/MTP, SGLang also supports **token-level speculative decoding** using a smaller **draft model**. Enable it with `--speculative-algorithm STANDALONE` and provide a draft model via `--speculative-draft-model-path`.

Relevant parameters:

| Parameter | Description | Default |
|---|---|---|
| `--speculative-draft-model-path` | Draft model weights (smaller than the target model). | `None` |
| `--speculative-num-steps` | Draft depth (how many steps the draft model runs autoregressively). | `3` (auto default for STANDALONE) |
| `--speculative-eagle-topk` | Branching factor (token candidates per step). | `1` (auto default for STANDALONE) |
| `--speculative-num-draft-tokens` | Verification capacity. | `4` (auto default for STANDALONE) |
| `--speculative-draft-model-quantization` | Quantization for the draft model. Use `"unquant"` to disable quantization on the draft even when the target is quantized. | Same as target |

> **Note:** Standalone speculative decoding currently **does not support** `--enable-dp-attention`.

```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 7 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## Speculative Decoding V2 (Overlap Scheduler)

SGLang provides an **experimental Speculative Decoding V2** implementation that enables an overlap scheduler and uses V2 speculative workers (e.g. `StandaloneWorkerV2`, `EAGLEWorkerV2`).

To enable it, set the environment variable:
- `SGLANG_ENABLE_SPEC_V2=True`

Notes:
- SpecV2 currently only supports `--speculative-eagle-topk 1`. When SpecV2 is enabled, **set `--speculative-eagle-topk 1` explicitly**.
- If you explicitly set `--speculative-eagle-topk > 1`, the server will error.
- If you omit `--speculative-eagle-topk`, auto-tuning may pick `topk > 1` for some models (e.g. Llama). This is incompatible with SpecV2 and may not always trigger an immediate config error, so set `--speculative-eagle-topk 1` explicitly.
- This applies to `EAGLE`, `EAGLE3`, and `STANDALONE`.

```bash
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## Ngram Speculative Decoding

SGLang also supports **ngram-based speculative decoding** (no separate draft model). It retrieves draft tokens from an ngram cache built from previously generated tokens, and then verifies them with the target model.

Enable it with:
- `--speculative-algorithm NGRAM`

### Ngram-specific parameters

| Parameter | Description | Default |
|---|---|---|
| `--speculative-num-draft-tokens` | Number of draft tokens verified per step. If omitted, defaults to `--speculative-ngram-max-match-window-size`. | `12` (with default ngram settings) |
| `--speculative-ngram-min-match-window-size` | Minimum matching window size. | `1` |
| `--speculative-ngram-max-match-window-size` | Maximum matching window size. | `12` |
| `--speculative-ngram-min-bfs-breadth` | Minimum BFS breadth. | `1` |
| `--speculative-ngram-max-bfs-breadth` | Maximum BFS breadth. | `10` |
| `--speculative-ngram-match-type` | Match type: `"BFS"` or `"PROB"`. | `"BFS"` |
| `--speculative-ngram-branch-length` | How many recent tokens to insert into the cache. | `18` |
| `--speculative-ngram-capacity` | Cache capacity (number of entries). | `10,000,000` |

Notes:
- Ngram speculative decoding **only supports CUDA**.
- It currently **does not support** `--enable-dp-attention`.
- It disables the overlap scheduler and mixed chunked prefill.
- If `--speculative-ngram-max-bfs-breadth > 1` (thus `speculative_eagle_topk > 1`) and `page_size > 1`, use `--attention-backend flashinfer`; otherwise the server will error.
- Optional: set `SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True` to force greedy verification.

```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 16 \
    --speculative-ngram-max-match-window-size 12 \
    --speculative-ngram-max-bfs-breadth 10 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

**Send a request:**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## Full Parameter Reference

Below is a comprehensive list of all speculative decoding parameters available in SGLang:

### Core parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--speculative-algorithm` | `str` | `None` | Algorithm to use: `EAGLE`, `EAGLE3`, `STANDALONE`, `NGRAM`, `NEXTN` (alias of `EAGLE`) |
| `--speculative-draft-model-path` | `str` | `None` | Path to the draft model weights |
| `--speculative-draft-model-revision` | `str` | `None` | Specific revision/commit of the draft model (`"main"` is auto-used when draft path is set and revision is omitted) |
| `--speculative-draft-load-format` | `str` | `None` | Load format for draft model weights |
| `--speculative-num-steps` | `int` | `None` (auto-chosen when omitted) | Autoregressive drafting depth |
| `--speculative-eagle-topk` | `int` | `None` (auto-chosen when omitted) | Branching factor per drafting step |
| `--speculative-num-draft-tokens` | `int` | `None` (auto-chosen when omitted) | Maximum number of draft tokens for verification |
| `--speculative-accept-threshold-single` | `float` | `1.0` | Single-token acceptance threshold |
| `--speculative-accept-threshold-acc` | `float` | `1.0` | Accumulated acceptance threshold |
| `--speculative-token-map` | `str` | `None` | Path to FR-Spec high-frequency token map |
| `--speculative-attention-mode` | `str` | `"prefill"` | Attention mode for speculative operations (`"prefill"` or `"decode"`) |
| `--speculative-draft-attention-backend` | `str` | `None` | Override attention backend for the draft model |
| `--speculative-moe-runner-backend` | `str` | `None` | MoE runner backend for the draft model |
| `--speculative-moe-a2a-backend` | `str` | `None` | MoE all-to-all backend for the draft model |
| `--speculative-draft-model-quantization` | `str` | Same as target | Quantization for the draft model (`"unquant"` to disable) |

### Ngram-specific parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--speculative-ngram-min-match-window-size` | `int` | `1` | Minimum ngram matching window |
| `--speculative-ngram-max-match-window-size` | `int` | `12` | Maximum ngram matching window |
| `--speculative-ngram-min-bfs-breadth` | `int` | `1` | Minimum BFS breadth |
| `--speculative-ngram-max-bfs-breadth` | `int` | `10` | Maximum BFS breadth |
| `--speculative-ngram-match-type` | `str` | `"BFS"` | Match type: `"BFS"` or `"PROB"` |
| `--speculative-ngram-branch-length` | `int` | `18` | Recent tokens to insert into cache |
| `--speculative-ngram-capacity` | `int` | `10,000,000` | Cache capacity |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `SGLANG_ENABLE_SPEC_V2` | `False` | Enable Speculative Decoding V2 (overlap scheduler) |
| `SGLANG_NGRAM_FORCE_GREEDY_VERIFY` | `False` | Force greedy verification for ngram decoding |

### Other related flags

| Parameter | Description |
|---|---|
| `--enable-multi-layer-eagle` | Enable multi-layer EAGLE (auto-enabled for MiMoV2 and Step3p5 models) |
| `--enable-torch-compile` | Enable `torch.compile` for kernel-level optimizations |
| `--torch-compile-max-bs` | Maximum batch size for `torch.compile` |

---

## OOM Troubleshooting

> [!WARNING]
> **Out of Memory (OOM)?** Speculative decoding may increase GPU memory usage because the draft tree, CUDA graphs, and verification-related buffers consume additional VRAM. If you encounter OOM errors, try the following adjustments.

### Step 1: Reduce draft tree size (most effective)

These three parameters directly control how much memory the draft tree consumes:

```bash
# Before (aggressive, high memory)
--speculative-num-steps 5 --speculative-eagle-topk 8 --speculative-num-draft-tokens 64

# After (conservative, lower memory)
--speculative-num-steps 3 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16
```

- **`--speculative-num-draft-tokens`**: This is the single most impactful parameter. Reducing from 64 → 16 can cut draft-related memory by ~75%. Start here.
- **`--speculative-eagle-topk`**: Reducing from 8 → 4 or even 2 halves the branching factor.
- **`--speculative-num-steps`**: Reducing from 5 → 3 shortens the draft depth.

### Step 2: Lower static memory fraction

```bash
# Give more room for dynamic allocations (CUDA graphs, draft model, etc.)
--mem-fraction-static 0.5   # when omitted, this value is auto-computed
```

### Step 3: Reduce CUDA graph batch size

```bash
# Fewer CUDA graph captures = less memory reserved
--cuda-graph-max-bs 4   # or even 2 for tight memory situations
```

### Step 4: Limit concurrent requests

```bash
# Fewer concurrent requests lowers in-flight load and can reduce OOM risk
--max-running-requests 4
```

### Step 5: Use quantization

```bash
# Quantize the target model (if supported by your checkpoint/hardware)
--quantization fp8

# Or quantize only the draft model (keep target at full precision)
--speculative-draft-model-quantization fp8
```

### Step 6: Use a smaller dtype

```bash
--dtype float16   # instead of bfloat16/float32 (when supported)
```

### Step 7: Use FR-Spec to reduce lm_head memory (EAGLE-2 / STANDALONE)

```bash
--speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt
```
> Note: For EAGLE-3, `--speculative-token-map` is ignored because EAGLE-3 models already provide built-in hot-token handling.

### Quick OOM recovery recipe

If you're hitting OOM and just want something that works, start with this minimal configuration and scale up:

```bash
python3 -m sglang.launch_server \
    --model <your-model> \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path <your-draft-model> \
    --speculative-num-steps 2 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 8 \
    --cuda-graph-max-bs 2 \
    --mem-fraction-static 0.5 \
    --max-running-requests 4 \
    --dtype float16 \
    --log-level warning
```

Then gradually increase `--speculative-num-draft-tokens`, `--speculative-eagle-topk`, and `--cuda-graph-max-bs` until you find the sweet spot for your GPU.

> [!TIP]
> **Memory budget rule of thumb**: during automatic `--mem-fraction-static` estimation, STANDALONE reserves about 6 GB and EAGLE/EAGLE3 reserves about 2 GB as additional headroom. Plan your `--mem-fraction-static` accordingly.

---

## References

EAGLE process is as follows:

- Within EAGLE the draft model predicts the next feature vector, i.e. the last hidden state of the original LLM, using the feature sequence $(f_1, ..., f_k)$ and the token sequence $(t_2, ..., t_{k+1})$.
- The next token is then sampled from $p_{k+2}=\text{LMHead}(f_{k+1})$. Afterwards, the two sequences are extended in a tree style—branching out multiple potential continuations, with the branching factor per step controlled by the `speculative_eagle_topk` parameter—to ensure a more coherent connection of context, and are given as input again.
- In SGLang's EAGLE-2 implementation, the draft tree is expanded for the configured steps and then reranked to select the top `speculative_num_draft_tokens` final nodes as draft tokens.
- EAGLE-3 removes the feature prediction objective, incorporates low and mid-layer features, and is trained in an on-policy manner.

This enhances drafting accuracy by operating on features instead of tokens for more regular inputs and by additionally passing tokens from the next timestep to reduce sampling randomness. For more details, see the [EAGLE-2](https://arxiv.org/abs/2406.16858) and [EAGLE-3](https://arxiv.org/abs/2503.01840) papers.

For guidance how to train your own EAGLE model please see the [EAGLE repo](https://github.com/SafeAILab/EAGLE/tree/main?tab=readme-ov-file#train).
