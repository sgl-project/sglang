# Hy3-preview Usage

Hy3-preview is a large-scale language model (295B parameters, 21B active parameters) from Tencent Hunyuan team. SGLang supports serving Hy3-preview. This guide describes how to run Hy3-preview with native BF16.

## Installation

### Docker

```bash
docker pull lmsysorg/sglang:hy3-preview
```

### Build from Source

```bash
# Install SGLang
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install "transformers>=5.6.0"
pip3 install -e "python"
```

## Launch Hy3-preview with SGLang

To serve the [Hy3-preview](https://huggingface.co/tencent/Hy3-preview) model on 8 GPUs. On 8x96GB H20, SGLang can barely deploy the BF16 model and can only run small batch sizes or short requests. Use larger-memory GPUs such as H20-3e when possible.

```bash
python3 -m sglang.launch_server \
  --model tencent/Hy3-preview \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --served-model-name hy3-preview
```

### EAGLE Speculative Decoding

**Description**: SGLang supports Hy3-preview models with [EAGLE speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html#eagle-decoding).

**Usage**:
Add `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-eagle-topk`, and `--speculative-num-draft-tokens` to enable this feature. For example:

```bash
python3 -m sglang.launch_server \
  --model tencent/Hy3-preview \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --speculative-algorithm EAGLE \
  --served-model-name hy3-preview
```

## OpenAI Client Example

First, install the OpenAI Python client:

```bash
uv pip install -U openai
```

You can use the OpenAI client as follows to verify thinking-mode responses.

```python
from openai import OpenAI

# If running SGLang locally with its default OpenAI-compatible port:
#   http://localhost:30000/v1
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:30000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello."},
]

# Thinking mode is disabled by default (no need to pass chat_template_kwargs).
resp = client.chat.completions.create(
    model="hy3-preview",
    messages=messages,
    temperature=1,
    max_tokens=4096,
)
print(resp.choices[0].message.content)

# Thinking mode is enabled only if 'reasoning_effort' and 'interleaved_thinking' are set in 'chat_template_kwargs'.
# 'reasoning_effort' supports: 'high', 'low', 'no_think'.
resp_think = client.chat.completions.create(
    model="hy3-preview",
    messages=messages,
    temperature=1,
    max_tokens=4096,
    extra_body={
      "chat_template_kwargs": {
          "reasoning_effort": "high",
          "interleaved_thinking": True
      },
    },
)
output_msg = resp_think.choices[0].message
# thinking content
print(output_msg.reasoning_content)
# response content
print(output_msg.content)
```

### cURL Usage

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hy3-preview",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello."}
    ],
    "temperature": 1,
    "max_tokens": 4096
  }'
```

## Benchmarking Results

For benchmarking, disable prefix caching by adding `--disable-radix-cache` to the server command.

The following example runs the benchmark on 8 H20 GPUs with 96 GB memory each.

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --flush-cache \
    --dataset-name random \
    --random-range-ratio 1.0 \
    --random-input-len 4096 \
    --random-output-len 4096 \
    --num-prompts 5 \
    --max-concurrency 1 \
    --output-file hy3_preview_h20.jsonl \
    --model tencent/Hy3-preview \
    --served-model-name hy3-preview
```

If successful, you will see the following output.

```shell
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     5
Benchmark duration (s):                  176.41
Total input tokens:                      20480
Total input text tokens:                 20480
Total generated tokens:                  20480
Total generated tokens (retokenized):    20480
Request throughput (req/s):              0.03
Input token throughput (tok/s):          116.09
Output token throughput (tok/s):         116.09
Peak output token throughput (tok/s):    118.00
Peak concurrent requests:                2
Total token throughput (tok/s):          232.19
Concurrency:                             1.00
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   35279.06
Median E2E Latency (ms):                 35275.60
P90 E2E Latency (ms):                    35294.13
P99 E2E Latency (ms):                    35294.41
---------------Time to First Token----------------
Mean TTFT (ms):                          355.93
Median TTFT (ms):                        309.28
P99 TTFT (ms):                           518.36
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.53
Median TPOT (ms):                        8.54
P99 TPOT (ms):                           8.54
---------------Inter-Token Latency----------------
Mean ITL (ms):                           8.53
Median ITL (ms):                         8.54
P95 ITL (ms):                            8.62
P99 ITL (ms):                            8.74
Max ITL (ms):                            31.70
==================================================
```
