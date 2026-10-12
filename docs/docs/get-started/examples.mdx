---
title: "Examples"
description: "Run a model locally, add chat to an application, process a batch of texts, or build a tool-using assistant."
---

Choose an example for what you want to do:

| I want to… | Example |
| --- | --- |
| Run a model on my own computer | [Local inference](#run-a-model-locally) on an NVIDIA GPU or Apple Silicon Mac. |
| Serve a larger model on a GPU server | [Deploy Qwen3.8-27B](#serve-qwen3-8-27b) on an NVIDIA H200. |
| Add chat to an application | [Stream a reply](#stream-a-reply) through the OpenAI-compatible API. |
| Process a collection of documents | [Summarize multiple texts](#process-multiple-prompts) concurrently. |
| Build an agent that uses tools | [Look up an order](#build-a-customer-support-agent) and answer a customer’s question. |

## Run a model locally

Run `Qwen/Qwen3-0.6B` on your own computer and ask it a question. This small model is useful for trying the workflow; choose a larger model for more demanding tasks.

<Tabs>
<Tab title="NVIDIA GPU (Linux)">

This example uses one CUDA 13-compatible NVIDIA GPU, Linux, Docker, and NVIDIA Container Toolkit. For setup, see [Installation](/docs/get-started/install).

Run this command on your computer. Docker pulls the image if needed, and SGLang downloads the model on first use:

```bash
docker run --rm --gpus all --ipc=host \
  -p 127.0.0.1:30000:30000 \
  lmsysorg/sglang:latest \
  sglang serve Qwen/Qwen3-0.6B \
    --host 0.0.0.0 --port 30000
```

If you already installed SGLang with uv, run this in that Python environment instead:

```bash
sglang serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 30000
```

</Tab>
<Tab title="Apple Silicon Mac">

On an Apple Silicon Mac with macOS 14 or newer, first follow the [Apple Silicon installation instructions](/docs/hardware-platforms/apple_metal#install-sglang) to install SGLang with its MLX dependencies. Use that Python environment to launch the model with 4-bit quantization:

```bash
SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --quantization mlx_q4 \
  --disable-cuda-graph \
  --host 127.0.0.1 --port 30000
```

The model runs on your Mac through MLX. It downloads the weights on first use and quantizes them in memory when loading.

</Tab>
</Tabs>

Keep the server running. Once you see `The server is fired up and ready to roll!`, open another terminal on the same computer and send a request:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is the capital of France? /no_think"}],
    "max_tokens": 128
  }'
```

Look for the answer in `choices[0].message.content`. Both the model and the API server are running locally. Press `Ctrl+C` in the server terminal when you are done, or leave it running for the chat and batch examples below.

<a id="serve-qwen3-8-27b" />

## Serve Qwen3.8-27B

This configuration runs `Qwen/Qwen3.8-27B-FP8` on one NVIDIA H200. It requires Linux, a CUDA 13-compatible driver, Docker, and NVIDIA Container Toolkit configured for GPU access. For environment setup, see [Installation](/docs/get-started/install). For other hardware configurations, see the [Qwen3.8-27B Cookbook](/cookbook/autoregressive/Qwen/Qwen3.8-27B).

Set `HF_CACHE_DIR` to an absolute directory on the host for downloaded model weights, replacing the placeholder below. Run this command on the GPU host:

```bash
HF_CACHE_DIR="/path/to/huggingface-cache"
docker run --rm --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  --ipc=host \
  lmsysorg/sglang:latest \
  sglang serve \
    --trust-remote-code \
    --model-path Qwen/Qwen3.8-27B-FP8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --mamba-full-memory-ratio 4.59 \
    --host 0.0.0.0 \
    --port 30000 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-ssm-dtype float32
```

Docker downloads the image if needed. SGLang downloads the model on first use and starts the server. Keep this terminal open and wait for `The server is fired up and ready to roll!` before sending a request.

<Accordion title="Already installed with uv?">

Run this command in the activated Python environment where you installed SGLang. It starts the same server without Docker:

```bash
sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --host 0.0.0.0 \
  --port 30000 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32
```

For Python environment setup, see [Install with uv](/docs/get-started/install#method-1-with-pip-or-uv).

</Accordion>

Verify the running server with the [Quickstart request](/docs/get-started/quickstart#send-a-request), using `Qwen/Qwen3.8-27B-FP8` as the model name.

## Connect to a model

The following Python examples connect to the local server above. You can also use a remote SGLang server; only the server needs hardware that can run the model.

Install the Python client in your activated environment:

```bash
uv pip install openai
```

Run this setup before each example in the same Python session:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",
)
model = "Qwen/Qwen3-0.6B"
```

These settings match the local Qwen3-0.6B server above. For a remote server, replace `localhost` with its reachable address, use its model name, and supply an API key if authentication is enabled.

## Stream a reply

Ask a question and print the answer as it arrives:

```python
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Explain prefix caching in two sentences."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

The terminal displays the model's explanation incrementally. You can use the same stream to update a chat interface.

See the [API guide](/docs/basic_usage/openai_api_completions) for more request options.

## Process multiple prompts

Send several independent requests concurrently, for example to summarize customer feedback:

```python
from concurrent.futures import ThreadPoolExecutor

feedback = [
    "Setup was quick, but I could not find the export button.",
    "The new search is faster and finds the documents I need.",
    "Please add a way to share saved reports with my team.",
]

def summarize(text):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Summarize in one sentence: {text}"}],
    )
    return response.choices[0].message.content

with ThreadPoolExecutor(max_workers=3) as pool:
    for summary in pool.map(summarize, feedback):
        print(summary)
```

You get one summary per input, in the original order. For batch processing inside a Python process without an HTTP server, see the [offline engine API](/docs/basic_usage/offline_engine_api).

<a id="request-a-tool-call" />

## Build a customer support agent

Let an agent answer “Where is my order?” by looking up order data. The model chooses a tool, your application executes it, and the model uses the returned data to answer the customer.

Use the [Qwen3.8-27B server above](#serve-qwen3-8-27b), which enables a tool parser, or configure another model with the [tool-calling guide](/docs/advanced_features/tool_parser). The small local example does not configure tool calling. This script uses the OpenAI client installed above; update the endpoint and model ID to match your server.

Save this as `order_agent.py` and run `python order_agent.py`. The order data is a local fixture, so no external service or API key is needed for the lookup.

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
model = "Qwen/Qwen3.8-27B-FP8"

orders = {
    "SGL-1042": {"status": "shipped", "estimated_delivery": "Friday"},
}

def lookup_order(order_id):
    return orders.get(order_id, {"error": "Order not found"})

tools = [{
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Look up an order's shipping status and estimated delivery.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}]

messages = [
    {"role": "system", "content": "Use lookup_order for order status. Do not invent order details."},
    {"role": "user", "content": "Where is order SGL-1042, and when should it arrive?"},
]

for _ in range(3):
    response = client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto",
    )
    message = response.choices[0].message
    messages.append(message.model_dump(exclude_none=True))
    if not message.tool_calls:
        print(message.content)
        break

    for call in message.tool_calls:
        try:
            if call.function.name != "lookup_order":
                raise ValueError("Unknown tool")
            arguments = json.loads(call.function.arguments)
            result = lookup_order(**arguments)
        except (ValueError, TypeError) as error:
            result = {"error": str(error)}
        print(f"{call.function.name}: {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result),
        })
else:
    print("Stopped after three model turns without a final answer.")
```

You should see the lookup result followed by an answer that the order has shipped and is expected on Friday. The wording can vary. SGLang serves the model; the Python application runs the tools and manages the conversation. Replace `lookup_order` with an authorized order-system query to use real data.
