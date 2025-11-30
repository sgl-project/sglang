![SGLang worker banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-sglang_banner-A9R2vQzvSUmLvqMZ8MzehfZtRDxHJR.jpeg)

Run LLMs and VLMs using [SGLang](https://docs.sglang.ai)

---

[![RunPod](https://api.runpod.io/badge/sgl-project/sglang)](https://www.runpod.io/console/hub/sgl-project/sglang)

---

## Endpoint Configuration

All behaviour is controlled through environment variables:

| Environment Variable              | Description                                    | Default                               | Options                                                                                   |
| --------------------------------- | ---------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------- |
| `MODEL_PATH`                      | Path of the model weights                      | "meta-llama/Meta-Llama-3-8B-Instruct" | Local folder or Hugging Face repo ID                                                      |
| `TOKENIZER_PATH`                  | Path of the tokenizer                          |                                       |                                                                                           |
| `TOKENIZER_MODE`                  | Tokenizer mode                                 | "auto"                                | "auto", "slow"                                                                            |
| `LOAD_FORMAT`                     | Format of model weights to load                | "auto"                                | "auto", "pt", "safetensors", "npcache", "dummy"                                           |
| `DTYPE`                           | Data type for weights and activations          | "auto"                                | "auto", "half", "float16", "bfloat16", "float", "float32"                                 |
| `CONTEXT_LENGTH`                  | Model's maximum context length                 |                                       |                                                                                           |
| `QUANTIZATION`                    | Quantization method                            |                                       | "awq", "fp8", "gptq", "marlin", "gptq_marlin", "awq_marlin", "squeezellm", "bitsandbytes" |
| `SERVED_MODEL_NAME`               | Override model name in API                     |                                       |                                                                                           |
| `CHAT_TEMPLATE`                   | Chat template name or path                     |                                       |                                                                                           |
| `MEM_FRACTION_STATIC`             | Fraction of memory for static allocation       |                                       |                                                                                           |
| `MAX_RUNNING_REQUESTS`            | Maximum number of running requests             |                                       |                                                                                           |
| `MAX_TOTAL_TOKENS`                | Maximum tokens in memory pool                  |                                       |                                                                                           |
| `CHUNKED_PREFILL_SIZE`            | Max tokens in chunk for chunked prefill        |                                       |                                                                                           |
| `MAX_PREFILL_TOKENS`              | Max tokens in prefill batch                    | 16384                                 |                                                                                           |
| `SCHEDULE_POLICY`                 | Request scheduling policy                      | "fcfs"                                | "lpm", "random", "fcfs", "dfs-weight"                                                     |
| `SCHEDULE_CONSERVATIVENESS`       | Conservativeness of schedule policy            | 1.0                                   |                                                                                           |
| `TENSOR_PARALLEL_SIZE`            | Tensor parallelism size                        | 1                                     |                                                                                           |
| `STREAM_INTERVAL`                 | Streaming interval in token length             | 1                                     |                                                                                           |
| `RANDOM_SEED`                     | Random seed                                    |                                       |                                                                                           |
| `LOG_LEVEL`                       | Logging level for all loggers                  | "info"                                |                                                                                           |
| `LOG_LEVEL_HTTP`                  | Logging level for HTTP server                  |                                       |                                                                                           |
| `API_KEY`                         | API key for the server                         |                                       |                                                                                           |
| `FILE_STORAGE_PATH`               | Directory for storing uploaded/generated files | "sglang_storage"                      |                                                                                           |
| `DATA_PARALLEL_SIZE`              | Data parallelism size                          | 1                                     |                                                                                           |
| `LOAD_BALANCE_METHOD`             | Load balancing strategy                        | "round_robin"                         | "round_robin", "shortest_queue"                                                           |
| `SKIP_TOKENIZER_INIT`             | Skip tokenizer init                            | false                                 | boolean (true or false)                                                                   |
| `TRUST_REMOTE_CODE`               | Allow custom models from Hub                   | false                                 | boolean (true or false)                                                                   |
| `LOG_REQUESTS`                    | Log inputs and outputs of requests             | false                                 | boolean (true or false)                                                                   |
| `SHOW_TIME_COST`                  | Show time cost of custom marks                 | false                                 | boolean (true or false)                                                                   |
| `DISABLE_RADIX_CACHE`             | Disable RadixAttention for prefix caching      | false                                 | boolean (true or false)                                                                   |
| `DISABLE_CUDA_GRAPH`              | Disable CUDA Graph                             | false                                 | boolean (true or false)                                                                   |
| `DISABLE_OUTLINES_DISK_CACHE`     | Disable disk cache for Outlines grammar        | false                                 | boolean (true or false)                                                                   |
| `ENABLE_TORCH_COMPILE`            | Optimize model with torch.compile              | false                                 | boolean (true or false)                                                                   |
| `ENABLE_P2P_CHECK`                | Enable P2P check for GPU access                | false                                 | boolean (true or false)                                                                   |
| `ENABLE_FLASHINFER_MLA`           | Enable FlashInfer MLA optimization             | false                                 | boolean (true or false)                                                                   |
| `TRITON_ATTENTION_REDUCE_IN_FP32` | Cast Triton attention reduce op to FP32        | false                                 | boolean (true or false)                                                                   |

## API Specification

Two flavours, one schema.

- **OpenAI-compatible** – drop-in replacement for the API from OpenAI, which makes it possible to point any OpenAI-aware client at
  `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1` as the `baseUrl` and use your [RunPod API key](https://docs.runpod.io/get-started/api-keys) as the `apiKey`.
- **Standard RunPod** – call `/run` (async) or `/runsync` (blocking) with a JSON body under the
  `input` key. Base URL `https://api.runpod.ai/v2/<ENDPOINT_ID>`.

Except for transport (path + wrapper) the JSON you send/receive is identical. The tables below
describe the shared payload.

### List Models

| Flavour  | Method | Path         | Body                                            |
| -------- | ------ | ------------ | ----------------------------------------------- |
| OpenAI   | `GET`  | `/v1/models` | –                                               |
| Standard | `POST` | `/runsync`   | `{ "input": { "openai_route": "/v1/models" } }` |

#### Response

```jsonc
{
  "data": [{ "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "stats": {} }]
}
```

---

### Generate / Chat Completions

For text generation you may call the native `/generate` endpoint or the OpenAI chat-completion
endpoint. The handler translates between them so the fields are shared.

| Field             | Type                | Required | Notes                                                |
| ----------------- | ------------------- | -------- | ---------------------------------------------------- |
| `model`           | string              | ✔️       | Model id (defaults to `MODEL_PATH` when omitted).    |
| `messages`        | array&lt;object&gt; | ✔️\*     | OpenAI chat format. Required for the OpenAI flavour. |
| `text`            | string / array      | ✔️\*     | Raw prompt(s) for the native `/generate` flavour.    |
| `sampling_params` | object              | optional | max_new_tokens, temperature, top_p, etc.             |
| `stream`          | bool                | optional | `true` to receive Server-Sent Events.                |

_Provide either `messages` **or** `text` depending on the flavour you use._

| Flavour  | Method | Path                   | Body example                                                             |
| -------- | ------ | ---------------------- | ------------------------------------------------------------------------ |
| OpenAI   | `POST` | `/v1/chat/completions` | `{ "model": "…", "messages": [ … ] }`                                    |
| Standard | `POST` | `/runsync`             | `{ "input": { "text": "…", "sampling_params": {"max_new_tokens":64} } }` |

#### Response (non-stream)

```jsonc
{
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Paris." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10 }
}
```

---

## Usage

Below are minimal `python` snippets so you can copy-paste to get started quickly or get inspired on how to use the endpoint.

> Replace `<ENDPOINT_ID>` with your endpoint ID and `<API_KEY>` with a [RunPod API key](https://docs.runpod.io/get-started/api-keys).

### OpenAI compatible API

Minimal Python example using the official `openai` SDK:

```python
from openai import OpenAI
import os

# Initialize the OpenAI Client with your RunPod API Key and Endpoint URL
client = OpenAI(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url=f"https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
)
```

`Chat Completions (Non-Streaming)`

```python
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Give a two lines on Planet Earth ?"}],
    temperature=0,
    max_tokens=100,

)
print(f"Response: {response}")
```

`Chat Completions (Streaming)`

```python
response_stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Give a two lines on Planet Earth ?"}],
    temperature=0,
    max_tokens=100,
    stream=True

)
for response in response_stream:
    print(response.choices[0].delta.content or "", end="", flush=True)
```

### Standard RunPod Calls

If you prefer to call the SGLang _native_ endpoint (what the handler sends when no OpenAI-style keys are detected) wrap the body under the `input` key as usual:

```jsonc
{
  "input": {
    "text": "<|user|>\nWhat is the capital of France?\n<|assistant|>",
    "sampling_params": {
      "max_new_tokens": 64,
      "temperature": 0.0
    },
    "stream": false
  }
}
```

Key fields under `input` you can provide:

| Field             | Type                   | Required | Notes                                                                               |
| ----------------- | ---------------------- | -------- | ----------------------------------------------------------------------------------- |
| `text`            | string or list[string] | ✔️       | The prompt. Specify **either** `text`, `input_ids`, **or** `input_embeds`.          |
| `sampling_params` | object                 | optional | Same schema as the OpenAI parameters (temperature, top_p, n, max_new_tokens, etc.). |
| `stream`          | bool                   | optional | If `true`, response is server-sent events just like OpenAI streaming.               |

> [!IMPORTANT]
>
> SGLang will **not** auto-wrap your prompt when you call `/run`, `/runsync` or `/generate` directly.
> **You have three options:**
>
> 1. **Use the OpenAI-compatible endpoints** (`/v1/chat/completions`, etc.). The server adds the template for you.
> 2. **Wrap the prompt yourself.** Embed the appropriate chat-template tokens (see example above).
> 3. **Let the server apply a template automatically** by setting:
>
>    ```bash
>    CHAT_TEMPLATE=llama3   # or qwen, openchat, mistral-instruct, …
>    ```
>
>    This becomes `--chat-template llama3` for `sglang.launch_server`.
>    Full list: <https://docs.sglang.ai/backend/server_arguments.html#model-processor-and-tokenizer>

## Compatibility

Anything not recognized by worker-sglang is forwarded verbatim to `/generate`, so advanced options in the SGLang docs (logprobs, sessions, images, etc.) also work.
