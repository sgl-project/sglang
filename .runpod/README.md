<h1> SgLang Worker</h1>

🚀 | SGLang is fast serving framework for large language models and vision language models.

---

[![RunPod](https://api.runpod.io/badge/timpietrusky/sglang)](https://www.runpod.io/console/hub/timpietrusky/sglang)

---

## SGLang Server Configuration

When launching an endpoint, you can configure the SGLang server using environment variables. These variables allow you to customize various aspects of the server's behavior without modifying the code.

### How to Use

Define these variables in your endpoint template.
The SGLang server will read these variables at startup and configure itself accordingly.
If a variable is not set, the server will use its default value.

### Available Environment Variables

The following table lists all available environment variables for configuring the SGLang server:

| Environment Variable        | Description                              | Default                               | Options                                                                                   |
| --------------------------- | ---------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------------------- |
| `MODEL_PATH`                | Path of the model weights                | "meta-llama/Meta-Llama-3-8B-Instruct" | Local folder or Hugging Face repo ID                                                      |
| `HOST`                      | Host of the server                       | "0.0.0.0"                             |                                                                                           |
| `PORT`                      | Port of the server                       | 30000                                 |                                                                                           |
| `TOKENIZER_PATH`            | Path of the tokenizer                    |                                       |                                                                                           |
| `ADDITIONAL_PORTS`          | Additional ports for the server          |                                       |                                                                                           |
| `TOKENIZER_MODE`            | Tokenizer mode                           | "auto"                                | "auto", "slow"                                                                            |
| `LOAD_FORMAT`               | Format of model weights to load          | "auto"                                | "auto", "pt", "safetensors", "npcache", "dummy"                                           |
| `DTYPE`                     | Data type for weights and activations    | "auto"                                | "auto", "half", "float16", "bfloat16", "float", "float32"                                 |
| `CONTEXT_LENGTH`            | Model's maximum context length           |                                       |                                                                                           |
| `QUANTIZATION`              | Quantization method                      |                                       | "awq", "fp8", "gptq", "marlin", "gptq_marlin", "awq_marlin", "squeezellm", "bitsandbytes" |
| `SERVED_MODEL_NAME`         | Override model name in API               |                                       |                                                                                           |
| `CHAT_TEMPLATE`             | Chat template name or path               |                                       |                                                                                           |
| `MEM_FRACTION_STATIC`       | Fraction of memory for static allocation |                                       |                                                                                           |
| `MAX_RUNNING_REQUESTS`      | Maximum number of running requests       |                                       |                                                                                           |
| `MAX_NUM_REQS`              | Maximum requests in memory pool          |                                       |                                                                                           |
| `MAX_TOTAL_TOKENS`          | Maximum tokens in memory pool            |                                       |                                                                                           |
| `CHUNKED_PREFILL_SIZE`      | Max tokens in chunk for chunked prefill  |                                       |                                                                                           |
| `MAX_PREFILL_TOKENS`        | Max tokens in prefill batch              |                                       |                                                                                           |
| `SCHEDULE_POLICY`           | Request scheduling policy                |                                       | "lpm", "random", "fcfs", "dfs-weight"                                                     |
| `SCHEDULE_CONSERVATIVENESS` | Conservativeness of schedule policy      |                                       |                                                                                           |
| `TENSOR_PARALLEL_SIZE`      | Tensor parallelism size                  |                                       |                                                                                           |
| `STREAM_INTERVAL`           | Streaming interval in token length       |                                       |                                                                                           |
| `RANDOM_SEED`               | Random seed                              |                                       |                                                                                           |
| `LOG_LEVEL`                 | Logging level for all loggers            |                                       |                                                                                           |
| `LOG_LEVEL_HTTP`            | Logging level for HTTP server            |                                       |                                                                                           |
| `API_KEY`                   | API key for the server                   |                                       |                                                                                           |
| `FILE_STORAGE_PTH`          | Path of file storage in backend          |                                       |                                                                                           |
| `DATA_PARALLEL_SIZE`        | Data parallelism size                    |                                       |                                                                                           |
| `LOAD_BALANCE_METHOD`       | Load balancing strategy                  |                                       | "round_robin", "shortest_queue"                                                           |
| `NCCL_INIT_ADDR`            | NCCL init address for multi-node         |                                       |                                                                                           |
| `NNODES`                    | Number of nodes                          |                                       |                                                                                           |
| `NODE_RANK`                 | Node rank                                |                                       |                                                                                           |

**Boolean Flags** (set to "true", "1", or "yes" to enable):

| Flag                          | Description                               |
| ----------------------------- | ----------------------------------------- |
| `SKIP_TOKENIZER_INIT`         | Skip tokenizer init                       |
| `TRUST_REMOTE_CODE`           | Allow custom models from Hub              |
| `LOG_REQUESTS`                | Log inputs and outputs of requests        |
| `SHOW_TIME_COST`              | Show time cost of custom marks            |
| `DISABLE_FLASHINFER`          | Disable flashinfer attention kernels      |
| `DISABLE_FLASHINFER_SAMPLING` | Disable flashinfer sampling kernels       |
| `DISABLE_RADIX_CACHE`         | Disable RadixAttention for prefix caching |
| `DISABLE_REGEX_JUMP_FORWARD`  | Disable regex jump-forward                |
| `DISABLE_CUDA_GRAPH`          | Disable cuda graph                        |
| `DISABLE_DISK_CACHE`          | Disable disk cache                        |
| `ENABLE_TORCH_COMPILE`        | Optimize model with torch.compile         |
| `ENABLE_P2P_CHECK`            | Enable P2P check for GPU access           |
| `ENABLE_MLA`                  | Enable Multi-head Latent Attention        |
| `ATTENTION_REDUCE_IN_FP32`    | Cast attention results to fp32            |
| `EFFICIENT_WEIGHT_LOAD`       | Enable memory efficient weight loading    |

## Usage

### OpenAI compatible API

```python
from openai import OpenAI
import os

# Initialize the OpenAI Client with your RunPod API Key and Endpoint URL
client = OpenAI(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url=f"https://api.runpod.ai/v2/<endpoint_id>/openai/v1",
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
