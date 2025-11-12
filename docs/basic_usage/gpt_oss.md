# GPT OSS Usage

Please refer to [https://github.com/sgl-project/sglang/issues/8833](https://github.com/sgl-project/sglang/issues/8833).

## Responses API & Built-in Tools

### Responses API

GPT‑OSS is compatible with the OpenAI Responses API. Use `client.responses.create(...)` with `model`, `instructions`, `input`, and optional `tools` to enable built‑in tool use. You can set reasoning level via `instructions`, e.g., "Reasoning: high" (also supports "medium" and "low") — levels: low (fast), medium (balanced), high (deep).

### Built-in Tools

GPT‑OSS can call built‑in tools for web search and Python execution. You can use the demo tool server or connect to external MCP tool servers.

#### Python Tool

- Executes short Python snippets for calculations, parsing, and quick scripts.
- By default runs in a Docker-based sandbox. To run on the host, set `PYTHON_EXECUTION_BACKEND=UV` (this executes model-generated code locally; use with care).
- Ensure Docker is available if you are not using the UV backend. It is recommended to run `docker pull python:3.11` in advance.

#### Web Search Tool

- Uses the Exa backend for web search.
- Requires an Exa API key; set `EXA_API_KEY` in your environment. Create a key at `https://exa.ai`.

### Tool & Reasoning Parser

- We support OpenAI Reasoning and Tool Call parser, as well as our SGLang native api for tool call and reasoning. Refer to [reasoning parser](../advanced_features/separate_reasoning.ipynb) and [tool call parser](../advanced_features/function_calling.ipynb) for more details.


## Notes

- Use **Python 3.12** for the demo tools. And install the required `gpt-oss` packages.
- The default demo integrates the web search tool (Exa backend) and a demo Python interpreter via Docker.
- For search, set `EXA_API_KEY`. For Python execution, either have Docker available or set `PYTHON_EXECUTION_BACKEND=UV`.

Examples:
```bash
export EXA_API_KEY=YOUR_EXA_KEY
# Optional: run Python tool locally instead of Docker (use with care)
export PYTHON_EXECUTION_BACKEND=UV
```

Launch the server with the demo tool server:

```bash
python3 -m sglang.launch_server \
  --model-path openai/gpt-oss-120b \
  --tool-server demo \
  --tp 2
```

For production usage, sglang can act as an MCP client for multiple services. An [example tool server](https://github.com/openai/gpt-oss/tree/main/gpt-oss-mcp-server) is provided. Start the servers and point sglang to them:
```bash
mcp run -t sse browser_server.py:mcp
mcp run -t sse python_server.py:mcp

python -m sglang.launch_server ... --tool-server ip-1:port-1,ip-2:port-2
```
The URLs should be MCP SSE servers that expose server information and well-documented tools. These tools are added to the system prompt so the model can use them.

### Quick Demo

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="sk-123456"
)

tools = [
    {"type": "code_interpreter"},
    {"type": "web_search_preview"},
]

# Reasoning level example
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helpful assistant."
    reasoning_effort="high" # Supports high, medium, or low
    input="In one sentence, explain the transformer architecture.",
)
print("====== reasoning: high ======")
print(response.output_text)

# Test python tool
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helfpul assistant, you could use python tool to execute code.",
    input="Use python tool to calculate the sum of 29138749187 and 29138749187", # 58,277,498,374
    tools=tools
)
print("====== test python tool ======")
print(response.output_text)

# Test browser tool
response = client.responses.create(
    model="openai/gpt-oss-120b",
    instructions="You are a helfpul assistant, you could use browser to search the web",
    input="Search the web for the latest news about Nvidia stock price",
    tools=tools
)
print("====== test browser tool ======")
print(response.output_text)
```

Example output:
```
====== test python tool ======
The sum of 29,138,749,187 and 29,138,749,187 is **58,277,498,374**.
====== test browser tool ======
**Recent headlines on Nvidia (NVDA) stock**

| Date (2025) | Source | Key news points | Stock‑price detail |
|-------------|--------|----------------|--------------------|
| **May 13** | Reuters | The market data page shows Nvidia trading “higher” at **$116.61** with no change from the previous close. | **$116.61** – latest trade (delayed ≈ 15 min)【14†L34-L38】 |
| **Aug 18** | CNBC | Morgan Stanley kept an **overweight** rating and lifted its price target to **$206** (up from $200), implying a 14 % upside from the Friday close. The firm notes Nvidia shares have already **jumped 34 % this year**. | No exact price quoted, but the article signals strong upside expectations【9†L27-L31】 |
| **Aug 20** | The Motley Fool | Nvidia is set to release its Q2 earnings on Aug 27. The article lists the **current price of $175.36**, down 0.16 % on the day (as of 3:58 p.m. ET). | **$175.36** – current price on Aug 20【10†L12-L15】【10†L53-L57】 |

**What the news tells us**

* Nvidia’s share price has risen sharply this year – up roughly a third according to Morgan Stanley – and analysts are still raising targets (now $206).
* The most recent market quote (Reuters, May 13) was **$116.61**, but the stock has surged since then, reaching **$175.36** by mid‑August.
* Upcoming earnings on **Aug 27** are a focal point; both the Motley Fool and Morgan Stanley expect the results could keep the rally going.

**Bottom line:** Nvidia’s stock is on a strong upward trajectory in 2025, with price targets climbing toward $200‑$210 and the market price already near $175 as of late August.

```

# Quick Start Recipe for GPT-OSS on NVIDIA Blackwell & Hopper Hardware

This section provides step-by-step instructions for running the GPT-OSS-20b/120b model using SGLang with MXFP4 quantization, optimized for NVIDIA Blackwell and Hopper architecture GPUs. It covers the complete setup required; from accessing model weights and preparing the software environment to configuring SGLang parameters, launching the server, and validating inference output.

The recipe is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA’s accelerated stack—building a docker image with SGLang for model serving and FlashInfer for optimized CUDA kernels.

## Prerequisites

- OS: Linux
- Drivers: CUDA Driver 575 or above
- GPU: Blackwell architecture or Hopper Architecture
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

## Deployment Steps

Please refer to the [SGLang installation guide](https://docs.sglang.ai/get_started/install.html) about how to install SGLang. In this guide, we assume using the official SGLang docker image.

### Pull Docker Image

Pull the SGLang latest release docker image.

`pull_image.sh`
```
# On x86_64 systems:
docker pull --platform linux/amd64 lmsysorg/sglang:latest
# On aarch64 systems:
# docker pull --platform linux/aarch64 lmsysorg/sglang:latest

docker tag lmsysorg/sglang:latest lmsysorg/sglang:deploy
```

### Run Docker Container

Run the docker container using the docker image `lmsysorg/sglang:deploy`.

`run_container.sh`
```
docker run -e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME" --ipc=host --gpus all --entrypoint "/bin/bash" --rm -it lmsysorg/sglang:deploy
```

Note: You can mount additional directories and paths using the `-v <local_path>:<path>` flag if needed, such as mounting the downloaded weight paths.

The `-e HF_TOKEN="$HF_TOKEN" -e HF_HOME="$HF_HOME"` flags are added so that the models are downloaded using your HuggingFace token and the downloaded models can be cached in $HF_HOME. Refer to [HuggingFace documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) for more information about these environment variables and refer to [HuggingFace Quickstart guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication) about steps to generate your HuggingFace access token.

### Launch the SGLang Server

Below is an example command to launch the SGLang server with GPT-OSS-120b model. The instruction is the same for GPT-OSS-20b with the model name replaced with `openai/gpt-oss-20b`.

`launch_server.sh`
```
TP=1
MAX_RUNNING_REQUESTS=1024

python3 -m sglang.launch_server \
--model openai/gpt-oss-120b \
--kv-cache-dtype fp8_e4m3 \
--disable-radix-cache \
--mem-fraction-static 0.95 \
--stream-interval 20 \
--tensor-parallel-size ${TP} \
--max-running-requests ${MAX_RUNNING_REQUESTS} \
--cuda-graph-max-bs ${MAX_RUNNING_REQUESTS} &
```

After the server is set up, the client can now send prompt requests to the server and receive results.

### Configs and Parameters

You can specify the IP address and the port that you would like to run the server with using these flags:

- `host`: IP address of the server. By default, it uses 127.0.0.1.
- `port`: The port to listen to by the server. By default, it uses port 30000.

Below are the config flags that we do not recommend changing or tuning with:

- `kv-cache-dtype`: Kv-cache data type. We recommend setting it to `fp8_e4m3` for best performance.
- `disable-radix-cache`: Disable prefix caching. We recommend always adding this flag if running with synthetic dataset for consistent performance measurement.
- `mem-fraction-static`: The fraction of GPU memory that SGLang server is allowed to use. Re recommend setting it to `0.95` to maximize the throughput by allowing larger `MAX_RUNNING_REQUESTS`.
- `stream-interval`: The interval between output token streaming responses. We recommend setting this to `20` to maximize the throughput.
- `cuda-graph-max-bs`: Specify the max size for cuda graphs. We recommend setting this to `MAX_RUNNING_REQUESTS` to leverage the benefit of cuda graphs.

Below are a few tunable parameters you can modify based on your serving requirements:

- `tensor-parallel-size`: Tensor parallelism size. Increasing this will increase the number of GPUs that are used for inference.
  - Set this to `1` to achieve the best throughput per GPU, and set this to `2`, `4`, or `8` to achieve better per-user latencies.
- `max-running-requests`: Maximum number of sequences per batch.
  - Set this to a large number like `1024` to achieve the best throughput, and set this to a small number like `16` to achieve better per-user latencies.
- `context-length`: Maximum number of total tokens, including the input tokens and output tokens, for each request.
  - This must be set to a larger number if the expected input/output sequence lengths are large.
  - For example, if the maximum input sequence length is 1024 tokens and maximum output sequence length is 1024, then this must be set to at least 2048.

Refer to the "Balancing between Throughput and Latencies" about how to adjust these tunable parameters to meet your deployment requirements.

## Validation & Expected Behavior

### Basic Test

After the SGLang server is set up and shows `The server is fired up and ready to roll!` message, you can send requests to the server.

`run_basic_test.sh`
```
curl http://127.0.0.1:30000/v1/completions -H "Content-Type: application/json" -d '{ "model": "openai/gpt-oss-120b", "prompt": "San Francisco is a", "max_tokens": 20, "temperature": 0 }'
```

Here is an example response, showing that the SGLang server returns "*city in California, USA. It is known for its iconic landmarks such as the Golden Gate Bridge,*", completing the input sequence with up to 20 tokens.

```
{"id":"91ecdb584d7244dca2a083f56bf5687d","object":"text_completion","created":1762760361,"model":"openai/gpt-oss-120b","choices":[{"index":0,"text":" city in California, USA. It is known for its iconic landmarks such as the Golden Gate Bridge,","logprobs":null,"finish_reason":"length","matched_stop":null}],"usage":{"prompt_tokens":4,"total_tokens":24,"completion_tokens":20,"prompt_tokens_details":null,"reasoning_tokens":0},"metadata":{"weight_version":"default"}}
```

### Verify Accuracy

When the server is still running, we can run accuracy tests using the official GPT-OSS evaluation tool.

`run_accuracy.sh`
```
# Install GPT-OSS evaluation tool
pip install gpt-oss[eval]
export OPENAI_API_KEY="test"

# Run GPT-OSS evaluation tool
python3 -m gpt_oss.evals \
--base-url http://127.0.0.1:30000/v1 \
--model openai/gpt-oss-120b \
--reasoning-effort low \
--n-threads 128 \
--eval gpqa
```

Here is an example accuracy result with the `openai/gpt-oss-120b` model on one B200 GPU:

```
[{'eval_name': 'gpqa', 'model_name': 'openai__gpt-oss-120b-low_temp1.0_20251110_074418', 'metric': 0.6407828282828283}]
```

### Benchmarking Performance

To benchmark the performance, you can use the `sglang.bench_serving` tool.

`run_performance.sh`
```
python3 -m sglang.bench_serving \
--model openai/gpt-oss-120b \
--backend sglang-oai \
--dataset-name random \
--max-concurrency 1024 \
--num-prompts 5120 \
--random-input-len 1024 \
--random-output-len 1024 \
--random-range-ratio 1
```

Explanations for the flags:

- `--backend`: Specifies the backend used by the benchmark serving script.
- `--dataset-name`: Which dataset to use for benchmarking. We use a `random` dataset here.
- `--max-concurrency`: Maximum number of in-flight requests. We recommend matching this with the `--max-running-requests` flag used to launch the server.
- `--num-prompts`: Total number of prompts used for performance benchmarking. We recommend setting it to at least five times of the `--max-concurrency` to measure the steady state performance.
- `--random-input-len`: Specifies the average input sequence length.
- `--random-output-len`: Specifies the average output sequence length.
- `--random-range-ratio`: Specifies the dynamic range of the input and output sequence lengths. Setting it to `1` results in static input and output sequence lengths.

### Interpreting Performance Benchmarking Output

Sample output by the `python3 -m sglang.bench_serving` command:

```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf
Max request concurrency:                 1024
Successful requests:                     5120
Benchmark duration (s):                  XXXX
Total input tokens:                      5242880
Total input text tokens:                 5242880
Total input vision tokens:               0
Total generated tokens:                  5242880
Total generated tokens (retokenized):    5102478
Request throughput (req/s):              XXXX
Input token throughput (tok/s):          XXXX
Output token throughput (tok/s):         XXXX
Total token throughput (tok/s):          XXXX
Concurrency:                             XXXX
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   XXXX
Median E2E Latency (ms):                 XXXX
---------------Time to First Token----------------
Mean TTFT (ms):                          XXXX
Median TTFT (ms):                        XXXX
P99 TTFT (ms):                           XXXX
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          XXXX
Median TPOT (ms):                        XXXX
P99 TPOT (ms):                           XXXX
---------------Inter-Token Latency----------------
Mean ITL (ms):                           XXXX
Median ITL (ms):                         XXXX
P95 ITL (ms):                            XXXX
P99 ITL (ms):                            XXXX
Max ITL (ms):                            XXXX
==================================================
```

Explanations for key metrics:

- `Median Time to First Token (TTFT)`: T​​he typical time elapsed from when a request is sent until the first output token is generated.
- `Median Time Per Output Token (TPOT)`: The typical time required to generate each token after the first one.
- `Median Inter-Token Latency (ITL)`: The typical time delay between a response for the completion of one output token (or output tokens) and the next response for the completion of token(s).
  - If the `--stream-interval 20` flag is added in the server command, the ITL will be the completion time for every 20 output tokens.
- `Median End-to-End Latency (E2EL)`: The typical total time from when a request is submitted until the final token of the response is received.
- `Output token throughput`: The rate at which the system generates the output (generated) tokens.
- `Total token throughput`: The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.

### Balancing between Throughput and Latencies

In SGLang inference, the "throughput" can be defined as the number of generated tokens per second (the `Output token throughput` metric above) or the number of processed tokens per second (the `Total token throughput` metric above). These two throughput metrics are highly correlated. We usually divide the throughput by the number of GPUs used to get the "per-GPU throughput" when comparing across different parallelism configurations. The higher per-GPU throughput is, the fewer GPUs are needed to serve the same amount of the incoming requests.

On the other hand, the “latency” can be defined as the latency from when a request is sent until the first output token is generated (the `TTFT` metric), the latency between two generated tokens after the first one has been generated (the `TPOT` metric), or the end-to-end latency from when a request is sent to when the final token is generated (the `E2EL` metric). The TTFT affects the E2EL more when the input (prompt) sequence lengths are much longer than the output (generated) sequence lengths, while the TPOT affects the E2EL more in the opposite cases.

To achieve higher throughput, tokens from multiple requests must be batched and processed together, but that increases the latencies. Therefore, a balance must be made between throughput and latencies depending on the deployment requirements.

The two main tunable configs for GPT-OSS are the `--tensor-parallel-size` (TP) and `--max-running-requests` (BS). How they affect the throughput and latencies can be summarized as the following:

- At the same BS, higher TP typically results in lower latencies but also lower throughput.
- At the same TP size, higher BS typically results in higher throughput but worse latencies, but the maximum BS is limited by the amount of available GPU memory for the kv-cache after the weights are loaded.
- Therefore, increasing TP (which would lower the throughput at the same BS) may allow higher BS to run (which would increase the throughput), and the net throughput gain/loss depends on models and configurations.

Note that the statements above assume that the concurrency setting on the client side, like the `--max-concurrency` flag in the performance benchmarking command, matches the `--max-running-requests` (BS) setting on the server side.

Below are the recommended configs for different throughput-latency scenarios on B200 GPUs:

- Max Throughput: Set TP to 1, and increase BS to the maximum possible value without exceeding KV cache capacity.
- Min Latency: Set TP to 4 or 8, and set BS to a small value (like `8`) that meets the latency requirements.
- Balanced: Set TP to 2 and set BS to 128.
