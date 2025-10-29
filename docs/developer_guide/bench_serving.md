## Bench Serving Guide

This guide explains how to benchmark online serving throughput and latency using `python -m sglang.bench_serving`. It supports multiple inference backends via OpenAI-compatible and native endpoints, and produces both console metrics and optional JSONL outputs.

### What it does

- Generates synthetic or dataset-driven prompts and submits them to a target serving endpoint
- Measures throughput, time-to-first-token (TTFT), inter-token latency (ITL), per-request end-to-end latency, and more
- Supports streaming or non-streaming modes, rate control, and concurrency limits

### Supported backends and endpoints

- `sglang` / `sglang-native`: `POST /generate`
- `sglang-oai`, `vllm`, `lmdeploy`: `POST /v1/completions`
- `sglang-oai-chat`, `vllm-chat`, `lmdeploy-chat`: `POST /v1/chat/completions`
- `trt` (TensorRT-LLM): `POST /v2/models/ensemble/generate_stream`
- `gserver`: Custom server (Not Implemented yet in this script)
- `truss`: `POST /v1/models/model:predict`

If `--base-url` is provided, requests are sent to it. Otherwise, `--host` and `--port` are used. When `--model` is not provided, the script will attempt to query `GET /v1/models` for an available model ID (OpenAI-compatible endpoints).

### Prerequisites

- Python 3.8+
- Dependencies typically used by this script: `aiohttp`, `numpy`, `requests`, `tqdm`, `transformers`, and for some datasets `datasets`, `pillow`, `pybase64`. Install as needed.
- An inference server running and reachable via the endpoints above
- If your server requires authentication, set environment variable `OPENAI_API_KEY` (used as `Authorization: Bearer <key>`)

### Quick start

Run a basic benchmark against an sglang server exposing `/generate`:

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
```

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

Or, using an OpenAI-compatible endpoint (completions):

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --num-prompts 1000 \
  --model meta-llama/Llama-3.1-8B-Instruct
```

### Datasets

Select with `--dataset-name`:

- `sharegpt` (default): loads ShareGPT-style pairs; optionally restrict with `--sharegpt-context-len` and override outputs with `--sharegpt-output-len`
- `random`: random text lengths; sampled from ShareGPT token space
- `random-ids`: random token ids (can lead to gibberish)
- `image`: generates images and wraps them in chat messages; supports custom resolutions, multiple formats, and different content types
- `generated-shared-prefix`: synthetic dataset with shared long system prompts and short questions
- `mmmu`: samples from MMMU (Math split) and includes images

Common dataset flags:

- `--num-prompts N`: number of requests
- `--random-input-len`, `--random-output-len`, `--random-range-ratio`: for random/random-ids/image
- `--image-count`: Number of images per request (for `image` dataset).

- `--apply-chat-template`: apply tokenizer chat template when constructing prompts
- `--dataset-path PATH`: file path for ShareGPT json; if blank and missing, it will be downloaded and cached

Generated Shared Prefix flags (for `generated-shared-prefix`):

- `--gsp-num-groups`
- `--gsp-prompts-per-group`
- `--gsp-system-prompt-len`
- `--gsp-question-len`
- `--gsp-output-len`

Image dataset flags (for `image`):

- `--image-count`: Number of images per request
- `--image-resolution`: Image resolution; supports presets (4k, 1080p, 720p, 360p) or custom 'heightxwidth' format (e.g., 1080x1920, 512x768)
- `--image-format`: Image format (jpeg or png)
- `--image-content`: Image content type (random or blank)

### Examples

1. To benchmark image dataset with 3 images per request, 500 prompts, 512 input length, and 512 output length, you can run:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --disable-radix-cache
```

```bash
python -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --dataset-name image \
    --num-prompts 500 \
    --image-count 3 \
    --image-resolution 720p \
    --random-input-len 512 \
    --random-output-len 512
```

2. To benchmark random dataset with 3000 prompts, 1024 input length, and 1024 output length, you can run:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-3B-Instruct
```

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --num-prompts 3000 \
    --random-input 1024 \
    --random-output 1024 \
    --random-range-ratio 0.5
```

### Choosing model and tokenizer

- `--model` is required unless the backend exposes `GET /v1/models`, in which case the first model ID is auto-selected.
- `--tokenizer` defaults to `--model`. Both can be HF model IDs or local paths.
- For ModelScope workflows, setting `SGLANG_USE_MODELSCOPE=true` enables fetching via ModelScope (weights are skipped for speed).
- If your tokenizer lacks a chat template, the script warns because token counting can be less robust for gibberish outputs.

### Rate, concurrency, and streaming

- `--request-rate`: requests per second. `inf` sends all immediately (burst). Non-infinite rate uses a Poisson process for arrival times.
- `--max-concurrency`: caps concurrent in-flight requests regardless of arrival rate.
- `--disable-stream`: switch to non-streaming mode when supported; TTFT then equals total latency for chat completions.

### Other key options

- `--output-file FILE.jsonl`: append JSONL results to file; auto-named if unspecified
- `--output-details`: include per-request arrays (generated texts, errors, ttfts, itls, input/output lens)
- `--extra-request-body '{"top_p":0.9,"temperature":0.6}'`: merged into payload (sampling params, etc.)
- `--disable-ignore-eos`: pass through EOS behavior (varies by backend)
- `--warmup-requests N`: run warmup requests with short output first (default 1)
- `--flush-cache`: call `/flush_cache` (sglang) before main run
- `--profile`: call `/start_profile` and `/stop_profile` (requires server to enable profiling, e.g., `SGLANG_TORCH_PROFILER_DIR`)
- `--lora-name name1 name2 ...`: randomly pick one per request and pass to backend (e.g., `lora_path` for sglang)
- `--tokenize-prompt`: send integer IDs instead of text (currently supports `--backend sglang` only)

### Authentication

If your target endpoint requires OpenAI-style auth, set:

```bash
export OPENAI_API_KEY=sk-...yourkey...
```

The script will add `Authorization: Bearer $OPENAI_API_KEY` automatically for OpenAI-compatible routes.

### Metrics explained

Printed after each run:

- Request throughput (req/s)
- Input token throughput (tok/s) - includes both text and vision tokens
- Output token throughput (tok/s)
- Total token throughput (tok/s) - includes both text and vision tokens
- Total input text tokens and Total input vision tokens - per-modality breakdown
- Concurrency: aggregate time of all requests divided by wall time
- End-to-End Latency (ms): mean/median/std/p99 per-request total latency
- Time to First Token (TTFT, ms): mean/median/std/p99 for streaming mode
- Inter-Token Latency (ITL, ms): mean/median/std/p95/p99/max between tokens
- TPOT (ms): Token processing time after first token, i.e., `(latency - ttft)/(tokens-1)`
- Accept length (sglang-only, if available): speculative decoding accept length

The script also retokenizes generated text with the configured tokenizer and reports "retokenized" counts.

### JSONL output format

When `--output-file` is set, one JSON object is appended per run. Base fields:

- Arguments summary: backend, dataset, request_rate, max_concurrency, etc.
- Duration and totals: completed, total_input_tokens, total_output_tokens, retokenized totals
- Throughputs and latency statistics as printed in the console
- `accept_length` when available (sglang)

With `--output-details`, an extended object also includes arrays:

- `input_lens`, `output_lens`
- `ttfts`, `itls` (per request: ITL arrays)
- `generated_texts`, `errors`

### End-to-end examples

1) sglang native `/generate` (streaming):

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.5 \
  --num-prompts 2000 \
  --request-rate 100 \
  --max-concurrency 512 \
  --output-file sglang_random.jsonl --output-details
```

2) OpenAI-compatible Completions (e.g., vLLM):

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --sharegpt-output-len 256
```

3) OpenAI-compatible Chat Completions (streaming):

```bash
python3 -m sglang.bench_serving \
  --backend vllm-chat \
  --base-url http://127.0.0.1:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --num-prompts 500 \
  --apply-chat-template
```

4) Images (VLM) with chat template:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 2 \
  --image-resolution 720p \
  --random-input-len 128 --random-output-len 256 \
  --num-prompts 200 \
  --apply-chat-template
```

4a) Images with custom resolution:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 512x768 \
  --random-input-len 64 --random-output-len 128 \
  --num-prompts 100 \
  --apply-chat-template
```

4b) 1080p images with PNG format and blank content:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model your-vlm-model \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 1080p \
  --image-format png \
  --image-content blank \
  --random-input-len 64 --random-output-len 128 \
  --num-prompts 100 \
  --apply-chat-template
```

5) Generated shared prefix (long system prompts + short questions):

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 64 --gsp-prompts-per-group 16 \
  --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 256 \
  --num-prompts 1024
```

6) Tokenized prompts (ids) for strict length control (sglang only):

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --tokenize-prompt \
  --random-input-len 2048 --random-output-len 256 --random-range-ratio 0.2
```

7) Profiling and cache flush (sglang):

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --profile \
  --flush-cache
```

8) TensorRT-LLM streaming endpoint:

```bash
python3 -m sglang.bench_serving \
  --backend trt \
  --base-url http://127.0.0.1:8000 \
  --model your-trt-llm-model \
  --dataset-name random \
  --num-prompts 100 \
  --disable-ignore-eos
```

9) Evaluating large-scale KVCache sharing with mooncake trace (sglang only):

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --model mode-name \
  --dataset-name mooncake \
  --mooncake-slowdown-factor 1.0 \
  --mooncake-num-rounds 1000 \
  --mooncake-workload conversation|mooncake|agent|synthetic
  --use-trace-timestamps true \
  --random-output-len 256
```

### Troubleshooting

- All requests failed: verify `--backend`, server URL/port, `--model`, and authentication. Check warmup errors printed by the script.
- Throughput seems too low: adjust `--request-rate` and `--max-concurrency`; verify server batch size/scheduling; ensure streaming is enabled if appropriate.
- Token counts look odd: prefer chat/instruct models with proper chat templates; otherwise tokenization of gibberish may be inconsistent.
- Image/MMMU datasets: ensure you installed extra deps (`pillow`, `datasets`, `pybase64`).
- Authentication errors (401/403): set `OPENAI_API_KEY` or disable auth on your server.

### Notes

- The script raises the file descriptor soft limit (`RLIMIT_NOFILE`) to help with many concurrent connections.
- For sglang, `/get_server_info` is queried post-run to report speculative decoding accept length when available.
