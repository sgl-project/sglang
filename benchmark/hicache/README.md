## Run synthetic multi-turn benchmark

```
# SGLang server with radix cache disabled
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --disable-radix-cache

# SGLang server with radix cache on and first-come-first-serve policy
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --schedule-policy fcfs

# The default SGLang server with radix cache on and long-prefix-match policy
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000

# SGLang server with hierarchical radix cache enabled
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --enable-hierarchical-cache

```

```
python bench_multiturn.py --model-path Qwen/Qwen2.5-14B-Instruct
```

Note: The performance gain of hierarchical caching depends on the ratio of reusable tokens to GPU memory capacity. The more tokens to be reused, the larger the model, and the more constrained the GPU memory size, the greater the benefit one can expect from hierarchical caching.


# Benchmark with more datasets
## Download Dataset
```bash
./download.sh {sharegpt|ultragpt|loogle|nextqa|all}
```
This script will automatically download the required dataset to the current working directory

## Multiturn Benchmark
### Supported Datasets
- sharegpt
- ultrachat
- loogle
### Example Usage:
```bash
python3 bench_serving.py --model mistralai/Mistral-7B-Instruct-v0.3 --backend sglang \
--dataset-path longdep_qa.json --dataset-name loogle --request-rate 10 --num-prompts 10  \
--port 8001 --enable-multiturn --disable-shuffle
```
This uses `mistralai/Mistral-7B-Instruct-v0.3` model with `sglang` as backend. The dataset
is `longdep_qa.json`. We send `10 conversations` with `10 req/s` to port 8001. We enable
multiturn chat without shuffling the order of conversations (i.e. following the original
order in the dataset file).

### Note:
The requests of multiple conversations are sent in a round robin fashion.
For example, if we have 3 conversations A, B, C whose rounds are `[2, 3, 4]` correspondingly,
multiturn chat will send the requests to the backend in the following order: `[A1, B1, C1, A2, B2, C2, B3, C3, C4]`
This has implications on the cache reuse patterns: the cache reuse distance is the largest
under this request pattern (which means a prefix-aware local scheduler in the backend can
yield the most benefit compared to a FIFO scheduler)

## Shared Prefix Benchmark
### Supported Datasets
- loogle
### Example Usage:
```bash
python3 bench_serving.py --model mistralai/Mistral-7B-Instruct-v0.3 --backend sglang \
--dataset-path longdep_qa.json --dataset-name loogle --request-rate 10 --num-prompts 10  \
--port 8001 --enable-shared-prefix --disable-shuffle
```
### Note:
Shared Prefix benchmark sends the questions for the same prompt together. For example,
if we have 3 shared prefix A, B, C, which have [2, 3, 4] questions correspondingly,
the shared prefix benchmark will send the requests to the
backend in the following order: `[A+Q1, A+Q2, B+Q1, B+Q2, B+Q3, C+Q1, C+Q2, C+Q3]`.


## Multi Modality Benchmark (WIP)
### Supported Datasets:
- nextqa
### Example Usage:
```bash
Server:
python3 -m sglang.launch_server --model-path lmms-lab/LLaVA-NeXT-Video-7B  --tp 2 --dp 1 --port 8001 \
--host 0.0.0.0 --mem-fraction-static 0.9 --tokenizer-path llava-hf/llava-1.5-7b-hf \
--json-model-override-args "{\"architectures\": [\"LlavaVidForCausalLM\"], \"model_type\":\"llava\", \"mm_spatial_pool_stride\":2}"

Client:
python3 bench_serving.py --model lmms-lab/LLaVA-NeXT-Video-7B --backend sglang  --dataset-path \
NExTVideo  --dataset-name nextqa --request-rate 10 --num-prompts 1 --disable-shuffle --port 8001 \ --enable-multiturn --max-frames 16 --tokenizer llava-hf/llava-1.5-7b-hf --fixed-output-len 2048
```
Note: for the server args, `tokenizer-path`, overriding architecture are necessary.

## Warm Cache Benchmark
`bench_warm_cache.py` measures cache-hit performance with exact control over
shared-prefix ratio. For each shared-prefix percentage it flushes the KV cache,
warms the shared prefix once, then benchmarks the full prompts.

### Example Usage:
```bash
# Native SGLang /generate endpoint (default)
python3 bench_warm_cache.py --model Qwen/Qwen2.5-14B-Instruct --backend sglang \
  --total-tokens 70000 --num-prompts 64 --pcts 0,50,90,99

# OpenAI-compatible chat/completions (e.g. via SGLang OAI frontend)
python3 bench_warm_cache.py --model Qwen/Qwen2.5-14B-Instruct --backend sglang-oai-chat \
  --total-tokens 70000 --num-prompts 64 --pcts 0,50,90,99

# OpenAI-compatible completions
python3 bench_warm_cache.py --model Qwen/Qwen2.5-14B-Instruct --backend sglang-oai \
  --total-tokens 70000 --num-prompts 64 --pcts 0,50,90,99
```

Key arguments:
- `--total-tokens`: total input length per request (prefix + suffix)
- `--pcts`: comma-separated shared-prefix percentages to sweep
- `--output-file results.jsonl`: append per-percentage results to a JSONL file

Supports the same backends as `bench_serving.py` (see table below).

### Backend performance note
The `sglang`, `sglang-oai`, `vllm`, and `lmdeploy` backends send pre-tokenized
IDs directly — the server skips tokenization and results are the most accurate
for cache studies. The `*-chat` backends send text via `/v1/chat/completions`,
so the server must tokenize every request; this adds per-request overhead that
grows with `--total-tokens` and is unrelated to cache performance.

## Supported Backends
`bench_serving.py` supports `sglang`, `vllm`, and `lmdeploy`. `bench_warm_cache.py`
additionally supports the `-oai` and `-chat` variants.
Use `--base-url` when the server is not on the backend's default port.

| Backend | Endpoint | Default port |
| --- | --- | --- |
| `sglang` | `/generate` (native, token-id level) | 30000 |
| `sglang-oai` | `/v1/completions` | 30000 |
| `sglang-oai-chat` | `/v1/chat/completions` | 30000 |
| `vllm` | `/v1/completions` | 8000 |
| `vllm-chat` | `/v1/chat/completions` | 8000 |
| `lmdeploy` | `/v1/completions` | 23333 |
| `lmdeploy-chat` | `/v1/chat/completions` | 23333 |
