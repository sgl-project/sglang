### Benchmark sglang

Run Llama-7B

```
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

Run Mixtral-8x7B
(When there is a CUDA out-of-memory error, try to reduce the `--mem-fraction-static`)

```
python3 -m sglang.launch_server --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 --port 30000 --tp-size 8
```

Benchmark(short output)

```
python3 bench_sglang.py --tokenizer meta-llama/Llama-2-7b-chat-hf
```

Benchmark(long output)

```
python3 bench_sglang.py --tokenizer meta-llama/Llama-2-7b-chat-hf --long
```

### Benchmark Router Responses

This adapter reuses the existing multi-turn workload generator against a router
Responses endpoint without replacing the direct-server scripts above.

HTTP SSE and WebSocket comparison on a regular gRPC-worker router:

```
python3 bench_router_responses.py \
  --base-url http://127.0.0.1:30000 \
  --model Qwen/Qwen2.5-72B-Instruct \
  --tokenizer Qwen/Qwen2.5-72B-Instruct \
  --client-transport both \
  --chain-mode full_replay \
  --store-mode true \
  --worker-transport grpc \
  --router-topology regular_grpc_worker
```

Persistent WebSocket continuation path with incremental `previous_response_id`
chaining:

```
python3 bench_router_responses.py \
  --base-url http://127.0.0.1:30000 \
  --model Qwen/Qwen2.5-72B-Instruct \
  --tokenizer Qwen/Qwen2.5-72B-Instruct \
  --client-transport websocket \
  --chain-mode previous_response_id \
  --store-mode false \
  --worker-transport grpc \
  --router-topology regular_grpc_worker
```

Notes:

- `full_replay` mirrors the direct multi-turn-chat shape by replaying the full
  local conversation window each turn.
- `previous_response_id` isolates the Responses continuation path instead.
- HTTP SSE with `previous_response_id` requires `--store-mode true`; otherwise
  the chain cannot continue across requests.
- Add `--capture-event-trace` to record bounded per-turn event timing traces in
  the JSON summary. This is useful when comparing HTTP `first_event` vs
  `first_content` gaps against the persistent WebSocket path.

### Benchmark vLLM

Run Llama-7B

```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf  --disable-log-requests --port 21000
```

Run Mixtral-8x7B

```
python3 -m vllm.entrypoints.api_server --tokenizer-mode auto --model mistralai/Mixtral-8x7B-Instruct-v0.1 --disable-log-requests --port 21000 --tensor-parallel-size 8
```

Benchmark(short output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend vllm
```

Benchmark(long output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend vllm --long
```

### Benchmark guidance

Benchmark Llama-7B (short output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf
```

Benchmark Llama-7B (long output)

```
python3 bench_other.py --tokenizer meta-llama/Llama-2-7b-chat-hf --backend guidance --parallel 1 --n-ctx 4096 --model-path path/to/gguf --long
```
