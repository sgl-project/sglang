# Container Runbook

Use this runbook when the benchmark environment is container-based. It records
the exact image, command, help output, server log, benchmark log, and cleanup
step for each framework.

This runbook is target-agnostic. Every `docker run` / `docker exec` command
works on a local box, an SSH-reachable remote GPU host, or a CI runner; the
per-host skills (for example `h100`, `b200`, `rtx5090`, `radixark02`,
`radixark03`) only add the SSH wrapper, container name, and workspace path
for a specific operator box. Substitute those values where you see
`$SGLANG_CONTAINER`, `$SGLANG_WORKSPACE`, and similar; nothing below assumes
an H100.

## Common Setup

Pull the images that will be used:

```bash
docker pull lmsysorg/sglang:dev
docker pull vllm/vllm-openai:latest
docker pull nvcr.io/nvidia/tensorrt-llm/release:latest
```

Use quoted Docker GPU device lists:

```bash
GPU_ARG='"device=6,7"'
docker run --gpus "$GPU_ARG" ...
```

The unquoted form `--gpus device=6,7` can be parsed incorrectly by Docker.

Mount the shared Hugging Face cache and pass tokens through environment variables
when gated models are used:

```bash
-v /data/.cache:/root/.cache \
-e HF_TOKEN \
-e HUGGINGFACE_HUB_TOKEN
```

Do not print token values into logs.

Set the run variables once and pass them into containers that need them:

```bash
export MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export TP=1
export PP=1
export PORT=8000
export RUN_DIR=/tmp/llm-serving-auto-benchmark
mkdir -p "$RUN_DIR"
```

For synthetic validation, use two aligned scenarios rather than one tiny request
shape:

```bash
# chat-like
RANDOM_INPUT_LEN=1000
RANDOM_OUTPUT_LEN=1000

# summarization-like
RANDOM_INPUT_LEN=8000
RANDOM_OUTPUT_LEN=1000
```

For a fast smoke on larger models, 20 prompts per scenario is a reasonable
minimum. Do not treat that as a performance result.

Set each framework's sequence-length limit to cover the largest scenario. For
the example above, use at least 9000 tokens for SGLang `--context-length`, vLLM
`--max-model-len`, and TensorRT-LLM `--max_seq_len`.

Before launching a server, save the help output:

```bash
python -m sglang.launch_server --help > artifacts/help/sglang_launch_server.txt
python -m sglang.bench_serving --help > artifacts/help/sglang_bench_serving.txt
vllm serve --help=all > artifacts/help/vllm_serve_all.txt
vllm bench serve --help=all > artifacts/help/vllm_bench_serve_all.txt
vllm bench sweep serve --help=all > artifacts/help/vllm_bench_sweep_serve_all.txt
trtllm-serve serve --help > artifacts/help/trtllm_serve.txt
python -m tensorrt_llm.serve.scripts.benchmark_serving --help \
  > artifacts/help/trtllm_benchmark_serving.txt
```

## SGLang

If a prepared GPU host already has a long-running SGLang container (local or
reached via ssh; name is operator-specific), reuse it via `docker exec`
instead of creating a new container. The per-host skills — `h100`,
`h100-sglang-diffusion`, `b200`, `rtx5090`, `radixark02`, `radixark03`,
and similar — provide the concrete container name and workspace path for
that box; this runbook assumes the operator substitutes them:

```bash
docker exec \
  -e MODEL \
  -e TP \
  -e PORT \
  "$SGLANG_CONTAINER" bash -lc "
cd \"\$SGLANG_WORKSPACE\"
python -m sglang.launch_server \\
  --model-path \"\$MODEL\" \\
  --tp-size \"\$TP\" \\
  --host 0.0.0.0 \\
  --port \"\$PORT\"
"
```

For a fresh container:

```bash
docker run -d --name llmbench-sglang \
  --gpus "$GPU_ARG" \
  --network host \
  --ipc=host \
  -v /data/.cache:/root/.cache \
  -e MODEL \
  -e TP \
  -e PORT \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_TOKEN \
  --entrypoint bash \
  lmsysorg/sglang:dev -lc '
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size "$TP" \
  --host 0.0.0.0 \
  --port "$PORT"
'
```

Then run either SGLang auto benchmark:

```bash
python -m sglang.auto_benchmark run --config /path/to/sglang.yaml
```

or a tiny OpenAI-compatible smoke benchmark:

```bash
python -m sglang.bench_serving \
  --backend sglang-oai \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dataset-name random \
  --random-input-len 32 \
  --random-output-len 8 \
  --num-prompts 4 \
  --request-rate 1 \
  --max-concurrency 2 \
  --output-file "$RUN_DIR/sglang/results.json" \
  --output-details
```

## vLLM

Server template:

```bash
docker run -d --name llmbench-vllm \
  --gpus "$GPU_ARG" \
  --network host \
  --ipc=host \
  -v /data/.cache:/root/.cache \
  -e MODEL \
  -e TP \
  -e PORT \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_TOKEN \
  --entrypoint bash \
  vllm/vllm-openai:latest -lc '
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --kv-cache-dtype auto \
  --enable-prefix-caching \
  --trust-remote-code
'
```

Benchmark template:

```bash
docker run --rm \
  --network host \
  -v /data/.cache:/root/.cache \
  -v "$RUN_DIR:/artifacts" \
  -e MODEL \
  -e PORT \
  --entrypoint bash \
  vllm/vllm-openai:latest -lc '
vllm bench serve \
  --backend vllm \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8 \
  --max-concurrency 64 \
  --save-result \
  --result-dir /artifacts/vllm \
  --result-filename results.json
'
```

Use `vllm bench sweep serve` when the target image supports it and the search
can be described with serve/bench parameter JSON files.

## TensorRT-LLM

This skill only supports the TensorRT-LLM PyTorch server backend. Keep
`--backend pytorch` in every `trtllm-serve serve` command. Do not switch the
server to `--backend trt`, an engine path, or any other backend; mark that
candidate unsupported instead.

For single-node multi-GPU TensorRT-LLM containers, keep the IPC, ulimit, shared
memory, and NCCL settings below. In a multi-GPU PyTorch-backend validation
run (captured on an H100 host; the rule is not H100-specific), the server
entered `PyTorchConfig` but failed NCCL allreduce without these container
options; the same model and candidate list passed after adding them. Expect
the same requirement on any single-node multi-GPU target.

Server template:

```bash
docker run -d --name llmbench-trtllm \
  --gpus "$GPU_ARG" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  --network host \
  -v /data/.cache:/root/.cache \
  -e MODEL \
  -e TP \
  -e PP \
  -e PORT \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_TOKEN \
  -e NCCL_IB_DISABLE=1 \
  --entrypoint bash \
  nvcr.io/nvidia/tensorrt-llm/release:latest -lc '
trtllm-serve serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --backend pytorch \
  --tp_size "$TP" \
  --pp_size "$PP" \
  --max_batch_size 64 \
  --max_num_tokens 8192 \
  --max_seq_len 4096 \
  --kv_cache_free_gpu_memory_fraction 0.75 \
  --trust_remote_code
'
```

Benchmark template:

```bash
docker run --rm \
  --network host \
  -v /data/.cache:/root/.cache \
  -v "$RUN_DIR:/artifacts" \
  -e MODEL \
  -e PORT \
  --entrypoint bash \
  nvcr.io/nvidia/tensorrt-llm/release:latest -lc '
python -m tensorrt_llm.serve.scripts.benchmark_serving \
  --backend openai \
  --host 127.0.0.1 \
  --port "$PORT" \
  --endpoint /v1/completions \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-ids \
  --num-prompts 80 \
  --request-rate 8 \
  --max-concurrency 64 \
  --save-result \
  --result-dir /artifacts/trtllm \
  --result-filename results.json
'
```

For TensorRT-LLM 1.0.0, the serving benchmark client `--backend` choices are
`openai` and `openai-chat`. Do not pass `--backend trtllm`. This client flag is
separate from the server backend pinned above.

## Cleanup

Use unique container names per run and clean up by name:

```bash
docker rm -f llmbench-sglang llmbench-vllm llmbench-trtllm
```

If a port remains bound after container cleanup, inspect it before killing
anything:

```bash
ss -ltnp | grep ':8000'
ps -eo pid,ppid,user,etime,cmd | grep '<model-or-port>'
```

Only kill raw PIDs when the command line proves they belong to the current
validation run.
