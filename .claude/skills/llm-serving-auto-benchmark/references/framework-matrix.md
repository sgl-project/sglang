# Framework Matrix

Use this table to choose the native runner for each framework. Always verify the
actual CLI in the target container with `--help` before a long run.

| Framework | Server | Benchmark | Notes |
| --- | --- | --- | --- |
| SGLang | `python -m sglang.launch_server` | `python -m sglang.auto_benchmark` or `python -m sglang.bench_serving` | Use `auto_benchmark` when available for tiered server-flag search. `bench_serving` supports native and OpenAI-compatible endpoints. |
| vLLM | `vllm serve` | `vllm bench sweep serve` or `vllm bench serve` | `vllm bench sweep serve` can launch `vllm serve` repeatedly and sweep serve/bench parameter JSON files. |
| TensorRT-LLM | `trtllm-serve serve --backend pytorch` | TensorRT-LLM serving benchmark client or a common OpenAI-compatible benchmark client | This skill pins TensorRT-LLM serving to the PyTorch backend. Non-PyTorch server backends and engine-serving paths are unsupported here. |

For parameter coverage by framework, see
[parameter-coverage.md](parameter-coverage.md). For Docker image pull, launch,
benchmark, and cleanup commands, see
[container-runbook.md](container-runbook.md).
For the reusable cookbook-derived config set and validation workflow, see
[cookbook-configs.md](cookbook-configs.md).

## Source Links

- SGLang Bench Serving Guide: <https://docs.sglang.ai/developer_guide/bench_serving.html>
- vLLM benchmark sweeps: <https://docs.vllm.ai/en/latest/benchmarking/sweeps/>
- vLLM `bench sweep serve` CLI: <https://docs.vllm.ai/en/latest/cli/bench/sweep/serve.html>
- TensorRT-LLM `trtllm-serve`: <https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html>
- TensorRT-LLM model recipes: <https://nvidia.github.io/TensorRT-LLM/deployment-guide/index.html>
- TensorRT-LLM serving benchmark tutorial: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/commands/trtllm-serve/run-benchmark-with-trtllm-serve.html>

## Command Templates

### SGLang

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --tp-size <tp> \
  --port 30000

python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8
```

`sglang.bench_serving` has two SGLang-facing backends:

- `--backend sglang` targets SGLang's native `/generate` endpoint. Use this for
  SGLang-internal comparisons where the native request path is the most direct
  measurement.
- `--backend sglang-oai` targets the OpenAI-compatible endpoint
  (`/v1/completions` or `/v1/chat/completions`). Use this when the cross-framework
  comparison requires an identical OpenAI-compatible request path for every
  framework.

For this skill, prefer `--backend sglang-oai` whenever the same benchmark run
has to compare SGLang against vLLM and TensorR-LLM, and note the backend choice
in the result row's `workload.endpoint`.

### vLLM

```bash
vllm serve <model> \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size <tp> \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8 \
  --max-concurrency 64

vllm bench sweep serve \
  --serve-cmd 'vllm serve <model> --port 8000' \
  --bench-cmd 'vllm bench serve --backend vllm --model <model> --port 8000 --dataset-name random --num-prompts 80' \
  --serve-params vllm_serve_params.json \
  --bench-params vllm_bench_params.json \
  --output-dir vllm_results
```

### TensorRT-LLM

```bash
trtllm-serve serve <model> \
  --backend pytorch \
  --tp_size <tp> \
  --pp_size <pp> \
  --kv_cache_free_gpu_memory_fraction 0.75 \
  --host 0.0.0.0 \
  --port 8000
```

Then benchmark `http://127.0.0.1:8000/v1/completions` or
`http://127.0.0.1:8000/v1/chat/completions` with the TensorRT-LLM serving
benchmark client or the same OpenAI-compatible client used for the other
frameworks. In the historical TensorRT-LLM 1.0.0 validation image, synthetic
random data needed `--random-ids` unless a ShareGPT `--download-path` was also
provided. That image also rejected `--free_gpu_memory_fraction` for
`trtllm-serve serve`; use `--kv_cache_free_gpu_memory_fraction` only after
checking `--help` on the target image. TensorRT-LLM 1.2.1 is the latest stable
GitHub release as of 2026-04-28, with 1.3.0 release candidates also published,
so treat the 1.0.0 notes as historical validation evidence. The 1.0.0 benchmark
client accepted `--backend openai` and `--backend openai-chat`, not
`--backend trtllm`.

Do not replace the server-side `--backend pytorch` with `trt` or an engine
backend in this skill. Treat those requests as unsupported candidates and record
that reason in the result table.

When launching Docker containers on a subset of GPUs, quote a comma-separated
device list:

```bash
docker run --gpus '"device=6,7"' ...
```

`--gpus device=6,7` can be parsed incorrectly by Docker and fail before the
server starts.
