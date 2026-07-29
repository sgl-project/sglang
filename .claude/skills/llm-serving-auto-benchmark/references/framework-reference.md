# Framework Reference

Use this file when choosing native framework commands or translating tuning
knobs across SGLang, vLLM, and TensorRT-LLM. Always verify the concrete CLI in
the target container with `--help` before a long run.

## Native Entry Points

| Framework | Server | Benchmark | Notes |
| --- | --- | --- | --- |
| SGLang | `python -m sglang.launch_server` | `python -m sglang.auto_benchmark` or `python -m sglang.bench_serving` | Use `auto_benchmark` when available for server-flag search. Use `bench_serving` for direct native or OpenAI-compatible endpoint checks. |
| vLLM | `vllm serve` | `vllm bench sweep serve` or `vllm bench serve` | Prefer `bench sweep serve` when sweeping server and benchmark parameter JSON files. |
| TensorRT-LLM | `trtllm-serve serve --backend pytorch` | TensorRT-LLM serving benchmark client or a common OpenAI-compatible client | This skill does not cover engine-backed serving or non-PyTorch server backends. |

Common source docs:

- SGLang bench serving: <https://docs.sglang.ai/developer_guide/bench_serving.html>
- vLLM benchmark sweeps: <https://docs.vllm.ai/en/latest/benchmarking/sweeps/>
- vLLM `bench sweep serve`: <https://docs.vllm.ai/en/latest/cli/bench/sweep/serve.html>
- TensorRT-LLM `trtllm-serve`: <https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html>
- TensorRT-LLM deployment guide: <https://nvidia.github.io/TensorRT-LLM/deployment-guide/index.html>

## Command Templates

### SGLang

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --tp-size <tp> \
  --port 30000

python -m sglang.bench_serving \
  --backend sglang-oai \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8
```

Use `--backend sglang` for SGLang-native `/generate` checks. Use
`--backend sglang-oai` when comparing against vLLM or TensorRT-LLM through an
OpenAI-compatible path.

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
  --num-prompts 80
```

### TensorRT-LLM

```bash
trtllm-serve serve <model> \
  --backend pytorch \
  --tp_size <tp> \
  --kv_cache_free_gpu_memory_fraction 0.75 \
  --host 0.0.0.0 \
  --port 8000
```

Benchmark the OpenAI-compatible endpoint with the TensorRT-LLM serving benchmark
client or the same OpenAI-compatible client used for the other frameworks. Keep
server backend choice fixed to `pytorch`.

## Knob Family Mapping

Do not copy flag names across frameworks. Compare knob families, then translate
to the target CLI.

| Family | SGLang | vLLM | TensorRT-LLM |
| --- | --- | --- | --- |
| Parallelism | `--tp-size`, `--pp-size`, `--dp-size`, `--ep-size`, `--expert-parallel-size` | `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--enable-expert-parallel` | `--tp_size`, `--pp_size`, `--ep_size`, `--gpus_per_node`, `--cluster_size` |
| Memory and KV cache | `--mem-fraction-static`, `--max-total-tokens`, `--kv-cache-dtype`, `--page-size`, `--cpu-offload-gb` | `--gpu-memory-utilization`, `--kv-cache-memory-bytes`, `--kv-cache-dtype`, `--block-size`, `--cpu-offload-gb` | `--kv_cache_free_gpu_memory_fraction`, plus `--max_num_tokens`, `--max_seq_len`, `--max_batch_size` |
| Batching and scheduler | `--max-running-requests`, `--schedule-policy`, `--chunked-prefill-size`, `--max-prefill-tokens`, `--prefill-max-requests` | `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-chunked-prefill`, partial-prefill and DBO flags | `--max_batch_size`, `--max_num_tokens`, `--max_seq_len`; extra scheduler knobs may require `--extra_llm_api_options` |
| Attention/backend | `--attention-backend`, `--prefill-attention-backend`, `--decode-attention-backend`, `--sampling-backend` | `--attention-backend`, `--gdn-prefill-backend`, `--mm-encoder-attn-backend` | `--backend pytorch` is fixed; do not search backend choice |
| CUDA graph and compile | `--disable-cuda-graph`, `--cuda-graph-bs`, `--cuda-graph-max-bs`, `--disable-piecewise-cuda-graph`, `--enable-torch-compile` | `--enforce-eager`, `--compilation-config`, `--cudagraph-capture-sizes`, `--max-cudagraph-capture-size` | use direct flags or `--extra_llm_api_options`; record resolved PyTorch config from logs |
| Prefix/speculative | `--disable-radix-cache`, `--disable-chunked-prefix-cache`, speculative decoding flags | `--enable-prefix-caching`, `--speculative-config` | only use PyTorch-backend options accepted by the target image |
| Dtype, quantization, loading | `--dtype`, `--quantization`, `--load-format`, `--model-loader-extra-config`, `--trust-remote-code` | `--dtype`, `--quantization`, `--load-format`, `--model-loader-extra-config`, `--trust-remote-code`, `--hf-token` | `--trust_remote_code`, `--tokenizer`; engine build and non-PyTorch quantization flows are out of scope |

## Version Rules

Framework CLIs move quickly. For every real run:

1. Record the framework package version, git commit, image tag, and help files.
2. Validate concrete flags with
   `scripts/validate_cookbook_configs.py --help-dir <artifact-help-dir>`.
3. Move renamed or removed flags out of the run plan before benchmarking.
4. Record which frameworks were model-smoked and which only passed preflight.

Historical validation from April 2026 used SGLang `0.5.10rc0`, vLLM `0.19.1`,
and TensorRT-LLM `1.0.0`. Treat those notes as old evidence, not as current
compatibility guarantees.
