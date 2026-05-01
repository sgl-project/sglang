# Parameter Coverage

Do not compare SGLang, vLLM, and TensorRT-LLM by copying the same flag names
across frameworks. Compare knob families, then translate each family to the
flags that the target CLI actually accepts.

## Historical Audit Snapshot

The concrete flag counts and captured help output below come from a single
historical audit on a 4xH100 host. They are evidence of what each CLI exposed
in a specific image, not a requirement that the next run uses that hardware.
When you repeat the audit on a different GPU target, re-capture `--help` on
that image and update the numbers.

Captured on 2026-04-22 in:

`/tmp/llm_serving_auto_benchmark_param_audit_20260422_094955`

Versions:

- SGLang: `sglang 0.5.10rc0`, repo commit
  `30cd2cf32b20355e361f7e2b3badc376fc1330af`
- vLLM image: `vllm/vllm-openai:latest`, `vllm 0.19.1`
- TensorRT-LLM image: `nvcr.io/nvidia/tensorrt-llm/release:latest`,
  `tensorrt_llm 1.0.0`

Help files captured:

- `help/sglang_launch_server.txt`
- `help/sglang_bench_serving.txt`
- `help/sglang_auto_benchmark_run.txt`
- `help/vllm_serve_all.txt`
- `help/vllm_bench_serve_all.txt`
- `help/vllm_bench_sweep_serve_all.txt`
- `help/trtllm_serve.txt`
- `help/trtllm_benchmark_serving.txt`
- `help/trtllm_bench.txt`

Observed flag counts in that environment:

| CLI | Flag Count | Note |
| --- | ---: | --- |
| `python -m sglang.launch_server --help` | 384 | Large server flag surface. |
| `vllm serve --help=all` | 302 | Plain `--help` only shows config groups. |
| `vllm bench serve --help=all` | 97 | Covers random, ShareGPT, HF, custom, multimodal, and sampling knobs. |
| `vllm bench sweep serve --help=all` | 15 | Sweeps serve and bench parameter JSON files. |
| `trtllm-serve serve --help` | 23 | Direct serving surface is smaller; this skill fixes `--backend pytorch` and uses direct flags or PyTorch backend config for tuning. |
| `python -m tensorrt_llm.serve.scripts.benchmark_serving --help` | 53 | OpenAI-compatible benchmark client. |

## TensorRT-LLM Backend Policy

For this skill, TensorRT-LLM serving is pinned to `trtllm-serve serve --backend
pytorch`. Keep `backend: pytorch` in `base_server_flags`, and do not put
`backend` in `search_space`.

Reject `trt`, engine-backed serving, or any other non-PyTorch TensorRT-LLM
server backend as unsupported for this skill. The benchmark client `--backend`
is separate: TensorRT-LLM 1.0.0 uses `openai` or `openai-chat` to target the
OpenAI-compatible endpoint.

## Knob Family Mapping

| Family | SGLang | vLLM | TensorRT-LLM |
| --- | --- | --- | --- |
| Parallelism | `--tp-size`, `--pp-size`, `--dp-size`, `--ep-size`, `--expert-parallel-size`, `--attention-context-parallel-size` | `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--decode-context-parallel-size`, `--enable-expert-parallel` | `--tp_size`, `--pp_size`, `--ep_size`, `--gpus_per_node`, `--cluster_size` |
| Memory and KV cache | `--mem-fraction-static`, `--max-total-tokens`, `--kv-cache-dtype`, `--page-size`, `--cpu-offload-gb` | `--gpu-memory-utilization`, `--kv-cache-memory-bytes`, `--kv-cache-dtype`, `--block-size`, `--cpu-offload-gb` | `--kv_cache_free_gpu_memory_fraction`; combine with `--max_num_tokens`, `--max_seq_len`, and `--max_batch_size` |
| Batching and scheduler | `--max-running-requests`, `--max-queued-requests`, `--schedule-policy`, `--schedule-conservativeness`, `--chunked-prefill-size`, `--max-prefill-tokens`, `--prefill-max-requests` | `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-chunked-prefill`, `--max-num-partial-prefills`, `--max-long-partial-prefills`, `--long-prefill-token-threshold`, `--enable-dbo`, `--dbo-prefill-token-threshold`, `--dbo-decode-token-threshold` | `--max_batch_size`, `--max_num_tokens`, `--max_seq_len`; more scheduler details may require `--extra_llm_api_options` |
| Attention/backend | `--attention-backend`, `--prefill-attention-backend`, `--decode-attention-backend`, `--sampling-backend`, `--grammar-backend` | `--attention-backend`, `--gdn-prefill-backend`, `--mm-encoder-attn-backend` | `--backend pytorch` is fixed; do not search backend choice |
| CUDA graph and compile | `--disable-cuda-graph`, `--cuda-graph-bs`, `--cuda-graph-max-bs`, `--disable-piecewise-cuda-graph`, `--piecewise-cuda-graph-max-tokens`, `--enable-torch-compile` | `--enforce-eager`, `--compilation-config`, `--cudagraph-capture-sizes`, `--max-cudagraph-capture-size` | use `--extra_llm_api_options`; server logs expose the resolved `PyTorchConfig` |
| Prefix/speculative | `--disable-radix-cache`, `--disable-chunked-prefix-cache`, `--speculative-algorithm`, `--speculative-draft-model-path`, `--speculative-num-steps`, `--speculative-num-draft-tokens` | `--enable-prefix-caching`, `--speculative-config` | only use PyTorch-backend options accepted by `--extra_llm_api_options` |
| Dtype, quantization, loading | `--dtype`, `--quantization`, `--load-format`, `--model-loader-extra-config`, `--trust-remote-code` | `--dtype`, `--quantization`, `--load-format`, `--model-loader-extra-config`, `--trust-remote-code`, `--hf-token` | `--trust_remote_code`, `--tokenizer`; engine build and non-PyTorch quantization flows are outside this skill |

## Benchmark Client Notes

SGLang `bench_serving` in the validated SGLang container image writes artifacts
with `--output-file` and `--output-details`. It does not accept the vLLM-style
`--save-result`, `--result-dir`, and `--result-filename` flags in that image.
Re-check on the target image; this is an SGLang CLI difference, not an H100
behavior.

Both vLLM and TensorRT-LLM benchmark clients accepted these shared
random-workload flags in the captured audit:

- `--dataset-name`
- `--random-input-len`
- `--random-output-len`
- `--num-prompts`
- `--request-rate`
- `--max-concurrency`
- `--save-result`
- `--result-dir`
- `--result-filename`
- `--ignore-eos`
- `--temperature`
- `--top-p`
- `--top-k`

TensorRT-LLM 1.0.0 additionally needs `--random-ids` for a fast synthetic random
run unless a ShareGPT download path is provided. vLLM 0.19.1 does not have that
flag.

## Search Guidance

- SGLang and vLLM both have large runtime flag surfaces. Give both frameworks a
  sweep over batching, scheduler, CUDA graph/compile, and prefix cache options
  before comparing winners.
- TensorRT-LLM should still be searched, but its direct `trtllm-serve serve`
  surface is narrower. Search the exposed server flags first, then use only
  PyTorch-backend options accepted by `--extra_llm_api_options`.
- Keep every translated flag in the artifact manifest. If a family is not
  represented for one framework, say why instead of filling the gap with an
  unrelated option.

Attention backend hardware notes:

- SGLang's `fa3` (FlashAttention-3) prefill and decode backends require Hopper
  (H100/H200/GB200). On non-Hopper GPUs (A100, L40S, RTX 5090, older cards)
  `fa3` is not available: drop it from the SGLang `search_space` and fall back
  to `flashinfer`, or to `triton` for the non-Blackwell path when FlashInfer is
  missing.
- vLLM `--attention-backend` has a similar shape: `FLASH_ATTN` and `FLASHINFER`
  are the safer defaults, and `FLASHINFER_MLA` or DeepSeek-MLA specific paths
  need a verified driver/vLLM combination in the target image.
- TensorRT-LLM's backend here is always `pytorch`; attention backend choice
  inside `PyTorchConfig` is version- and model-dependent. Prefer the defaults
  emitted in the server log until you have a reason to override them via
  `--extra_llm_api_options`.

Default searches should not sweep memory fractions:

- keep SGLang `mem_fraction_static` in `base_server_flags`
- keep vLLM `gpu_memory_utilization` in `base_server_flags`
- keep TensorRT-LLM `kv_cache_free_gpu_memory_fraction` in `base_server_flags`

Move them into `search_space` only for a fit/capacity study. For a normal serving
comparison, search scheduler, batching, attention/backend, prefix cache, and
CUDA graph or compile behavior first.

In the 2026-04-22 validation run, vLLM accepted `--enable-dbo` but the server
failed during config validation without a supported all2all backend. Treat DBO
as a version- and environment-gated extension knob on any GPU target, not part
of the default candidate list.

The same validation showed that vLLM concurrent partial prefill is model- and
runtime-gated: `--max-num-partial-prefills 1` ran on `Qwen/Qwen3-30B-A3B`,
while `--max-num-partial-prefills 4` failed with "Concurrent Partial Prefill is
not supported." Keep `1` in the default pass and only raise it after a
model-level preflight in the target image.

Sequence-length candidates must cover the largest dataset scenario. In the same
validation run, a TensorRT-LLM candidate with `--max_seq_len 2048` passed the
512-to-64 chat-like scenario but failed the 2048-to-128 summarization-like
scenario. Do not include a `max_model_len`, `context_length`, or `max_seq_len`
candidate below `max(input_len + output_len)` on any target.
