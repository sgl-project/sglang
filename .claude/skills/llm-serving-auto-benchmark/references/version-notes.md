# Version Notes

This skill intentionally treats framework knobs as version-sensitive. Serving
CLIs move quickly, so a future run must capture a fresh version manifest instead
of trusting the examples blindly.

## Authored Snapshot

Last updated: 2026-04-28.

## 2026-04-28 Release Freshness Check

The H100 validation notes below are still useful as reproduced command evidence,
but they are no longer the newest upstream releases:

| Framework | Current release signal checked on 2026-04-28 | Impact |
| --- | --- | --- |
| SGLang | GitHub release `v0.5.10.post1` from 2026-04-09 | Historical `0.5.10rc0` smoke notes should be treated as pre-release validation. |
| vLLM | GitHub release `v0.20.0` from 2026-04-27 | vLLM moved past the `0.19.1` image used for the original smoke run; re-capture grouped `--help=all` output before using old search-space assumptions. |
| TensorRT-LLM | GitHub latest stable `v1.2.1` from 2026-04-20 and `v1.3.0rc12` pre-release from 2026-04-17 | The `tensorrt_llm 1.0.0` notes are historical. Re-check server flags, benchmark-client backend names, and PyTorch-backend profiling behavior on the target image. |

Source pages:

- <https://github.com/sgl-project/sglang/releases/tag/v0.5.10.post1>
- <https://github.com/vllm-project/vllm/releases/tag/v0.20.0>
- <https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1>
- <https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc12>

The initial skill text was informed by these local/source snapshots:

| Framework | Snapshot | Notes |
| --- | --- | --- |
| SGLang | local checkout `7044d5fe7`; H100 container package `sglang 0.5.10rc0` at repo commit `30cd2cf32` | SGLang `bench_serving` help was checked in the H100 validation container. This is now a historical pre-release snapshot. |
| vLLM | local checkout `ed2f282bc`; H100 image `vllm/vllm-openai:latest` with `vllm 0.19.1`; official docs for `vllm bench sweep serve` and benchmark sweeps | `vllm serve` and `vllm bench serve` were smoke-tested in the H100 validation image. Re-check against `v0.20.0` or the target image before reuse. |
| TensorRT-LLM | H100 image `nvcr.io/nvidia/tensorrt-llm/release:latest` with `tensorrt_llm 1.0.0`; official `trtllm-serve` and serving benchmark docs current on 2026-04-22 | `trtllm-serve serve --backend pytorch` and `tensorrt_llm.serve.scripts.benchmark_serving` were smoke-tested in the H100 validation image. Non-PyTorch server backends are out of scope for this skill. Re-check against `v1.2.1` or the target image before reuse. |

## Update Rule

When updating this skill:

1. Refresh this table with the exact source commit, package version, or docs
   version used.
2. Re-run `--help` for each server and benchmark CLI in the target environment.
3. Move renamed or removed flags out of the example run plan.
4. Record which frameworks were actually smoke-tested and which were only
   preflighted.

## H100 Validation Notes

On 2026-04-22, validation used the `h100_sglang` host:

- `sglang_bbuf` container: `sglang 0.5.10rc0`.
- `vllm/vllm-openai:latest` image: `vllm 0.19.1`.
- `nvcr.io/nvidia/tensorrt-llm/release:latest` image: `tensorrt_llm 1.0.0`.
- Hugging Face access came from the local H100 skill and was passed as
  environment variables. Do not print the token into logs.
- TensorRT-LLM serving is pinned to `trtllm-serve serve --backend pytorch` in
  this skill. Engine-backed and other non-PyTorch TensorRT-LLM server backends
  are intentionally rejected instead of searched.

### Parameter Audit

On 2026-04-22, a separate H100 parameter audit captured help output and ran
model-level smoke checks for expanded vLLM and TensorRT-LLM server flags.

Audit directory:

`/tmp/llm_serving_auto_benchmark_param_audit_20260422_094955`

Captured help files include:

- `help/sglang_launch_server.txt`
- `help/sglang_bench_serving.txt`
- `help/sglang_auto_benchmark_run.txt`
- `help/vllm_serve_all.txt`
- `help/vllm_bench_serve_all.txt`
- `help/vllm_bench_sweep_serve_all.txt`
- `help/trtllm_serve.txt`
- `help/trtllm_benchmark_serving.txt`

Flag-count summary from the captured help:

| CLI | Flags |
| --- | ---: |
| `python -m sglang.launch_server --help` | 384 |
| `vllm serve --help=all` | 302 |
| `vllm bench serve --help=all` | 97 |
| `vllm bench sweep serve --help=all` | 15 |
| `trtllm-serve serve --help` | 23 |
| `python -m tensorrt_llm.serve.scripts.benchmark_serving --help` | 53 |

GPUs 6 and 7 were not idle during this later audit, so the model-level parameter
smoke used idle GPU 4 and is labeled as a one-GPU flow check.

| Framework | Model | Expanded flags checked | Result |
| --- | --- | --- | --- |
| vLLM 0.19.1 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `--tensor-parallel-size 1`, `--gpu-memory-utilization 0.65`, `--max-model-len 2048`, `--max-num-seqs 16`, `--max-num-batched-tokens 4096`, `--enable-chunked-prefill`, `--kv-cache-dtype auto`, `--block-size 16`, `--enable-prefix-caching`, `--enforce-eager`, `--trust-remote-code` | pass: `model_smoke/vllm/results.json`; completed 4 requests |
| TensorRT-LLM 1.0.0 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `--backend pytorch`, `--tp_size 1`, `--pp_size 1`, `--max_batch_size 8`, `--max_num_tokens 4096`, `--max_seq_len 2048`, `--kv_cache_free_gpu_memory_fraction 0.65`, `--trust_remote_code` | pass: `model_smoke/trtllm/results.json`; completed 4 requests |

Audit lessons:

- vLLM has a large serving flag surface, but full discovery
  requires `vllm serve --help=all` and `vllm bench serve --help=all`.
- TensorRT-LLM's direct `trtllm-serve serve` flag surface is smaller. The server
  logs show deeper PyTorch backend settings in `PyTorchConfig`, but many of
  those settings are not top-level CLI flags.
- TensorRT-LLM 1.0.0 benchmark serving uses `--backend openai` or
  `--backend openai-chat`. `--backend trtllm` is rejected.
- The vLLM server log recorded the expanded non-default args exactly; the
  TensorRT-LLM server log recorded `max_seq_len`, `max_num_tokens`,
  `max_batch_size`, and the KV-cache fraction behavior.

The first smoke attempt for `Qwen/Qwen2.5-0.5B-Instruct` and
`Qwen/Qwen2.5-1.5B-Instruct` had a bad SGLang artifact: the generated config was
empty, so `python -m sglang.auto_benchmark run` failed before launching a real
candidate. Do not count those SGLang cells as validated. The later runs below
supersede that note.

### Cookbook Config Read Check

On 2026-04-23, the reusable cookbook config set was checked on `h100_sglang`
without launching model servers.

Remote help directory:

`/tmp/llm_serving_cookbook_config_help_20260423`

Local validation command:

```bash
python .claude/skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py \
  --help-dir /tmp/llm_serving_cookbook_config_help_20260423 \
  .claude/skills/llm-serving-auto-benchmark/configs/cookbook-llm
```

Result: all 38 cookbook-derived configs loaded, generated bounded candidate
server commands, and passed concrete server-flag validation against the captured
help files. This check did not start model servers.

Captured server help:

| CLI | Help file | Flags |
| --- | --- | ---: |
| `python -m sglang.launch_server --help` | `sglang_launch_server.txt` | 384 |
| `vllm serve --help=all` | `vllm_serve_all.txt` | 302 |
| `trtllm-serve serve --help` | `trtllm_serve.txt` | 23 |

### Validated Smoke Matrix

All rows used a tiny random workload: 4 prompts, 32 input tokens, 8 output
tokens, request rate 1, and max concurrency 2. These are flow checks, not
performance numbers.

| Model | GPU shape | SGLang auto benchmark | vLLM | TensorRT-LLM |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-7B-Instruct` | GPUs 6,7; TP=2 | pass: `/tmp/llm_serving_auto_benchmark_three_models_20260422_091958/qwen25_7b/sglang_auto/container_results/live_results.jsonl` | pass: `vllm serve` plus `vllm bench serve` | pass after using `--kv_cache_free_gpu_memory_fraction`: `/tmp/llm_serving_auto_benchmark_trt_retry_20260422_092852/qwen25_7b/results.json` |
| `google/gemma-3-4b-it` | GPUs 6,7; TP=2 for SGLang/vLLM | pass: `/tmp/llm_serving_auto_benchmark_three_models_20260422_091958/gemma3_4b/sglang_auto/container_results/live_results.jsonl` | pass: `vllm serve` plus `vllm bench serve` | fail in TensorRT-LLM 1.0.0: server container exited during startup, including a TP=1 retry on GPU 6 |
| `mistralai/Ministral-3-8B-Instruct-2512` | GPUs 6,7; TP=2 for SGLang/vLLM | pass: `/tmp/llm_serving_auto_benchmark_three_models_20260422_091958/ministral3_8b/sglang_auto/container_results/live_results.jsonl` | pass: `vllm serve` plus `vllm bench serve` | fail in TensorRT-LLM 1.0.0: server container exited during startup, including a TP=1 retry on GPU 6 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | GPU 6 or 7; TP=1 | pass after disabling piecewise CUDA graph: `/tmp/llm_serving_auto_benchmark_tinyllama_sglang_fixed/container_results/live_results.jsonl` | pass: `/tmp/llm_serving_auto_benchmark_more_gpu7_20260422_091149/tinyllama_1_1b/vllm/results.json` | pass: `/tmp/llm_serving_auto_benchmark_tinyllama_trt_manual/results.json` |

### Validation Lessons

- SGLang smoke configs must be real files. After writing a config through
  `docker exec`, read or checksum it before starting a long run.
- For quick SGLang flow checks, add both `disable_cuda_graph: true` and
  `disable_piecewise_cuda_graph: true`. TinyLlama hit a piecewise CUDA graph
  warmup failure without the second flag.
- In TensorRT-LLM 1.0.0, `trtllm-serve serve` accepts
  `--kv_cache_free_gpu_memory_fraction`; `--free_gpu_memory_fraction` exits with
  a CLI error.
- `benchmark_serving --dataset-name random` needs `--random-ids` for fast
  synthetic tests unless a ShareGPT `--download-path` is provided.
- When using Docker with specific GPUs, quote comma-separated device lists:
  `--gpus '"device=6,7"'`. The unquoted `--gpus device=6,7` form can fail before
  the container starts.
- Check GPU idleness before every framework run, not only at the start. Other
  jobs can appear mid-validation. If the GPU count changes, record the run as a
  smoke-only check and do not compare throughput against the earlier runs.
- Clean up by port and container name. Avoid killing raw PIDs unless they are
  proven to belong to the current validation run.

### Larger Model Search Smoke

On 2026-04-22, a larger H100 flow check used idle GPU 4 on `h100_sglang`. Other
H100s were occupied, so these runs are not throughput comparisons against a
4-GPU deployment. They are end-to-end checks that the search layout, framework
commands, cleanup, and result collection work on larger models that fit within a
4-H100 target budget.

Common workload:

- random `chat`: 512 input tokens, 64 output tokens
- random `summarization`: 2048 input tokens, 128 output tokens
- 20 prompts per scenario
- request rate 1, max concurrency 4
- 10 server candidates per framework

Artifact directories:

- `/tmp/llm_serving_auto_benchmark_big_models_20260422_101647`
- `/tmp/llm_serving_auto_benchmark_big_models_20260422_110906`

Both directories contain `summary.tsv`, per-candidate server logs, benchmark
logs, and successful `results.json` files. The second directory also contains
`metrics.tsv`; the first directory was interrupted after `Qwen/Qwen3-14B` and
`metrics.tsv` was generated afterwards from the saved result JSON files.

Scenario-level result counts:

| Model | Framework | Pass | Fail | Notes |
| --- | --- | ---: | ---: | --- |
| `Qwen/Qwen3-14B` | SGLang | 16 | 2 | Two startup failures from searched SGLang candidates. |
| `Qwen/Qwen3-14B` | vLLM | 16 | 2 | DBO candidates failed without a supported all2all backend. |
| `Qwen/Qwen3-14B` | TensorRT-LLM | 16 | 4 | `max_seq_len 2048` candidates passed chat but failed summarization. |
| `Qwen/Qwen3-30B-A3B` | SGLang | 16 | 2 | Same SGLang candidate pattern as `Qwen3-14B`. |
| `Qwen/Qwen3-30B-A3B` | vLLM | 18 | 1 | `max_num_partial_prefills 4` failed; `1` passed. |
| `Qwen/Qwen3-30B-A3B` | TensorRT-LLM | 20 | 0 | All corrected TensorRT-LLM candidates passed. |
| `mistralai/Ministral-3-14B-Instruct-2512` | SGLang | 20 | 0 | All candidates passed. |
| `mistralai/Ministral-3-14B-Instruct-2512` | vLLM | 18 | 1 | `max_num_partial_prefills 4` failed. |
| `mistralai/Ministral-3-14B-Instruct-2512` | TensorRT-LLM | 0 | 10 | TensorRT-LLM 1.0.0 failed startup with `KeyError: 'ministral3'`. |

Lessons from the larger run:

- SGLang `bench_serving` in this container needs `--output-file` and
  `--output-details`; the vLLM-style `--save-result` flags are not accepted.
- Keep DBO out of the default vLLM search unless the environment has the
  required all2all backend.
- Keep vLLM concurrent partial prefill at `max_num_partial_prefills: 1` by
  default. Raising it to 4 failed on the tested larger models.
- TensorRT-LLM `max_seq_len` candidates must cover the largest input plus output
  length in the dataset. The too-small `2048` candidates failed only on the
  longer scenario.
- TensorRT-LLM model support is version-specific. In the tested 1.0.0 image,
  `Ministral-3-14B-Instruct-2512` failed before serving because the bundled
  Transformers mapping did not include `ministral3`.
- Cleanup used `codex-big-*` container names and left GPU 4 idle at the end.

### TensorRT-LLM PyTorch Backend 4-GPU Validation

After pinning TensorRT-LLM to the PyTorch server backend, a focused H100 run
validated the container path on four idle H100s.

Run directory:

`/tmp/llm_serving_auto_benchmark_trt_torch4_10cfg_20260422_131031`

Setup:

- model: `Qwen/Qwen3-30B-A3B`
- GPUs: 0, 1, 2, 3 on `h100_sglang`
- image: `nvcr.io/nvidia/tensorrt-llm/release:latest`
- TensorRT-LLM: `1.0.0`
- server backend: `trtllm-serve serve --backend pytorch`
- workload: random `chat` 512 -> 64 and random `summarization` 2048 -> 128
- request count: 20 per scenario
- request rate: 1
- max concurrency: 4
- candidates: 10 TensorRT-LLM server configs over `max_batch_size`,
  `max_num_tokens`, and `max_seq_len`

The passing Docker server options included:

```bash
--ipc=host
--ulimit memlock=-1
--ulimit stack=67108864
--shm-size=16g
-e NCCL_IB_DISABLE=1
```

Result summary:

| Candidates | Scenario cells | Pass | Fail |
| ---: | ---: | ---: | ---: |
| 10 | 20 | 20 | 0 |

Candidate grid:

| Candidate | `max_batch_size` | `max_num_tokens` | `max_seq_len` |
| ---: | ---: | ---: | ---: |
| 1 | 64 | 12288 | 4096 |
| 2 | 64 | 16384 | 4096 |
| 3 | 128 | 12288 | 4096 |
| 4 | 128 | 16384 | 4096 |
| 5 | 64 | 12288 | 8192 |
| 6 | 64 | 16384 | 8192 |
| 7 | 128 | 12288 | 8192 |
| 8 | 128 | 16384 | 8192 |
| 9 | 32 | 8192 | 4096 |
| 10 | 32 | 8192 | 8192 |

Representative metrics:

| Candidate | Scenario | Completed | Request/s | Output tok/s | p99 TTFT ms | p99 TPOT ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | chat | 20 | 0.879 | 56.237 | 176.222 | 12.669 |
| 1 | summarization | 20 | 0.879 | 112.483 | 70.347 | 6.889 |
| 8 | chat | 20 | 0.878 | 56.221 | 178.766 | 4.998 |
| 8 | summarization | 20 | 0.879 | 112.478 | 62.188 | 4.481 |
| 10 | chat | 20 | 0.879 | 56.228 | 178.633 | 4.960 |
| 10 | summarization | 20 | 0.879 | 112.485 | 68.294 | 4.530 |

An earlier 10-candidate attempt in
`/tmp/llm_serving_auto_benchmark_trt_torch4_20260422_130118` used the same
model and backend but omitted the multi-GPU Docker options above. Each candidate
entered `PyTorchConfig` and then failed startup at NCCL allreduce. Keep those
container options in the runbook for multi-GPU TensorRT-LLM validation.
