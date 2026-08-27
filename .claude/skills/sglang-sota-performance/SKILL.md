---
name: sglang-sota-performance
description: End-to-end SGLang SOTA performance workflow. Use when a user names an LLM model and wants SGLang to match or beat the best observed vLLM and TensorRT-LLM serving performance by searching each framework's best deployment command, benchmarking them fairly, profiling SGLang if it is slower, identifying kernel/overlap/fusion bottlenecks, patching SGLang code, and revalidating with real model runs.
---

# SGLang SOTA Performance

## Overview

Use this skill as the top-level optimization loop for one model at a time.
It composes two lower-level skills:

- `llm-serving-auto-benchmark`: search and compare best deployment commands across SGLang, vLLM, and TensorRT-LLM.
- `llm-torch-profiler-analysis`: capture or analyze torch-profiler traces and produce kernel, overlap-opportunity, and fuse-pattern tables.

This skill's goal is not "run one benchmark." Its goal is a reproducible
SGLang improvement loop: tune every framework fairly, prove whether SGLang is
behind, explain the gap with profiler evidence, patch SGLang, and re-run the
same model workload until the result is SOTA for the target environment.

Treat "SOTA" as "best observed, reproducible performance under the recorded
model, workload, hardware, framework commits, precision, and SLA." Do not claim
global SOTA without enough external evidence.

## Required Companion Reads

Before a real run, read only the needed sections from:

- `../llm-serving-auto-benchmark/SKILL.md`
- `../llm-torch-profiler-analysis/SKILL.md`

If the run uses a remote GPU host, also read the matching host skill such as
`h100`, `b200`, `rtx5090`, or another operator-side skill that gives SSH,
container, workspace, and artifact-path conventions.

## Required Inputs

Collect or infer these before starting a long search:

- model id or local checkpoint path, tokenizer path, precision, quantization,
  trust-remote-code policy, and max context length
- target GPU type/count, single-node or multi-node allowance, and VRAM budget
- workload distribution: dataset, input/output lengths, request rate or
  concurrency mode, sampling settings, endpoint style, and SLA target
- frameworks to compare: default to SGLang, vLLM, and TensorRT-LLM when all are
  available in the target environment
- artifact root for commands, logs, benchmark JSONL, profiles, analysis reports,
  patches, and final comparison tables

If the user only provides a model, choose a reasonable first workload and state
it explicitly. Prefer the closest cookbook config from
`llm-serving-auto-benchmark/configs/cookbook-llm/` when available.

## Artifact Layout

Use one run directory per model and date, for example:

```text
runs/YYYYMMDD_<model_slug>_sota_loop/
  manifest.txt
  help/
  benchmark/
  profiles/
  analysis/
  patches/
  final_report.md
```

Record exact framework versions, git commits, container names/images, CUDA/NCCL
versions, GPU ids, launch commands, benchmark commands, and environment knobs.
Never write Hugging Face tokens or other secrets into artifacts.

## Workflow

### 1. Preflight The Model And Environment

Verify the model can be loaded by each framework before launching a sweep.
Capture each framework's current `--help` output and version. Remove candidate
flags that are not accepted by that exact environment.

For TensorRT-LLM, keep the server backend within the scope of
`llm-serving-auto-benchmark`: `trtllm-serve serve --backend pytorch`.
If that backend is unavailable, mark TensorRT-LLM unsupported for the run
instead of silently switching to a different serving stack.

### 2. Search Each Framework's Best Command

Use `llm-serving-auto-benchmark` as the source of truth for benchmark fairness,
candidate generation, result schema, and comparison tables.

Run a bounded search for every available framework. Do not compare SGLang's
tuned command against competitor defaults. Each framework must get a real chance
to find its best deployment command under the same:

- model weights and tokenizer
- precision and quantization policy
- GPU type/count and memory budget
- dataset and request distribution
- endpoint path and sampling settings
- SLA target and measurement window

Keep failed candidates and their failure reasons. The fastest SLA-failing
candidate is not the winner.

### 3. Compare The Best Commands

Normalize the benchmark output with
`llm-serving-auto-benchmark/scripts/compare_benchmark_results.py`.

The comparison must include:

- best server command per framework
- benchmark command and workload settings
- SLA pass/fail status
- throughput and goodput
- TTFT, ITL, end-to-end latency, and p95/p99 where available
- peak memory or allocator evidence when available
- failed candidate summary

If SGLang is within benchmark noise of the best framework, rerun enough samples
to decide whether the difference is real. Use a default regression threshold of
3-5% unless the user specifies a tighter target.

### 4. Profile SGLang When It Is Behind

If SGLang is meaningfully slower, fails SLA while another framework passes, or
uses much more memory for the same workload, run profiler triage before patching.

Use `llm-torch-profiler-analysis` against the SGLang best command first:

- capture live SGLang profiles with `--profile-workload both`; the profiler
  skill labels `prefill/` and `decode/` by workload directory for this mode
- keep separate `extend/prefill` and `decode` traces; do not use one mixed
  request as the default profiler workload
- set profiler lengths from the slow benchmark scenario instead of the profiler
  defaults: prefill uses the slow input length with output `1`, and decode uses
  input `1` with the slow output length
- for mixed benchmark datasets, choose the slowest representative bucket
  already reported by the benchmark, usually p50 or p95 input/output lengths,
  and record that bucket beside the profiler artifact path
- run mapping+formal triage if single-trace output cannot map kernels to useful
  Python source locations
- save the kernel, overlap-opportunity, and fuse-pattern tables in artifacts

Profile the winning competitor too when the SGLang table alone cannot explain
why the other framework is faster. Compare stage by stage, not just total QPS.

### 5. Turn Tables Into A Root Cause

Use the profiler tables to identify the narrowest plausible bottleneck.

Typical signals:

- kernel table: attention, MoE routing, quantization, sampling, GEMM shape,
  cache update, communication, or framework overhead dominates GPU time
- overlap-opportunity table: CPU scheduling, host-to-device work, collectives,
  or decode bookkeeping leaves GPU idle time
- fuse-pattern table: a known fusion or overlap path should have applied but did
  not, or competitor traces show a fused path SGLang lacks
- source map: hot kernels map to a concrete SGLang Python/CUDA/Triton path that
  can be patched

Do not patch from vibes. State the table row, stage, source location, and
benchmark symptom that justify the code change.

### 6. Patch SGLang Conservatively

Patch SGLang only after the benchmark gap and profiler evidence agree.

Good patch candidates:

- enable or select a better existing kernel for the model/hardware shape
- fix a missed fast path, fusion, overlap, or batching condition
- reduce unnecessary synchronization, CPU scheduling overhead, or tensor copies
- improve model-specific routing, quantization, attention, or cache handling
- add a guarded heuristic that is backed by benchmark and profiler evidence

Avoid changes that merely make the benchmark easier:

- weakening correctness, output quality, safety checks, or tokenizer handling
- changing only the workload or SLA after seeing results
- disabling features for SGLang but not competitors
- claiming SOTA from synthetic data when the user asked for production traffic

Keep patches minimal and local. Add focused tests when behavior changes, and add
microbenchmarks or profiler evidence when performance is the only intended
change.

### 7. Revalidate The Patch

After patching, rerun:

- the relevant unit or integration tests
- the SGLang candidate that exposed the gap
- the same cross-framework benchmark comparison
- the profiler triage if the original gap was diagnosed from profiler tables

If the patch changes SGLang's available knobs, re-search SGLang's best command.
If competitor versions or commands changed during the work, rerun their best
commands too. Preserve before/after artifacts.

## H100 Validation Snapshot

On 2026-05-01, this workflow was smoke-validated on `h100_sglang` with two
real model runs and two competitor checks per run. Artifacts were saved
under
`/data/bbuf/validate/sglang_sota_performance_skill/runs/20260501_two_model_validation`.

| Model | GPUs | Workload | SGLang result | vLLM check | TensorRT-LLM check |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-7B-Instruct` | 2x H100, TP=2 | random, input 512/output 64, 24 prompts, 10 warmup requests | 52.09 req/s, mean TTFT 144.85 ms, mean ITL 4.91 ms | 51.06 req/s, mean TTFT 159.19 ms, mean ITL 4.85 ms | 49.71 req/s, mean TTFT 177.54 ms, mean ITL 4.77 ms |
| `Qwen/Qwen2.5-32B-Instruct` | 4x H100, TP=4 | random, input 512/output 64, 16 prompts, 10 warmup requests | 18.47 req/s, mean TTFT 247.06 ms, mean ITL 9.66 ms | 18.78 req/s, mean TTFT 218.68 ms, mean ITL 9.98 ms | 15.48 req/s, mean TTFT 445.62 ms, mean ITL 9.27 ms |

Use this only as a workflow health check, not as a universal performance
claim. The TensorRT-LLM checks used `trtllm-serve serve --backend pytorch` and
the same OpenAI-compatible random workload.

Additional 2-card validation on 2026-05-01 exercised the full handoff from
bounded cross-framework search into SGLang stage-separated profiling. The
benchmark workload was random input `512`, output `64`, 8 prompts, and the
profiler used the same slow-workload lengths: prefill `512->1` and decode
`1->64`, with warmup 10 and capture 5.

| Model | GPUs | Best SGLang | Best vLLM | Profiler result | Artifact root |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3-8B` | 2x H100, TP=2 | `sglang_mem086`, 21.64 req/s | `vllm_mem080`, 22.88 req/s | kernel, overlap, and fuse tables rendered with separate `extend/prefill` and `decode` sections | `/data/bbuf/validate/core_skill_validation_20260501/qwen3_8b/sota` |
| `mistralai/Mistral-7B-Instruct-v0.3` | 2x H100, TP=2 | `sglang_mem080`, 24.09 req/s | `vllm_mem090`, 24.76 req/s | kernel, overlap, and fuse tables rendered with separate `extend/prefill` and `decode` sections | `/data/bbuf/validate/core_skill_validation_20260501/mistral_7b_instruct_v03/sota` |

## Stop Conditions

Stop with a clear report when any of these is true:

- SGLang is the best SLA-passing framework for the target workload
- SGLang is within noise of the best framework and the remaining gap is not
  statistically stable
- SGLang remains behind but the root cause is external to SGLang, such as missing
  model weights, unavailable backend dependencies, or an unsupported hardware
  feature
- a patch improves SGLang but still does not reach SOTA; report the next table
  row or source path to investigate

## Final Report Contract

Return a compact report with:

- model, hardware, framework versions, workload, and artifact root
- best deployment command per framework
- benchmark comparison table before patch and after patch
- SGLang gap analysis, including exact profiler table rows and source paths
- patch summary with changed files and correctness tests
- real-model validation result and whether SGLang reached target-environment SOTA

If no code patch was needed, say why and include the benchmark evidence.
If a patch was attempted but not enough, be explicit about the remaining gap.
