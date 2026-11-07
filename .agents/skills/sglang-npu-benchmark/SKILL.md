---
name: sglang-npu-benchmark
description: >
  Config-driven automated benchmarking for SGLang inference on Ascend NPU (Atlas 800T A2/A3).
  Guides the agent through building YAML-configured benchmark sweeps: multi-parameter grid search
  (server flags x workload), SLA threshold filtering, per-cell health gating with HBM release
  verification, gsm8k accuracy gates, variance-aware ranking reports, and cross-run regression
  comparison. Encodes field-tested Ascend NPU pitfalls (CUDA-graph capture sizing, KV pool
  sizing, offline datasets, orphan-process HBM leaks, statistics crashes on non-finite values)
  and proven baseline numbers for Qwen3-8B and Qwen3.6-35B-A3B on a 2-die 910C A3 node.
  Use when benchmarking/tuning SGLang NPU deployments, building performance baselines,
  verifying regressions, or onboarding a new model spec onto Ascend NPU.
---

# SGLang NPU Benchmark (Ascend)

## Overview

A repeatable, config-driven workflow for benchmarking SGLang on Ascend NPU. One YAML file
describes a full sweep: model, server flags, workload grid, SLA thresholds, and run policy.
The sweep expands the grid, runs each cell in an isolated server session, gates on health /
HBM release / accuracy, then produces a GFM ranking report with an SLA pass/fail matrix and
a compatibility matrix (cells that cannot run, with server-log tails).

Reference implementation (config schema, sweep executor, SLA/ranking engine, 57 unit tests):
[`benchmark/ascend_bench/`](../../../benchmark/ascend_bench/) in this repository. This skill
teaches the methodology and pitfalls; use that tooling for execution.

## Trigger Conditions

- User wants to benchmark / performance-test SGLang on Ascend NPU or Atlas 800T A2/A3.
- User wants to find optimal server configs (multi-parameter search) for a model on NPU.
- User wants a performance baseline or regression verification (two-run comparison) for NPU inference.
- User asks about SLA gating, ranking reports, or benchmark methodology for NPU deployments.
- User is onboarding a new model spec onto SGLang NPU and wants validated cookbook configs.

## Prerequisites

- NPU host: Atlas 800T A2/A3 class (e.g. single A3 node = 2 x Ascend 910C die, 64 GB HBM each).
- SGLang with NPU backend installed (`--device npu --attention-backend ascend`); sgl-kernel-npu wheels.
- `npu-smi` available (fix PATH/LD_LIBRARY_PATH in containerized/chroot environments).
- Offline hosts: pre-download the gsm8k eval set and model weights; use `random-ids` dataset
  (see Pitfall P5).

## Input Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model_path` | string | yes | Local path or hub id of the model to benchmark |
| `model_spec` | string | no | Short spec name used in config/report (e.g. `qwen3-8b-bf16`) |
| `tp_size` | int | no | Tensor-parallel size. Must satisfy: weights + activation + KV pool fit in per-die HBM x tp_size |
| `grid` | map | no | Parameter axes to sweep, e.g. `mem-fraction-static: [0.80, 0.85]`, `max-concurrency: [1, 8, 32, 64]` |
| `sla` | map | no | Thresholds, e.g. `p99_ttft_ms: 2000`, `p99_tpot_ms: 100` |
| `repeats` | int | no | Repeats per cell for variance gate (default 3) |
| `run_id` | string | no | Identifier for manifest/resume/regression compare |
| `offline` | bool | no | Host has no internet: forces `random-ids` + local dataset paths |

## Execution Flow

### Step 1: Environment audit (before any sweep)

- `npu-smi info` must list all expected NPU dies with OK health; record the output as the
  provenance baseline (driver version, HBM total per die).
- Record HBM used at idle. This is the baseline for the per-cell HBM release gate (Step 4).
- Record software provenance: sglang version/git sha, CANN version, torch_npu version, driver,
  kernel versions, NPU topology. Every report must carry this — results without provenance
  are unreproducible.

### Step 2: Write the sweep YAML (five sections)

```yaml
model:    { model_path, spec, dtype }
server:   { tp_size, mem_fraction_static, cuda_graph_bs_decode, ... }   # real CLI flags
workload: { dataset_name, isl, osl, max_concurrency axis, num_prompts, request_rate }
sla:      { p99_ttft_ms, p99_tpot_ms, ... }                             # per-metric thresholds
run:      { repeats, timeout_per_cell, hbm_gate, straggler_kill, resume }
```

Rules:
- Every server flag must be validated against the installed sglang source (flag names change
  between minor versions — see Pitfall P1). A text-scan test that greps flags in the sglang
  tree catches this class of bug cheaply.
- The CUDA-graph capture list must cover the grid's maximum concurrency (Pitfall P2).

### Step 3: Dry-run then sweep

- Dry-run: expand the grid, print cell ids, verify every cell parses and flags exist.
  Confirm the cell count before burning NPU hours.
- Per cell: start server in an isolated session -> health poll with timeout -> run the
  benchmark client -> optional gsm8k accuracy gate -> stop server -> **verify HBM released**
  (compare `npu-smi` used-HBM against the Step-1 idle baseline) -> append a record to
  `manifest.jsonl`.
- A failed accuracy gate must NOT discard the cell's perf data: record `accuracy=None`,
  mark the gate failure, keep metrics. Accuracy is an advisory column, not a veto.
- Persist one manifest record per cell (config hash, status, metrics, durations, provenance).
  This enables resume after crashes and cross-run regression compare.

### Step 4: HBM release gate + straggler cleanup

Engine worker processes can outlive the server and hold GBs of HBM (observed: 5-8 GB
residue after killing the server), which poisons the next cell with phantom OOM.

- After each cell, poll used-HBM; if above baseline + margin after a grace period, sweep
  killer: `pkill -9 -f` over engine worker process titles (e.g. `sglang::` prefixed
  scheduler/detokenizer workers and the launch command), then re-verify.
- Never run `pkill -f "<pattern>"` from a shell whose own command line contains the pattern
  (it kills itself; observed exit 255 twice). Put patterns in a script file; use a character
  class (`sglang[.]launch_server`) when the pattern must appear in the caller.

### Step 5: Report — SLA matrix, ranking, compatibility

- **SLA pass/fail matrix**: per cell, per metric, `[ok]/[FAIL]` with observed values.
- **Ranking**: only SLA-passing, variance-clean cells are ranked; sort by output token
  throughput, tie-break on p99 TPOT. Always include a per-card throughput column so results
  are comparable across different tp_size.
- **Variance gate**: cells with CV > 0.15 on any ranked metric (across repeats) are marked
  `unrankable`, not silently averaged in.
- **Compatibility matrix**: cells whose server never became healthy keep a short server-log
  tail in the report. This is deliverable evidence of "config x runnability" on NPU.
- Do not beautify data. Saturated-queue cells failing a TTFT SLA are expected behavior —
  report them as-is and explain in the methodology (Pitfall P7).

### Step 6: Regression compare (two runs, same cell hash)

- Cells are keyed by a hash over config + workload (excluding repeat/SLA/run sections) so
  repeats aggregate and cross-run pairs align.
- The compare report prints per-cell deltas; large deltas on low-concurrency cells are
  often real run-to-run variance, not regressions — check the variance gate before blaming code.

## Common Pitfalls on Ascend NPU (all observed in production sweeps)

See [references/ascend-npu-benchmark-pitfalls.md](references/ascend-npu-benchmark-pitfalls.md)
for the full list with symptoms and fixes. Headlines:

- P1: `--cuda-graph-bs` was removed in 0.5.x (split into `-decode`/`-prefill`); NPU docs may lag.
- P2: CUDA-graph capture list below grid max concurrency -> silent eager decode, ~10x slower.
- P3: `--max-total-tokens` too small for high concurrency -> KV retract, doubled cell time.
- P4: `request_rate: Infinity` is legal in bench JSON and crashes `statistics.stdev` on Python 3.11.
- P5: `random` dataset still downloads an HF tokenizer; offline hosts need `random-ids`.
- P6: Orphan engine workers hold HBM across cells.
- P7: TTFT SLA at saturated concurrency measures queueing, not service — gate TPOT/throughput there.
- P8: NEXTN/MTP speculative decoding draft path may fail to compile on NPU (UB overflow in
  sgl-kernel-npu Triton kernel) — verify the draft path separately from the base model.

## Validated Baselines (Atlas 800I A3, 2 x 910C die, TP=2, single node, PD co-located)

See [references/cookbook-baselines.md](references/cookbook-baselines.md) for full tables.
Headlines (BF16, random-ids 1024/1024, gsm8k gated):

| Spec | Grid | Throughput range | Accuracy |
|---|---|---|---|
| Qwen3-8B | 24 cells, mem-fraction x concurrency | 96 -> 2525 tok/s (conc 1 -> 64); best SLA-clean cell 1720.8 tok/s (860.4/card) | 0.945-0.948 |
| Qwen3.6-35B-A3B (MoE, qwen3_5 GDN) | 18 cells | 393 / 955 / 1310 tok/s at conc 8/32/64; p99 TPOT <= 44.8 ms | 0.630-0.652 |
