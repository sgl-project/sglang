---
name: sglang-torch-profiler-analysis
description: "Unified SGLang torch-profiler skill for trace generation, kernel/category breakdown, two-stage overlap analysis, and small Perfetto trace repair. Use when Codex should inspect an existing `trace.json(.gz)` or profile directory, trigger `sglang.profiler` against a live server, break down prefill/decode GPU time by kernel family, correlate a graph-off mapping trace with a graph-on formal trace to find overlap headroom tied back to Python code, or rewrite a trace so Perfetto renders overlapped events more reliably."
---

# SGLang Torch Profiler Analysis

## Overview

Use this skill for all SGLang torch-profiler work. It replaces the old split between:

- kernel/category breakdown
- overlap-specific diagnosis
- small trace post-processing

Prefer the unified entrypoint:

- [scripts/analyze_sglang_torch_profile.py](scripts/analyze_sglang_torch_profile.py)

This entrypoint exposes four subcommands:

- `triage`: the default compact workflow that prints three main tables

- `breakdown`: one-trace kernel/category share analysis
- `overlap`: required two-trace overlap analysis with source mapping
- `perfetto-fix`: rewrite a trace when Perfetto drops some overlapped lanes

For normal use, prefer `triage`. It already collapses the result into three main tables:

- kernel table
- overlap-opportunity table
- fuse-opportunity table

Internal analyzers live here:

- [scripts/analyze_sglang_llm_torch_profile.py](scripts/analyze_sglang_llm_torch_profile.py)
- [scripts/analyze_sglang_profiler_overlap.py](scripts/analyze_sglang_profiler_overlap.py)
- [scripts/profile_common.py](scripts/profile_common.py)

## When To Use It

- inspect an SGLang torch profiler trace or profile directory
- profile a live SGLang server and immediately analyze the output
- quantify which kernel families dominate prefill or decode
- compare communication, attention, MoE, quantization, norm, or memory share
- map kernels back to Python code paths
- judge whether a kernel still has overlap headroom in production shape
- get a text table and a small ASCII timeline without opening Perfetto first
- repair a trace so Perfetto can render overlapping events more faithfully

Do not use Nsight Systems as the default path for this workflow. This merged skill is torch-profiler-first.

## Main Commands

### 1. Compact triage from existing trace directories

```bash
python3 scripts/analyze_sglang_torch_profile.py triage \
  --mapping-input /path/to/graph_off_profile_dir \
  --formal-input /path/to/graph_on_profile_dir
```

### 2. Compact triage from running servers

```bash
python3 scripts/analyze_sglang_torch_profile.py triage \
  --mapping-url http://127.0.0.1:31025 \
  --formal-url http://127.0.0.1:31026 \
  --num-steps 5 \
  --profile-by-stage
```

### 3. Breakdown from an existing trace or profile dir

```bash
python3 scripts/analyze_sglang_torch_profile.py breakdown \
  --input /path/to/profile_dir
```

### 4. Breakdown from a running server

```bash
python3 scripts/analyze_sglang_torch_profile.py breakdown \
  --url http://127.0.0.1:30000 \
  --num-steps 5 \
  --profile-by-stage \
  --table-only
```

### 5. Two-stage overlap analysis

```bash
python3 scripts/analyze_sglang_torch_profile.py overlap \
  --mapping-input /path/to/graph_off_profile_dir \
  --formal-input /path/to/graph_on_profile_dir \
  --table-only
```

Or profile both servers directly:

```bash
python3 scripts/analyze_sglang_torch_profile.py overlap \
  --mapping-url http://127.0.0.1:31025 \
  --formal-url http://127.0.0.1:31026 \
  --num-steps 5
```

### 6. Perfetto-friendly trace rewrite

```bash
python3 scripts/analyze_sglang_torch_profile.py perfetto-fix \
  --input /path/to/trace.json.gz
```

This small repair step is inspired by `torch_utils/src/convert_to_perfetto_compatible/convert_to_perfetto_compatible.py`.

## `profile_by_stage`

`profile_by_stage` is not only for PD disaggregation.

- On ordinary non-PD serving, it is still useful because prefill and decode usually have very different bottlenecks.
- On the current profile-v2 path inside SGLang, stage-based profiling is effectively the normal path.
- PD-disaggregated serving adds one extra rule: prefill workers and decode workers must be profiled separately. That is stricter than ordinary `profile_by_stage`.

## Which Mode To Choose

### `triage`

Use when you want the lowest-friction output:

- one kernel table
- one overlap-opportunity table
- one fuse-opportunity table
- optional stage-aware rows when the trace directory includes both `EXTEND` and `DECODE`

This is the recommended default for final user-facing reports.

### `breakdown`

Use when you need:

- category share such as attention, communication, MoE, norm, quantize, memory
- top kernels by cumulative GPU time
- stage-aware prefill vs decode summaries
- kernel tables keep full kernel names and full Python locations, already joined with CPU ops
- conservative source-backed fusion opportunities

This mode works with one trace. A graph-off pre-pass plus `--kernel-map` is optional but recommended for the final polished report.

### `overlap`

Use when you need:

- a strong answer about which code paths still have overlap headroom
- a table that says which kernels are already hidden and low ROI, with full kernel names and Python scopes
- dependency-risk hints near adjacent kernels
- an ASCII timeline around the most actionable windows

This mode requires two traces for a final answer:

1. mapping trace with `--disable-cuda-graph --disable-piecewise-cuda-graph`
2. formal trace with the real serving optimizations enabled

Do not call the mapping pass a "fast profile". It exists to recover `kernel -> cpu_op -> python scope`.

### `perfetto-fix`

Use only when Perfetto fails to render obviously overlapping events cleanly. It is a post-processing utility, not the main analysis flow.

## Workflow

### One-trace breakdown workflow

1. If the user only wants kernel/category share, one trace is enough.
2. Prefer rank-local `TP-0` traces over merged traces.
3. For a live server, this skill can call `sglang.profiler` and automatically send a small probe request.
4. Prefer `--profile-by-stage` even on standard serving unless the user explicitly wants an all-stage mixed trace.

### Two-trace overlap workflow

1. Produce a mapping trace first with graph disabled.
2. Produce a formal trace second with graph enabled and the real serving flags kept on.
3. Run `triage` for the compact three-table report, or `overlap` if you also want source context and ASCII timelines.
4. Read the results in this order:
   - kernel table
   - overlap-opportunity table
   - fuse-opportunity table
5. Before calling something a "new" optimization idea, compare the top rows against both [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md) and [references/overlap-catalog.md](references/overlap-catalog.md). Always check the `PR-backed / in-flight` sections too. Prefer reporting:
   - an existing fused or overlap path that should already apply here
   - an existing path that appears disabled, unsupported, or regressed in this trace
   - an upstream PR-backed pattern that already exists but is not merged into the checked-out tree
   - a truly new opportunity only when no catalog entry fits
6. Use the deeper `overlap` report only when you need source context or ASCII timelines beyond the compact three-table artifact.

## References

Load these only when needed:

- [references/source-map.md](references/source-map.md)
  - upstream SGLang profiler entrypoints and trace-writing source paths
- [references/validated-workflows.md](references/validated-workflows.md)
  - validated two-pass examples for real SGLang models
- [references/trace-workflow.md](references/trace-workflow.md)
  - practical guidance for mapping vs formal traces
- [references/heuristics.md](references/heuristics.md)
  - overlap labels, dependency-risk interpretation, and limits
- [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md)
  - mixed source-backed catalog of existing fuse and overlap patterns, including PR-backed / in-flight rows
- [references/overlap-catalog.md](references/overlap-catalog.md)
  - overlap-only lookup table across LLM, VLM, diffusion, disaggregation, HiSparse, and speculative scheduling

## Output Contract

### For `breakdown`

Return:

- trace path
- model/server args when available
- top categories
- top kernels
- one short conclusion about what dominates the run
- any source-backed fusion opportunities worth checking
