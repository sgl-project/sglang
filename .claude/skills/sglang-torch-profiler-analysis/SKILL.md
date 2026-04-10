---
name: sglang-torch-profiler-analysis
description: "Compact SGLang torch-profiler triage skill. Use when Codex should inspect an existing `trace.json(.gz)` or profile directory, trigger `sglang.profiler` against a live server, and return one compact report with kernel, overlap-opportunity, and fuse-pattern tables. Single-trace triage is enough for quick diagnosis; mapping+formal two-trace triage gives stronger overlap conclusions."
---

# SGLang Torch Profiler Analysis

## Overview

Use this skill for SGLang `torch.profiler` analysis.

There is only one public workflow:

- `triage`

Use the unified entrypoint:

- [scripts/analyze_sglang_torch_profile.py](scripts/analyze_sglang_torch_profile.py)

`triage` always prints the same three tables:

- kernel table
- overlap-opportunity table
- fuse-pattern table

By default, all three tables only render rows at or above `1.0%` cumulative GPU-time share.
Treat anything below that as noise unless the user explicitly asks for a lower cutoff.

The script-level fuse-pattern table should stay source-backed and deterministic.
Do not build a fuzzy string-matching engine into the script for typo-tolerance.

If exact/source-backed matching is weak but the agent judges that a cluster of kernels
still looks semantically close to a known pattern, add a short AI note after the table
with one of these labels:

- `high`: very likely the same pattern family; naming drift or minor implementation reshaping is the main uncertainty
- `medium`: several signals line up, but one important piece is still ambiguous
- `low`: weak resemblance only; mention it only if it is still worth a human follow-up

## When To Use It

- inspect an SGLang torch profiler trace or profile directory
- profile a live SGLang server and immediately analyze the output
- summarize which kernel families dominate prefill or decode
- map kernels back to Python code paths
- judge whether a code path still has overlap headroom
- check whether an already-known fusion or overlap path should have applied

## Diffusion Backend Gate

For diffusion benchmark or profiling work, only analyze traces produced by the native
SGLang diffusion backend.

If the run that generated the trace logs any of:
- `Falling back to diffusers backend`
- `Using diffusers backend`
- `Loaded diffusers pipeline`

stop the workflow instead of analyzing the trace. Treat it as a backend-selection issue,
not as valid SGLang diffusion profiler evidence.

## Main Flows

### 1. Single-trace triage from an existing profile dir or trace

```bash
python3 scripts/analyze_sglang_torch_profile.py \
  --input /path/to/profile_dir_or_trace.json.gz
```

Use this when you want the fastest read on kernel share and likely fused-kernel pattern matches.
The overlap table stays conservative in single-trace mode and will tell you when a mapping/formal pair is needed.

### 2. Single-trace triage from a running server

```bash
python3 scripts/analyze_sglang_torch_profile.py \
  --url http://127.0.0.1:30000 \
  --num-steps 5 \
  --profile-by-stage
```

### 3. Two-trace triage from existing profile dirs or traces

```bash
python3 scripts/analyze_sglang_torch_profile.py triage \
  --mapping-input /path/to/graph_off_profile_dir \
  --formal-input /path/to/graph_on_profile_dir
```

Use this when you need stronger overlap conclusions and cleaner kernel-to-source attribution.

### 4. Two-trace triage from running servers

```bash
python3 scripts/analyze_sglang_torch_profile.py triage \
  --mapping-url http://127.0.0.1:31025 \
  --formal-url http://127.0.0.1:31026 \
  --num-steps 5 \
  --profile-by-stage
```

## `profile_by_stage`

`profile_by_stage` is not only for PD disaggregation.

- On ordinary non-PD serving, it is still useful because prefill and decode usually have very different bottlenecks.
- On the current profile-v2 path inside SGLang, stage-based profiling is effectively the normal path.
- PD-disaggregated serving adds one extra rule: prefill workers and decode workers must be profiled separately. That is stricter than ordinary `profile_by_stage`.

## How To Choose The Triage Shape

### Single-trace triage

Use when you want the lowest-friction report:

- one trace is already available
- you mainly want kernel share and fusion clues
- you are comparing two runs side by side by running triage once per trace

This is the recommended default.

### Two-trace triage

Use when you need:

- a stronger answer about overlap headroom
- graph-off source mapping plus graph-on final behavior
- more trustworthy overlap recommendations in the middle table

1. mapping trace with `--disable-cuda-graph --disable-piecewise-cuda-graph`
2. formal trace with the real serving optimizations enabled

Do not call the mapping pass a "fast profile". It exists to recover `kernel -> cpu_op -> python scope`.

## Workflow

### Single-trace workflow

1. If the user only wants a quick diagnosis, one trace is enough.
2. Prefer rank-local `TP-0` traces over merged traces.
3. For a live server, this skill can call `sglang.profiler` and automatically send a small probe request.
4. Prefer `--profile-by-stage` even on standard serving unless the user explicitly wants an all-stage mixed trace.

### Two-trace workflow

1. Produce a mapping trace first with graph disabled.
2. Produce a formal trace second with graph enabled and the real serving flags kept on.
3. Run `triage` for the compact three-table report.
4. Read the results in this order:
   - kernel table
   - overlap-opportunity table
   - fuse-pattern table
5. Before calling something a "new" optimization idea, compare the top rows against both [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md) and [references/overlap-catalog.md](references/overlap-catalog.md). Always check the `PR-backed / in-flight` sections too. Prefer reporting:
   - an existing fused or overlap path that should already apply here
   - an existing path that appears disabled, unsupported, or regressed in this trace
   - an upstream PR-backed pattern that already exists but is not merged into the checked-out tree
   - a truly new opportunity only when no catalog entry fits
6. If no exact pattern fully matches but the trace still looks semantically close to a known family, add one flat `AI similarity judgment` note after the tables.
   Use `high`, `medium`, or `low` only.
   Base that note on the full pattern shape, not on one kernel name alone.
   Prefer semantic cues such as producer-consumer chain, source locations, CPU op names, TP context, and model-specific structure.
   Do not rewrite the script table itself to include these heuristic judgments.

## References

Load these only when needed:

- [references/source-map.md](references/source-map.md)
  - upstream SGLang profiler entrypoints and trace-writing source paths
- [references/heuristics.md](references/heuristics.md)
  - overlap labels, dependency-risk interpretation, and limits
- [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md)
  - mixed source-backed catalog of existing fuse and overlap patterns, including PR-backed / in-flight rows
- [references/overlap-catalog.md](references/overlap-catalog.md)
  - overlap-only lookup table across LLM, VLM, diffusion, disaggregation, HiSparse, and speculative scheduling

## Output Contract

Return:

- trace path or generated profile path
- model/server args when available
- kernel table
- overlap-opportunity table
- fuse-pattern table
- optional `AI similarity judgment` note with `high` / `medium` / `low` when exact matching is inconclusive
- one short conclusion about what dominates the run
- whether the overlap conclusion came from single-trace triage or mapping/formal two-trace triage
