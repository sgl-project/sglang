---
name: llm-torch-profiler-analysis
description: "Unified LLM torch-profiler triage skill for `sglang`, `vllm`, and `TensorRT-LLM`. Use it to inspect an existing `trace.json(.gz)` or profile directory, or to drive live profiling against a running server and return one three-table report with kernel, overlap-opportunity, and fuse-pattern tables."
---

# Unified LLM Torch Profiler Analysis

## Overview

Use this skill for `torch.profiler` analysis across:

- `sglang`
- `vllm`
- `TensorRT-LLM`

There is only one public workflow:

- `triage`

Preferred unified entrypoint:

- [scripts/analyze_llm_torch_profile.py](scripts/analyze_llm_torch_profile.py)

Backwards-compatibility shim (kept so older `docker exec ... analyze_sglang_torch_profile.py ...` calls keep working; it just forwards to the unified entrypoint):

- [scripts/analyze_sglang_torch_profile.py](scripts/analyze_sglang_torch_profile.py)

Markdown bundling helper:

- [scripts/render_triage_markdown_bundle.py](scripts/render_triage_markdown_bundle.py)

`triage` always prints the same three tables:

- kernel table
- overlap-opportunity table
- fuse-pattern table

By default, all three tables only render rows at or above `1.0%` cumulative GPU-time share.
Rows below that are hidden by default unless the user asks for a lower cutoff.

Keep the fuse-pattern table source-backed and deterministic.
Do not turn it into a fuzzy matcher.

If exact source-backed matching is weak but a kernel cluster is still close to a known family,
add one short note after the tables with exactly one of:

- `high`
- `medium`
- `low`

## Capability Matrix

| Capability | SGLang | vLLM | TensorRT-LLM |
| --- | --- | --- | --- |
| Existing trace triage | yes | yes | yes |
| Single-trace live capture | yes | yes, if torch profiler is enabled on server | requires profiler control endpoints |
| Two-trace mapping+formal triage | yes | yes | yes |
| Stage-aware live capture | yes | no | no |
| `--profile-prefix` control | yes | usually ignored on HTTP profiler route | usually ignored on HTTP profiler route |

For TensorRT-LLM, live capture only works when the server exposes `/start_profile` and
`/stop_profile`, and when the deployment already provides a shared trace path plus the
required env vars.

## Validation Notes

This unified workflow has been validated with a `4x H100` matrix across SGLang,
vLLM, and TensorRT-LLM. Use these model shapes as representative coverage when
refreshing or extending the skill:

| Model | SGLang | vLLM | TensorRT-LLM | Result |
| --- | --- | --- | --- | --- |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; benchmark probes returned direct, non-empty text |
| `Qwen/Qwen2.5-32B-Instruct` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; benchmark probes returned direct, non-empty text |
| `Qwen/Qwen3-32B` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; vLLM and TensorRT-LLM chat probes often emitted `<think>` prefixes |

To render a validated run into one markdown document:

```bash
python3 scripts/render_triage_markdown_bundle.py \
  --analysis-root /path/to/analysis_root \
  --output /path/to/analysis_bundle.md
```

The bundle groups by model and keeps the three tables for each framework.

Validation notes:

- all three frameworks now render kernel, overlap, and fuse tables with separate `extend/prefill` and `decode` sections when the trace contains a clean stage split
- SGLang live capture is validated and calls the server profiler API directly instead of shelling out to `sglang.profiler`
- SGLang trace flush can lag well beyond a few seconds, so the runner waits longer for artifacts than the earlier implementation
- SGLang kernel-site reconstruction keeps sampling disabled in the mapping path so the optimized parser does not perturb SGLang table output; equality rechecks matched for `Mixtral-8x7B-Instruct-v0.1`, `Qwen3-32B`, and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`
- vLLM live capture requires `--output-dir` to match the server `torch_profiler_dir`; the validated H100 flow uses `--profiler-config {"profiler":"torch","torch_profiler_dir":"..."}` and then drives `/start_profile` and `/stop_profile`
- TensorRT-LLM validation stays on `--backend pytorch`; the H100 flow writes the trace with `TLLM_TORCH_PROFILE_TRACE` and then analyzes the saved trace
- the 2026-04-22 TensorRT-LLM 1.0.0 `py_executor.py` profiler setup still needed a `with_stack=True` override for table-quality Python locations; re-check this on TensorRT-LLM 1.2.1 or any 1.3.x release-candidate image before assuming the override is still required

## When To Use It

- inspect a `torch.profiler` trace or profile directory from `sglang`, `vllm`, or `TensorRT-LLM`
- profile a live serving endpoint and analyze the result
- summarize which kernel families dominate prefill or decode
- map kernels back to Python code paths
- judge whether a code path still leaves overlap opportunity
- check whether an already-known fusion or overlap path should have applied

## Diffusion Backend Gate

For diffusion benchmark or profiling work, only analyze traces produced by the native
SGLang diffusion backend.

If the run that generated the trace logs any of:

- `Falling back to diffusers backend`
- `Using diffusers backend`
- `Loaded diffusers pipeline`

stop the workflow instead of analyzing the trace.
Handle it as a backend-selection issue, not as native-kernel profiler evidence.

## Main Flows

### 1. Single-trace triage from an existing profile dir or trace

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --input /path/to/profile_dir_or_trace.json.gz
```

Use this when one trace is enough.
The overlap table stays conservative in single-trace mode and will tell you when a
mapping/formal pair is needed.

### 2. Single-trace live capture from SGLang

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework sglang \
  --url http://127.0.0.1:30000 \
  --output-dir /tmp/llm-profiler/sglang_profile_live \
  --num-steps 5 \
  --profile-by-stage
```

The script sends `POST /start_profile` to the SGLang server directly.
The script writes `server_args.json`, sends the probe requests after profiling is armed,
and waits longer for trace flush than the earlier implementation.

### 3. Single-trace live capture from vLLM

Launch vLLM with torch profiler enabled, for example:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/llm-profiler/vllm_profile"}'
```

Then run:

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework vllm \
  --url http://127.0.0.1:8000 \
  --output-dir /tmp/llm-profiler/vllm_profile \
  --num-steps 5 \
  --no-profile-by-stage
```

For vLLM, `--output-dir` must point to the same `torch_profiler_dir` the server uses.
The current vLLM profiler config already defaults `torch_profiler_with_stack=true`,
so the runner only needs to set `torch_profiler_dir`.

### 4. Single-trace live capture from TensorRT-LLM

Use this only when the server exposes `POST /start_profile` and `POST /stop_profile`,
and the trace path is shared with the current machine.

Typical env expectations are:

- `TLLM_PROFILE_START_STOP=1`
- `TLLM_TORCH_PROFILE_TRACE=/shared/path/trace.json` or `.json.gz`

Then run:

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework trtllm \
  --url http://127.0.0.1:8000 \
  --output-dir /shared/path \
  --num-steps 5 \
  --no-profile-by-stage
```

If the deployment does not expose the profiler control endpoints, fall back to analyzing
an existing trace instead of trying live capture.

On the current TensorRT-LLM mainline path, `py_executor.py` creates the torch profiler
with `record_shapes=True` and `with_modules=True` but not `with_stack=True`.
For table-quality validation, use the override generator:

```bash
python3 scripts/make_trtllm_py_executor_override.py \
  --source /path/to/original/py_executor.py \
  --output /tmp/llm-profiler/py_executor_with_stack.py
```

The validated TensorRT-LLM flow is:

1. launch `trtllm-serve` with `TLLM_TORCH_PROFILE_TRACE=/shared/path/trace.json`
2. run a few benchmark requests
3. analyze the emitted trace with `--input /shared/path/trace.json`

### 5. Two-trace triage from existing profile dirs or traces

```bash
python3 scripts/analyze_llm_torch_profile.py triage \
  --mapping-input /path/to/graph_off_profile_dir \
  --formal-input /path/to/graph_on_profile_dir
```

Use this when you need stronger overlap attribution and kernel-to-source mapping.

### 6. Two-trace triage from running servers

```bash
python3 scripts/analyze_llm_torch_profile.py triage \
  --framework sglang \
  --mapping-url http://127.0.0.1:31025 \
  --formal-url http://127.0.0.1:31026 \
  --num-steps 5 \
  --profile-by-stage
```

For `vllm` or `TensorRT-LLM`, use the same shape but pass:

- `--framework vllm` or `--framework trtllm`
- `--mapping-output-dir ...`
- `--formal-output-dir ...`
- `--no-profile-by-stage`

## `profile_by_stage`

`--profile-by-stage` is only meaningful on the SGLang live-capture path.

- On ordinary non-PD SGLang serving, it is still useful because prefill and decode usually have very different bottlenecks.
- On the current profile-v2 path inside SGLang, stage-based profiling is effectively the normal path.
- PD-disaggregated serving adds one extra rule: prefill workers and decode workers must be profiled separately. That is stricter than ordinary `profile_by_stage`.
- For `vllm` and `TensorRT-LLM`, disable it with `--no-profile-by-stage`.

## How To Choose The Triage Shape

### Single-trace triage

Use when you want the lowest-friction report:

- one trace is already available
- you mainly want kernel share and fusion clues
- you are comparing two runs side by side by running triage once per trace

Prefer this by default.

### Two-trace triage

Use when you need:

- a stronger overlap answer
- graph-off source mapping plus graph-on final behavior
- more trustworthy overlap recommendations in the middle table

1. mapping trace with graph disabled or with the lower-fusion / more-readable config
2. formal trace with the real serving optimizations enabled

Do not call the mapping pass a "fast profile".
It exists to recover `kernel -> cpu_op -> python scope`.

## Workflow

### Single-trace workflow

1. If the user only wants a diagnosis, one trace is enough.
2. Prefer one-rank traces over merged traces whenever the profiler emitted both.
3. For a live server, let the script drive the profiler only when the framework-specific prerequisites are already met.
4. Prefer SGLang `--profile-by-stage` unless the user explicitly wants an all-stage mixed trace.
5. Create or clean the target trace directory before live capture so the profiler can write artifacts without permission surprises.

### Two-trace workflow

1. Produce a mapping trace first with graph disabled or the lower-fusion configuration.
2. Produce a formal trace second with the real serving optimizations enabled.
3. Run `triage` for the three-table report.
4. Read the results in this order:
   - kernel table
   - overlap-opportunity table
   - fuse-pattern table
5. Before calling something a "new" optimization idea, compare the top rows against both [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md) and [references/overlap-catalog.md](references/overlap-catalog.md). Check mainline rows first, then the `PR-backed / in-flight` sections. Prefer reporting:
   - an existing fused or overlap path that should already apply here
   - an existing path that appears disabled, unsupported, or regressed in this trace
   - an upstream pattern that is mainline elsewhere but missing locally, or still open upstream
   - a truly new opportunity only when no catalog entry fits
6. If no exact pattern fully matches but the trace is still close to a known family, add one flat similarity note after the tables.
   Use `high`, `medium`, or `low` only.
   Base that note on the full pattern shape, not on one kernel name alone.
   Prefer semantic cues such as producer-consumer chain, source locations, CPU op names, TP context, and model-specific structure.
   Do not rewrite the script table itself to include these heuristic judgments.

## References

Load these only when needed:

- [references/source-map.md](references/source-map.md)
  - upstream SGLang profiler entrypoints and trace-writing paths; still most useful for SGLang-specific source follow-up
- [references/heuristics.md](references/heuristics.md)
  - overlap labels, dependency-risk interpretation, and limits
- [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md)
  - mixed source-backed catalog of existing fuse and overlap patterns, including mainline rows plus PR-backed / in-flight rows
- [references/overlap-catalog.md](references/overlap-catalog.md)
  - overlap-only lookup table across LLM, VLM, diffusion, disaggregation, HiSparse, and speculative scheduling

## Output Contract

Return:

- trace path or generated profile path
- framework
- model/server args when available
- kernel table
- overlap-opportunity table
- fuse-pattern table
- optional similarity note with `high` / `medium` / `low` when exact matching is inconclusive
- one short summary of what dominates the run
- whether the overlap read came from single-trace triage or mapping/formal two-trace triage
