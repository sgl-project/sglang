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
| Stage-separated live workload | yes | yes | yes, with a writable shared trace dir or per-stage host runner |
| `--profile-by-stage` capture | yes | no | no |
| `--profile-prefix` control | yes | usually ignored on HTTP profiler route | usually ignored on HTTP profiler route |

For TensorRT-LLM, live capture only works when the server exposes `/start_profile` and
`/stop_profile`, and when the deployment already provides a shared trace path plus the
required env vars.

## Real H100 Validation

The current reference run is the `4x H100` matrix captured on `2026-04-23` on
`h100_sglang` under:

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3`

Rendered markdown bundle:

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3/h100_large_model_matrix_v3_bundle.md`

Validated model directories:

- `mixtral_8x7b_instruct`
- `qwen2_5_32b_instruct`
- `qwen3_32b`

Each model directory contains:

- `analysis_sglang.txt`
- `analysis_vllm.txt`
- `analysis_trtllm.txt`
- framework-specific trace roots and probe artifacts

Validated matrix:

| Model | SGLang | vLLM | TensorRT-LLM | Result |
| --- | --- | --- | --- | --- |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; benchmark probes returned direct, non-empty text |
| `Qwen/Qwen2.5-32B-Instruct` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; benchmark probes returned direct, non-empty text |
| `Qwen/Qwen3-32B` | `4x H100` | `4x H100` | `4x H100` | three tables rendered correctly on all three frameworks; vLLM and TensorRT-LLM chat probes often emitted `<think>` prefixes |

Use this run as the main H100 reference.
The older `2026-04-22` single-card Qwen3 matrix is still useful for bring-up, but it is
not the default reference anymore.

Stage-separated workload validation captured on `2026-05-01` on `h100_sglang`:

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260501_stage_split_validation`
- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260501_stage_split_validation_large`

Validated models:

| Model | GPU | Workloads | Result |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | generated separate `prefill/*.trace.json.gz` and `decode/*.trace.json.gz`; kernel, overlap, and fuse tables rendered with separate `extend/prefill` and `decode` sections |
| `Qwen/Qwen2.5-1.5B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | generated separate `prefill/*.trace.json.gz` and `decode/*.trace.json.gz`; kernel, overlap, and fuse tables rendered with separate `extend/prefill` and `decode` sections |
| `Qwen/Qwen2.5-7B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | generated separate traces; prefill kernel table captured 28-layer GEMM/FA3/RMSNorm work, decode captured 5-step graph launches, and fuse rows were split by stage |
| `Qwen/Qwen2.5-14B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | generated separate traces; prefill kernel table captured 48-layer GEMM/FA3/RMSNorm work, decode captured 5-step graph launches, and fuse rows were split by stage |
| `Qwen/Qwen3-8B` | `2x H100`, TP=2 | prefill `4090->1`, decode `1->2048`, warmup 10/capture 5 | generated separate prefill/decode traces and all three tables; unique probe prompts avoided prefix-cache pollution in the prefill table |
| `mistralai/Mistral-7B-Instruct-v0.3` | `2x H100`, TP=2 | prefill `4090->1`, decode `1->2048`, warmup 10/capture 5 | generated separate prefill/decode traces and all three tables; server logs showed no repeated-prompt prefix-cache shortcut during the active prefill window |

This validation also covers the compatibility fix for older SGLang profiler
state machines: workload-separated live capture labels stages by output
directory and avoids nesting SGLang's internal `profile_by_stage` state machine
inside each workload. The helper
adds one internal scheduler guard step because SGLang increments `forward_ct`
before checking whether the profiler should stop; without that guard, a
`num_steps=1` prefill capture can stop just before the actual prefill forward.
The 2026-05-01 two-card validation artifacts for the additional models are:

- `/data/bbuf/validate/core_skill_validation_20260501/qwen3_8b/profiler`
- `/data/bbuf/validate/core_skill_validation_20260501/mistral_7b_instruct_v03/profiler`

Checked-in sample outputs:

- `references/validated_outputs/20260422_h100_qwen3_matrix/qwen3_30b_a3b`

To render a validated run into one markdown document:

```bash
python3 scripts/render_triage_markdown_bundle.py \
  --analysis-root /data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3 \
  --output /data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3/h100_large_model_matrix_v3_bundle.md
```

The bundle groups by model and keeps the three tables for each framework.

H100 notes:

- all three frameworks now render kernel, overlap, and fuse tables with separate `extend/prefill` and `decode` sections when the trace contains a clean stage split
- SGLang live capture is validated and calls the server profiler API directly instead of shelling out to `sglang.profiler`
- SGLang trace flush can lag well beyond a few seconds, so the runner waits longer for artifacts than the earlier implementation
- SGLang kernel-site reconstruction keeps sampling disabled in the mapping path so the optimized parser does not perturb SGLang table output; equality rechecks matched for `Mixtral-8x7B-Instruct-v0.1`, `Qwen3-32B`, and `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`
- vLLM live capture requires `--output-dir` to match the server `torch_profiler_dir`; the validated H100 flow uses `--profiler-config {"profiler":"torch","torch_profiler_dir":"..."}` and then drives `/start_profile` and `/stop_profile`
- TensorRT-LLM validation stays on `--backend pytorch`; the H100 flow writes the trace with `TLLM_TORCH_PROFILE_TRACE` and then analyzes the saved trace
- the 2026-04-22 TensorRT-LLM 1.0.0 `py_executor.py` profiler setup still needed a `with_stack=True` override for table-quality Python locations, and the matrix runner generated that override under `/data/bbuf/validate/unified_llm_profiler_skill/overrides/trtllm`; re-check this on TensorRT-LLM 1.2.1 or any 1.3.x release-candidate image before assuming the override is still required
- on this host, keep all trace roots under `/data/...`, not `/home/...`

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

## Stage-Separated Live Capture Contract

Live capture must not use one mixed prompt as the default.
By default, `analyze_llm_torch_profile.py --url ...` captures two labeled
workloads and then renders the same three tables with separate stage sections:

- prefill: synthetic input length `4090`, output length `1`
- decode: synthetic input length `1`, output length `2048`

Every live profiler path warms up `10` steps before arming the profiler and then
captures `5` active steps by default. Keep this warmup/active split aligned
across SGLang, vLLM, and TensorRT-LLM before comparing kernel tables.

Use these options to override the contract when the benchmark workload is known:

```bash
--profile-workload both \
--warmup-steps 10 --num-steps 5 \
--prefill-input-len 4090 --prefill-output-len 1 \
--decode-input-len 1 --decode-output-len 2048
```

Allowed `--profile-workload` values:

- `both`: default; capture prefill and decode separately
- `prefill`: capture only the long-input / one-token workload
- `decode`: capture only the one-input / long-output workload
- `legacy`: keep the old `--probe-prompt` / `--probe-max-new-tokens` behavior

For `sglang-sota-performance`, do not use the defaults if the slow SGLang
benchmark scenario has a known input/output distribution.
Set the profiler lengths from that slow scenario instead: prefill uses the slow
input length with output `1`, and decode uses input `1` with the slow output
length. For a mixed dataset, profile the slowest representative bucket such as
the p50 or p95 input/output pair used in the benchmark report, and record the
bucket in the artifact notes.

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
  --output-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example/sglang_profile_live \
  --num-steps 5 \
  --warmup-steps 10 \
  --profile-by-stage \
  --profile-workload both
```

The script sends `POST /start_profile` to the SGLang server directly.
Keep `--output-dir` under `/data/...` so later analysis and docs can see the trace.
The script writes `server_args.json`, warms up with the same workload shape,
sends the active probe requests after profiling is armed, captures separate
`prefill/` and `decode/` profile roots by default, and waits longer for trace
flush than the earlier implementation.
For the default workload-separated capture, the directory name labels the stage
and the SGLang internal `profile_by_stage` mode is not used inside each
workload. This avoids mixing a one-token prefill probe with a separate decode
profile. The helper still adds one internal guard step because older SGLang
profilers check the target counter before running the next forward.

### 3. Single-trace live capture from vLLM

Launch vLLM with torch profiler enabled, for example:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/data/bbuf/validate/unified_llm_profiler_skill/runs/example/vllm_profile"}'
```

Then run:

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework vllm \
  --url http://127.0.0.1:8000 \
  --output-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example/vllm_profile \
  --num-steps 5 \
  --warmup-steps 10 \
  --no-profile-by-stage \
  --profile-workload both
```

For vLLM, `--output-dir` must point to the same `torch_profiler_dir` the server uses.
The current vLLM profiler config already defaults `torch_profiler_with_stack=true`,
so the runner only needs to set `torch_profiler_dir`.
On `h100_sglang`, external vLLM containers should mount both:

- `/data/.cache/huggingface:/root/.cache/huggingface`
- `/data/bbuf/validate/unified_llm_profiler_skill:/data/bbuf/validate/unified_llm_profiler_skill`

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
  --no-profile-by-stage \
  --profile-workload both
```

If the deployment does not expose the profiler control endpoints, fall back to analyzing
an existing trace instead of trying live capture.
If the TensorRT-LLM trace output is configured as one fixed file path, use
`scripts/run_trtllm_pytorch_profile_host.sh --stage prefill` and `--stage decode`
instead of direct `--profile-workload both`, so each stage gets its own trace file.

On the current TensorRT-LLM mainline path, `py_executor.py` creates the torch profiler
with `record_shapes=True` and `with_modules=True` but not `with_stack=True`.
For table-quality validation, use the override generator:

```bash
python3 scripts/make_trtllm_py_executor_override.py \
  --source /path/to/original/py_executor.py \
  --output /data/bbuf/validate/unified_llm_profiler_skill/overrides/trtllm/py_executor_with_stack.py
```

The matrix runner does this automatically on H100 before TensorRT-LLM capture starts.

This is the validated TensorRT-LLM flow on `h100_sglang`:

1. launch `trtllm-serve` with `TLLM_TORCH_PROFILE_TRACE=/data/.../trace.json`
2. run a few benchmark requests
3. analyze the emitted trace with `--input /data/.../trace.json`

### 5. Two-trace triage from existing profile dirs or traces

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --mapping-input /path/to/graph_off_profile_dir \
  --formal-input /path/to/graph_on_profile_dir
```

Use this when you need stronger overlap attribution and kernel-to-source mapping.

### 6. Two-trace triage from running servers

```bash
python3 scripts/analyze_llm_torch_profile.py \
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

- With `--profile-workload both` / `prefill` / `decode`, workload directories
  are the stage labels; the live-capture helper disables SGLang's internal
  stage profiler per workload, warms up first, and captures the requested
  active step count for the selected workload.
- On legacy or hand-captured SGLang serving, internal `profile_by_stage` is
  still useful because prefill and decode usually have very different
  bottlenecks.
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
4. Prefer `--profile-workload both`; use `legacy` only when reproducing an old trace contract.
5. Prefer workload-separated SGLang capture; use internal `--profile-by-stage`
   mainly for `legacy` or manually collected traces.
6. When on `h100_sglang`, create or clean the target trace directory through `docker exec sglang_bbuf ...` so the path is definitely writable under `/data`.

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
- [references/vllm-torch-compile-fusions.md](references/vllm-torch-compile-fusions.md)
  - current vLLM torch.compile fusion passes and the source patterns they target
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
