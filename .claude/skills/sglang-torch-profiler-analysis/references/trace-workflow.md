# Trace Workflow

This skill is based on SGLang's existing profiling workflow.

Use:

- two traces for `triage`
- one trace for `breakdown`
- two traces for `overlap`
- optional post-processing for `perfetto-fix`

`profile_by_stage` is still useful on normal non-PD serving because it separates prefill and decode. PD disaggregation adds an extra requirement beyond that: prefill workers and decode workers must be profiled separately.

## Existing SGLang Sources

- `sglang/.claude/skills/generate-profile/SKILL.md`
- `sglang/docs/developer_guide/benchmark_and_profiling.md`
- `sglang/docs/diffusion/performance/profiling.md`
- `sglang/python/sglang/profiler.py`
- `sglang/python/sglang/srt/utils/profile_utils.py`
- `sglang/python/sglang/srt/utils/profile_merger.py`

## Required Two-Stage Flow

### Stage 1: Mapping trace

Collect a graph-off trace first.

Recommended properties:

- disable `cuda graph`
- disable `piecewise cuda graph` if it would otherwise hide launch attribution
- keep the same model, parallel shape, backend choices, and request pattern as much as possible

Purpose:

- preserve clean kernel launch attribution
- recover `kernel -> cpu_op -> python scope`

### Stage 2: Formal trace

Collect a second trace with the real serving optimizations enabled.

Recommended properties:

- enable `cuda graph` if the real deployment uses it
- enable `piecewise cuda graph` when the model normally captures it
- keep the production MoE, attention, communication, and quantization backends

Purpose:

- measure real overlap under the real schedule
- decide whether a code path still has overlap headroom

The final compact report should be built from:

- source attribution from stage 1
- overlap conclusions from stage 2

The merged skill's `triage` command turns that into three tables:

- kernel table
- overlap-opportunity table
- fuse-opportunity table

## Ways To Produce A Trace

### Live server

```bash
python3 -m sglang.profiler --url http://127.0.0.1:30000 --num-steps 5
```

### One-shot request plus profile

```bash
python3 -m sglang.test.send_one --profile
```

### Bench serving

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang-profile
python3 -m sglang.bench_serving --backend sglang --num-prompts 10 --profile
```

If you only call `python3 -m sglang.profiler`, remember that something still has to drive requests through the server while profiling is active. The merged skill's live URL flows handle this automatically by sending a small probe workload.

## Expected Output

Typical trace outputs are:

- `<profile_id>-TP-0.trace.json.gz`
- `<profile_id>-TP-0-DP-0-PP-0-EP-0.trace.json.gz`
- `merged-<profile_id>.trace.json.gz`
- `server_args.json`

For overlap analysis, prefer a single-rank trace over a merged multi-rank trace.

## Optional Perfetto Repair

When Perfetto fails to render clearly overlapping events on the same logical lane, the unified script exposes:

```bash
python3 scripts/analyze_sglang_torch_profile.py perfetto-fix --input /path/to/trace.json.gz
```

This is intentionally a narrow repair step. Do not make it part of the default profiling workflow unless rendering is actually broken.

## Why The Mapping Trace Must Exist

A single graph-on trace may still tell you that overlap is poor, but it is often not enough to say which Python code path owns the kernel.

That is why this skill requires:

- one trace for readable source attribution
- one trace for real overlap behavior

Do not call the graph-off trace a "fast profile" in the final write-up. Its role is source mapping, not shortcutting the real analysis.
