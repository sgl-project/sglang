---
name: diffusion-perf
description: Measure and compare sglang-diffusion performance. Use when benchmarking a model, comparing before/after performance, or generating a perf report for a PR.
user-invocable: true
allowed-tools: Bash, Read
argument-hint: <model-path> [--prompt "..."] [--baseline baseline.json]
---

# Diffusion Performance Measurement

Measure sglang-diffusion e2e latency via `--perf-dump-path`, then extract or compare results from the JSON dump.

## JSON dump structure

`--perf-dump-path` writes a JSON file with:

```json
{
  "total_duration_ms": 14959.11,
  "steps": [
    {"name": "TextEncodingStage", "duration_ms": 611.83},
    {"name": "DenoisingStage", "duration_ms": 14289.46}
  ],
  "denoise_steps_ms": [
    {"step": 0, "duration_ms": 240.5},
    {"step": 1, "duration_ms": 279.1}
  ],
  "commit_hash": "abc123",
  "timestamp": "...",
  "memory_checkpoints": {}
}
```

Key fields:
- `total_duration_ms` — e2e walltime (warmup excluded when `--warmup` is used)
- `steps` — per-stage breakdown
- `denoise_steps_ms` — per denoising step timing

## Workflow

### 1. Single measurement

```bash
sglang generate --model-path $MODEL --prompt "$PROMPT" --warmup --perf-dump-path result.json
```

Then read `total_duration_ms` from `result.json`.

### 2. Before/after comparison

```bash
# Baseline (on main branch or before changes)
sglang generate --model-path $MODEL --prompt "$PROMPT" --warmup --perf-dump-path baseline.json

# New (after changes)
sglang generate --model-path $MODEL --prompt "$PROMPT" --warmup --perf-dump-path new.json

# Compare — outputs a Markdown table suitable for PR descriptions
python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json new.json
```

### 3. Extracting a single number

To get e2e latency in seconds from a dump:

```bash
python3 -c "import json; print(f\"{json.load(open('result.json'))['total_duration_ms']/1000:.2f}\")"
```

## Arguments

If `$ARGUMENTS` is provided, parse it as:
- First positional arg → `--model-path`
- `--prompt "..."` → generation prompt (default: `"A curious raccoon"`)
- `--baseline <file>` → if given, run comparison against this baseline file

## Notes

- Always use `--warmup` for accurate timing (excludes CUDA warmup from measurement).
- Keep `--prompt` and all server/sampling args identical between baseline and new runs.
- For PR descriptions, paste the output of `compare_perf.py` directly.
