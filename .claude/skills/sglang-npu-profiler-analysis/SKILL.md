---
name: sglang-npu-profiler-analysis
description: "End-to-end Ascend NPU profiling workflow for SGLang: capture traces with torch_npu profiler, parse ASCEND_PROFILER_OUTPUT, and produce a standard bottleneck report. Use when tuning SGLang on Atlas 800T A2/A3, analyzing operator_details.csv / trace_view.json, or building NPU performance reports from bench_serving --profile output."
---

# SGLang NPU Profiler Analysis

## When to Use

- Ascend NPU inference bottleneck analysis for SGLang (`Atlas 800T A2` / `A3`)
- After collecting traces with `bench_serving --profile`, `/start_profile`, or `sglang.profiler`
- Before/after tuning graph mode, quantization, PD, speculative decoding, or MoE switches
- When you need a **repeatable report** from `operator_details.csv` / `kernel_details.csv`

Human-readable capture instructions live in [docs/.../optimization/profiling.mdx](../../../docs/docs/hardware-platforms/ascend-npus/optimization/profiling.mdx).

## Inputs

| Input | Required | Notes |
| --- | --- | --- |
| `trace_dir` | yes | Directory containing `*_ascend_pt/` or `ASCEND_PROFILER_OUTPUT/` |
| `model` | recommended | Model name for report header |
| `launch_cmd` | recommended | Exact `sglang serve` command |
| `workload` | optional | bench_serving dataset / concurrency description |
| `baseline_trace_dir` | optional | Second trace for before/after comparison |

## Outputs

Produce **one markdown report** with these sections (template: [references/report-template.md](references/report-template.md)):

1. Environment summary (hardware, CANN/TorchNPU, SGLang commit)
2. Capture method and parameters
3. Top operators table (from `operator_details.csv`)
4. Top kernels table (from `kernel_details.csv`, if present)
5. Bottleneck hypothesis (memory bound / compute bound / launch bound / comm bound)
6. Recommended next experiments (≤5 bullets)
7. Optional before/after delta table when `baseline_trace_dir` is provided

Optional JSON sidecar: `report.json` with the same tables for automation.

## Workflow

### Step 1 — Capture (production-like)

Preferred one-shot command on a running server:

```bash
export SGLANG_TORCH_PROFILER_DIR=./sglang_profile

python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /path/to/model \
  --tokenizer /path/to/model \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 8 \
  --profile \
  --profile-steps 5 \
  --profile-output-dir ./sglang_profile
```

Rules:

- Keep `--num-prompts` and output lengths small to avoid huge traces.
- Set `start_step` / `--profile-steps` to skip warmup (see profiling doc Method B).
- On NPU, `activities: ["CPU", "GPU"]` maps to NPU events via `torch_npu` patches.

### Step 2 — Locate parsed output

After capture completes, open the directory logged by the server:

`Profiling done. Traces are saved to: <path>`

Inside `<path>/*/ASCEND_PROFILER_OUTPUT/` expect:

| File | Use |
| --- | --- |
| `operator_details.csv` | Primary bottleneck table |
| `kernel_details.csv` | Kernel-level breakdown |
| `trace_view.json` | Timeline view in MindStudio Insight / Perfetto |
| `analysis.db` | Advanced DB queries |

See [references/npu-trace-layout.md](references/npu-trace-layout.md).

### Step 3 — Summarize CSVs

Run the helper script on the `ASCEND_PROFILER_OUTPUT` directory:

```bash
python .claude/skills/sglang-npu-profiler-analysis/scripts/summarize_npu_trace.py \
  --trace-dir ./sglang_profile/<timestamp>/<host>_*_ascend_pt/ASCEND_PROFILER_OUTPUT \
  --top 20 \
  --markdown-out ./npu_profile_report.md
```

If only raw `*_ascend_pt/` exists (interrupted capture), re-parse with:

```python
from torch_npu.profiler.profiler import analyse
analyse("./sglang_profile/<host>_*_ascend_pt/")
```

### Step 4 — Classify bottlenecks

Use this decision table:

| Signal in top operators | Likely bottleneck | Next knob |
| --- | --- | --- |
| Attention / FIA / MLA ops dominate | Compute or memory bandwidth | quant, MLAPO, page size, graph mode |
| AllGather / ReduceScatter / HCCL | Communication | TP/EP layout, DP attention, batch size |
| Small op fan-out + launch gaps | Launch overhead | `--cuda-graph-bs`, reduce capture sizes |
| Prefill-heavy timeline | Prefill scheduling | chunked prefill, PD split |
| Decode steady-state gaps | Decode batching | `--max-running-requests`, graph sizes |

### Step 5 — Graph-mode-specific profiling

| Goal | Server flag | Output |
| --- | --- | --- |
| Production-like decode with graphs | default / `--cuda-graph-bs ...` | normal `ASCEND_PROFILER_OUTPUT` |
| Python stack visibility | `--disable-cuda-graph` | clearer stacks, slower decode |
| Capture overhead debugging | `--enable-profile-cuda-graph` | `graph_capture_profile/` traces |

See [NPU Graph Mode Usage Guide](../../../docs/docs/hardware-platforms/ascend-npus/optimization/npu_graph_mode.mdx).

### Step 6 — Multi-hardware adaptation methodology

When porting this workflow off Ascend:

1. Keep the **same report schema** (sections 1–7 above).
2. Swap capture backend (CUDA Nsight, ROCm RPD, etc.) but preserve workload JSON.
3. Map operator CSV columns to the local profiler export format.
4. Record incompatible activities (NPU ignores `MEM`; ROCm `RPD` is unsupported on Ascend).
5. Store before/after artifacts under `out/<issue>/profiles/{baseline,tuned}/`.

For cross-framework torch-profiler triage on existing `trace.json`, also see `.claude/skills/llm-torch-profiler-analysis/`.

## Quality Gates

Before marking analysis complete:

- [ ] Trace directory path recorded in the report
- [ ] Capture command and server flags recorded
- [ ] Top operators table includes ≥10 rows or all rows if fewer
- [ ] At least one bottleneck hypothesis tied to evidence (operator name / time share)
- [ ] Recommendations are actionable SGLang flags or env vars (not generic advice)
- [ ] Graph-mode state documented (`enabled` / `disabled` / capture list)

## Anti-Patterns

- Do not compare traces captured with different graph-mode settings without labeling the delta.
- Do not use `--disable-cuda-graph` numbers as production TPOT baselines.
- Do not merge multi-node Ascend traces unless validated; prefer per-node `trace_view.json`.
- Do not run `analyse()` on already parsed trees unless you backed up `ASCEND_PROFILER_OUTPUT`.

## References

- [Capture guide (docs)](../../../docs/docs/hardware-platforms/ascend-npus/optimization/profiling.mdx)
- [Graph mode guide](../../../docs/docs/hardware-platforms/ascend-npus/optimization/npu_graph_mode.mdx)
- [Report template](references/report-template.md)
- [Trace layout](references/npu-trace-layout.md)
