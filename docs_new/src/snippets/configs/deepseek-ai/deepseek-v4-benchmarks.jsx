// DeepSeek-V4 cookbook benchmarks — paired with deepseek-v4.jsx + _deployment.jsx.
//
// One entry per cell in `config.cells`, keyed by the same `match` tuple
// (hw × variant × quant × strategy × nodes). When the user picks a
// combination in the Deploy panel, the engine looks up the entry whose
// `match` equals the current selection and renders it as a benchmark
// sub-card under the command box. Cells without any measured numbers
// render as the empty state ("Benchmark data pending …").
//
// Schema (everything except `match` is optional — present fields render,
// absent ones collapse out):
//
//   {
//     match:           { hw, variant, quant, strategy, nodes },     // REQUIRED
//     sglang_version:  string,                                       // shown in card header
//     latency:    { workload, ttft_ms, tpot_ms, e2e_ms_p50 },        // shown for low-latency / balanced
//     throughput: { workload, tokens_per_sec_per_gpu,                // shown for max-throughput / balanced
//                   p50_latency_ms, max_concurrency },
//     accuracy:   { gsm8k_pct, mmlu_pct },                           // shown when present
//     notes:      string,                                            // optional entry-level caveat
//     sweep:      [{ tpot_ms, throughput_per_gpu_tps,                // RESERVED for the future Pareto
//                    max_concurrency }, ...]                         // curve view — not rendered yet
//   }
//
// `workload` (per-block) describes WHAT was measured (e.g. "ShareGPT,
// in/out=1024/1024, bs=1" or "10 prompts at concurrency=1"). It renders
// as a small italic line under the block title, between the heading and
// the numeric rows. Leave it empty to defer; fill in when you know the
// exact harness command.
//
// Any field whose value is null renders as the empty state for that
// slot. A whole sub-block (e.g. `latency`) set to null or omitted
// collapses the corresponding column.
//
// Editing policy: this file turns over independently of the cell
// catalog (new sglang version → new numbers, but no flag changes).
// Keep the entry count and ordering in sync with deepseek-v4.jsx's
// `cells: [...]` so the two files diff cleanly side by side.

// Workload strings repeat across cells — every entry uses the same
// `sglang.bench_serving` invocation, only the numbers differ:
//   latency    = "random dataset, in/out=1024/1024, 10 prompts at concurrency=1"
//   throughput = "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100"
// Mintlify strips module-level statements, so we inline both literally.
//
// All numbers below were measured on **sglang v0.5.12**. When that version is
// updated, refresh the `sglang_version` field on every entry (or sweep it
// with a single find-and-replace).
export const benchmarks = [
  // ====================================================================
  // B200 + FP4
  // ====================================================================
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 86, tpot_ms: 3.42, e2e_ms_p50: 1174,
    },
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 3464, tpot_ms: 5.63, e2e_ms_p50: 5284,
    },
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 677, p50_latency_ms: 12867, max_concurrency: 100,
    },
  },
  {
    match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 1215, p50_latency_ms: 9521, max_concurrency: 100,
    },
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 171, tpot_ms: 5.12, e2e_ms_p50: 1800,
    },
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 183, tpot_ms: 8.49, e2e_ms_p50: 3407,
    },
    notes: "Throughput bench pending re-validation on this cell.",
  },
  {
    match: { hw: "b200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 372, p50_latency_ms: 15538, max_concurrency: 100,
    },
  },

  // ====================================================================
  // B300 + FP4
  // ====================================================================
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 84, tpot_ms: 3.33, e2e_ms_p50: 1153,
    },
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 157, tpot_ms: 5.59, e2e_ms_p50: 2039,
    },
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 1036, p50_latency_ms: 11167, max_concurrency: 100,
    },
  },
  {
    match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 1213, p50_latency_ms: 9473, max_concurrency: 100,
    },
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 115, tpot_ms: 5.04, e2e_ms_p50: 1768,
    },
    accuracy: { gsm8k_pct: 96.5, mmlu_pct: 88.5 },
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 1507, tpot_ms: 8.43, e2e_ms_p50: 4151,
    },
    notes: "Throughput bench pending re-validation on this cell.",
  },
  {
    match: { hw: "b300", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 359, p50_latency_ms: 16336, max_concurrency: 100,
    },
  },

  // ====================================================================
  // GB200 + FP4
  // ====================================================================
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 194, tpot_ms: 3.53, e2e_ms_p50: 1335,
    },
  },
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 335, tpot_ms: 6.18, e2e_ms_p50: 2449,
    },
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 614, p50_latency_ms: 19129, max_concurrency: 100,
    },
  },
  {
    match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 718, p50_latency_ms: 16051, max_concurrency: 100,
    },
  },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "multi-2" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 263, tpot_ms: 6.91, e2e_ms_p50: 2450,
    },
  },
  // GB200 Pro balanced: no cookbook recipe completed both lat+thp on 2
  // nodes — §3.1 cell carries the yellow Auto-Estimated badge, no benchmark
  // numbers written. Re-run once the deepep / megamoe-w4a8 bench finishes.
  { match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "multi-2" } },
  {
    match: { hw: "gb200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "multi-2" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 226, p50_latency_ms: 25926, max_concurrency: 100,
    },
  },

  // ====================================================================
  // GB300 + FP4
  // ====================================================================
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 226, tpot_ms: 4.30, e2e_ms_p50: 1617,
    },
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 5131, tpot_ms: 6.43, e2e_ms_p50: 7095,
    },
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 373, p50_latency_ms: 25094, max_concurrency: 100,
    },
  },
  {
    match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 682, p50_latency_ms: 17267, max_concurrency: 100,
    },
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 286, tpot_ms: 6.83, e2e_ms_p50: 2470,
    },
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 410, tpot_ms: 10.30, e2e_ms_p50: 3829,
    },
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 342, p50_latency_ms: 34191, max_concurrency: 100,
    },
  },
  {
    match: { hw: "gb300", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 378, p50_latency_ms: 31197, max_concurrency: 100,
    },
  },

  // ====================================================================
  // H200 + FP8
  // ====================================================================
  // FP8 sweep on H200 has not been benchmarked yet — all FP8 cells render
  // the pending-state empty card. Fill these in once the H200 FP8 deploy
  // (sgl-project repackaging on Hopper) is re-verified.
  { match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency",    nodes: "single"  } },
  { match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced",       nodes: "single"  } },
  { match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "max-throughput", nodes: "single"  } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "low-latency",    nodes: "single"  } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "low-latency",    nodes: "multi-2" } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "balanced",       nodes: "multi-2" } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "max-throughput", nodes: "multi-2" } },

  // ====================================================================
  // H200 + FP4
  //   Flash low-latency / max-throughput → marlin (cookbook)
  //   Flash balanced                     → flashinfer_mxfp4 (playground, +26% thp)
  //   Pro all 3 strategies               → flashinfer_mxfp4 (playground TP=4 sweep)
  // ====================================================================
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 132, tpot_ms: 3.48, e2e_ms_p50: 1274,
    },
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 139, tpot_ms: 4.71, e2e_ms_p50: 1748,
    },
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 746, p50_latency_ms: 15858, max_concurrency: 100,
    },
  },
  {
    match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 644, p50_latency_ms: 18329, max_concurrency: 100,
    },
  },
  // H200 Pro on FP4: numbers transcribed from the dsv4-benchmark playground
  // sweep (TP=4, flashinfer-mxfp4 runner). The cookbook command in §3.1
  // mirrors the same backend; cookbook bench re-run on TP=8 is the next
  // round of work.
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "low-latency", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 176, tpot_ms: 5.28, e2e_ms_p50: 1942,
    },
    accuracy: { gsm8k_pct: 97.5, mmlu_pct: 89.3 },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "balanced", nodes: "single" },
    sglang_version: "0.5.12",
    latency: {
      workload: "random dataset, in/out=1024/1024, 10 prompts at concurrency=1",
      ttft_ms: 190, tpot_ms: 7.47, e2e_ms_p50: 2721,
    },
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 270, p50_latency_ms: 44226, max_concurrency: 100,
    },
  },
  {
    match: { hw: "h200", variant: "pro", quant: "fp4", strategy: "max-throughput", nodes: "single" },
    sglang_version: "0.5.12",
    throughput: {
      workload: "random dataset, in/out=1024/1024, 1000 prompts at concurrency=100",
      tokens_per_sec_per_gpu: 288, p50_latency_ms: 41262, max_concurrency: 100,
    },
  },

  // ====================================================================
  // H100 + FP4 (all cells pending — H100 was not part of the v0.5.12 sweep)
  // ====================================================================
  { match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "low-latency",    nodes: "single"  } },
  { match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced",       nodes: "single"  } },
  { match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single"  } },
  { match: { hw: "h100", variant: "pro",   quant: "fp4", strategy: "low-latency",    nodes: "multi-2" } },
  { match: { hw: "h100", variant: "pro",   quant: "fp4", strategy: "balanced",       nodes: "multi-2" } },
  { match: { hw: "h100", variant: "pro",   quant: "fp4", strategy: "max-throughput", nodes: "multi-2" } },
];
