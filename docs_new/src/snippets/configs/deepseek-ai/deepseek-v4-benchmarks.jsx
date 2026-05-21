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

export const benchmarks = [
  // ====================================================================
  // B200 + FP4
  // ====================================================================
  { match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "low-latency",    nodes: "single" } },
  { match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "balanced",       nodes: "single" } },
  { match: { hw: "b200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" } },
  { match: { hw: "b200", variant: "pro",   quant: "fp4", strategy: "low-latency",    nodes: "single" } },
  { match: { hw: "b200", variant: "pro",   quant: "fp4", strategy: "balanced",       nodes: "single" } },
  { match: { hw: "b200", variant: "pro",   quant: "fp4", strategy: "max-throughput", nodes: "single" } },

  // ====================================================================
  // B300 + FP4
  // ====================================================================
  { match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "low-latency",    nodes: "single" } },
  { match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "balanced",       nodes: "single" } },
  { match: { hw: "b300", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" } },
  { match: { hw: "b300", variant: "pro",   quant: "fp4", strategy: "low-latency",    nodes: "single" } },
  { match: { hw: "b300", variant: "pro",   quant: "fp4", strategy: "balanced",       nodes: "single" } },
  { match: { hw: "b300", variant: "pro",   quant: "fp4", strategy: "max-throughput", nodes: "single" } },

  // ====================================================================
  // GB200 + FP4
  // ====================================================================
  { match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "low-latency",    nodes: "single"  } },
  { match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "balanced",       nodes: "single"  } },
  { match: { hw: "gb200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single"  } },
  { match: { hw: "gb200", variant: "pro",   quant: "fp4", strategy: "low-latency",    nodes: "multi-2" } },
  { match: { hw: "gb200", variant: "pro",   quant: "fp4", strategy: "balanced",       nodes: "multi-2" } },
  { match: { hw: "gb200", variant: "pro",   quant: "fp4", strategy: "max-throughput", nodes: "multi-2" } },

  // ====================================================================
  // GB300 + FP4
  // ====================================================================
  { match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "low-latency",    nodes: "single" } },
  { match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "balanced",       nodes: "single" } },
  { match: { hw: "gb300", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "pro",   quant: "fp4", strategy: "low-latency",    nodes: "single" } },
  { match: { hw: "gb300", variant: "pro",   quant: "fp4", strategy: "balanced",       nodes: "single" } },
  { match: { hw: "gb300", variant: "pro",   quant: "fp4", strategy: "max-throughput", nodes: "single" } },

  // ====================================================================
  // H200 + FP8
  // ====================================================================
  // Seed example: B200 flash fp4 low-latency single populated so the
  // card has something to render during initial review. Replace with
  // real measurements as they land. All other entries stay as
  // `{ match: {...} }` only and render the empty state.
  {
    match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "low-latency",    nodes: "single"  },
  },
  { match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "balanced",       nodes: "single"  } },
  { match: { hw: "h200", variant: "flash", quant: "fp8", strategy: "max-throughput", nodes: "single"  } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "low-latency",    nodes: "single"  } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "low-latency",    nodes: "multi-2" } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "balanced",       nodes: "multi-2" } },
  { match: { hw: "h200", variant: "pro",   quant: "fp8", strategy: "max-throughput", nodes: "multi-2" } },

  // ====================================================================
  // H200 + FP4
  // ====================================================================
  { match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "low-latency",    nodes: "single" } },
  { match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "balanced",       nodes: "single" } },
  { match: { hw: "h200", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "pro",   quant: "fp4", strategy: "low-latency",    nodes: "single" } },
  { match: { hw: "h200", variant: "pro",   quant: "fp4", strategy: "balanced",       nodes: "single" } },
  { match: { hw: "h200", variant: "pro",   quant: "fp4", strategy: "max-throughput", nodes: "single" } },

  // ====================================================================
  // H100 + FP4
  // ====================================================================
  // Seed example: filled in so reviewers can see the card layout end-to-
  // end on a real combination. Numbers here are PLACEHOLDERS — replace
  // with measurements before publishing.
  {
    match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "low-latency",    nodes: "single"  },
    sglang_version: "0.5.4-placeholder",
    latency:  {
      workload: "",  // e.g. "ShareGPT, in/out=1024/1024, bs=1" — fill in
      ttft_ms: 142, tpot_ms: 28, e2e_ms_p50: 820,
    },
    accuracy: { gsm8k_pct: 91.2, mmlu_pct: 87.4 },
    notes:    "Placeholder numbers — pending real measurement.",
  },
  { match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "balanced",       nodes: "single"  } },
  { match: { hw: "h100", variant: "flash", quant: "fp4", strategy: "max-throughput", nodes: "single"  } },
  { match: { hw: "h100", variant: "pro",   quant: "fp4", strategy: "low-latency",    nodes: "multi-2" } },
  { match: { hw: "h100", variant: "pro",   quant: "fp4", strategy: "balanced",       nodes: "multi-2" } },
  { match: { hw: "h100", variant: "pro",   quant: "fp4", strategy: "max-throughput", nodes: "multi-2" } },
];
