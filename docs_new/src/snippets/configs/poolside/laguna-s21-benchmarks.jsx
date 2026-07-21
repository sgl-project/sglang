// Laguna-S-2.1 benchmarks — one entry per cell `match` (same 5 keys as laguna-s21.jsx cells).
//
// All entries below are PENDING stubs — no measurements yet. Fill in real numbers as cells
// are verified (set verified:true in laguna-s21.jsx and add accuracy + sglang_version here).
// The benchmark card renders "pending" for bare { match } stubs.
//
// Target eval set:
//   GSM8K  — sgl-eval run gsm8k, full 1319 questions, greedy/non-thinking.
//   AIME25 — sgl-eval run aime25, 30 problems × 16 repeats, temperature 1.0, top-p 0.95,
//             max-tokens 64000, 128 threads. Requires serving with enable_thinking=true
//             (Laguna's template gates on enable_thinking, not the generic 'thinking' key;
//             see Configuration Tips: Thinking in the cookbook page).

export const benchmarks = [
  // ===== H200 (8-GPU HGX, --tp 8) — PENDING =====
  { match: { hw: "h200", variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8",   strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "fp8",   strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "int4",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "h200", variant: "default", quant: "int4",  strategy: "low-latency",     nodes: "single" } },

  // ===== B300 (8-GPU HGX, --tp 8) — PENDING =====
  { match: { hw: "b300", variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "fp8",   strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "fp8",   strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "int4",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "int4",  strategy: "low-latency",     nodes: "single" } },

  // ===== GB300 (4-GPU single node, --tp 4) — PENDING =====
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8",   strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8",   strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "int4",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "int4",  strategy: "low-latency",     nodes: "single" } },
];
