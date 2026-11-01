// One entry per cell `match` tuple (same 5 keys as the config cells). A bare
// `match` stub renders as "pending". GSM8K: sgl-eval full 1319 (temp 1.0,
// top_p 0.95), measured through SGLang on H200 — 93.63% with 100% stop rate,
// 0 truncated, 1319/1319 completions.

export const benchmarks = [
  {
    match: { hw: "gb300", variant: "1.7b", quant: "bf16", strategy: "default", nodes: "single" },
    accuracy: { gsm8k_pct: 94.47 },
    sglang_version: "main@80ca262b3e (pre-release)",
  },
  // ====================================================================
  // H200
  // ====================================================================
  {
    match: { hw: "h200", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
    sglang_version: "3efb189b93 (pre-release)",
    accuracy: { gsm8k_pct: 93.63 },
  },
  {
    match: { hw: "h200", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
  },

  // ====================================================================
  // H100 — vendor-tested (FP16); no measurements of our own yet
  // ====================================================================
  {
    match: { hw: "h100", variant: "1.7b", quant: "bf16", strategy: "balanced", nodes: "single" },
  },
  {
    match: { hw: "h100", variant: "vl", quant: "bf16", strategy: "balanced", nodes: "single" },
  },
];
