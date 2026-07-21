export const benchmarks = [

  // ── H200 (8×H200, tp 8, sglang 0.5.15.post1) ──

  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    // BF16 reasons ~2× longer than FP8/INT4 (median 34.8k vs 16.9k tokens);
    // truncation at max_tokens=64000 invalidates the result. Needs max_tokens ≥ 131072.
    accuracy: { gsm8k_pct: 93.18, aime25_pct: null },
  },
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.33, aime25_pct: null },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.24, aime25_pct: 67.29 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.47, aime25_pct: null },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 95.00, aime25_pct: 67.71 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.24, aime25_pct: null },
  },

  // ── B300 (8×B300, tp 8, sglang 0.5.15.post1) ──

  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.71, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.63, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.39, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.54, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 95.30, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.62, aime25_pct: null },
  },
  {
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },

  // ── GB300 (4×GB300, tp 4, sglang dev build 2026-07-20) ──
  // AIME25 null: sgl-eval --thinking sends the generic `thinking` key; Laguna's template
  // requires `enable_thinking`. Scores would reflect thinking-off behavior — omitted.

  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 93.33, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 93.86, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 94.77, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 94.54, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 94.77, aime25_pct: null },
  },
  {
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 95.15, aime25_pct: null },
  },
];
