// Laguna-S-2.1 benchmarks — one entry per cell `match` (same 5 keys as laguna-s21.jsx cells).
//
// H200 cells: REAL measured values (sglang 0.5.15.post1, lmsysorg/sglang:latest,
//   8×H200 TP=8, 2026-07-20). GSM8K = sgl-eval run gsm8k, full 1319 questions, greedy.
//   AIME25 = sgl-eval run aime25, 30 problems × 16 repeats, temp 1.0, top_p 0.95,
//   max_tokens 64000, 128 threads, thinking ON via enable_thinking (not the generic
//   'thinking' key). AIME25 measured on high-throughput cells only.
//
// BF16 AIME25 invalidity: BF16 reasons ~2× longer than FP8/INT4 (median 34.8k vs
//   16.9k tokens); 33.8% of responses truncated at max_tokens=64000. aime25_pct is
//   left null — re-run with max_tokens ≥ 128k for a valid score.
//
// B300/GB300 cells: PENDING (bare match stubs — benchmark card renders "pending").

export const benchmarks = [
  // ===== H200 (8-GPU HGX, tp 8) — ✅ REAL, full GSM8K =====
  {
    // ✅ REAL — 8×H200, BF16 dense, tp8, fa3 (Hopper auto-select), mem-frac 0.80.
    // GSM8K 93.18%. AIME25 pending re-run with higher max_tokens (truncation at 64k).
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.18, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×H200, BF16 + DFlash (matched bf16 draft), tp8, fa3, mem-frac 0.7.
    // GSM8K 93.33%. AIME25 not measured on low-latency cells.
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.33, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×H200, FP8 dense, tp8, fa3, SGLANG_SHARED_EXPERT_TP1=1.
    // GSM8K 94.24%, AIME25 67.29% pass@1 (avg-16, pass@16 86.7%, majority@16 76.7%).
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.24, aime25_pct: 67.29 },
  },
  {
    // ✅ REAL — 8×H200, FP8 + DFlash (fp8-calibrated draft, patched rope_theta), tp8, fa3.
    // GSM8K 94.47%. AIME25 not measured on low-latency cells.
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.47, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×H200, INT4 dense, tp8, fa3. INT4 shared expert stays bf16 (no flag).
    // GSM8K 95.00%, AIME25 67.71% pass@1 (avg-16, pass@16 86.7%, majority@16 80.0%).
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 95.00, aime25_pct: 67.71 },
  },
  {
    // ✅ REAL — 8×H200, INT4 + DFlash (int4-calibrated draft), tp8, fa3.
    // GSM8K 94.24%. AIME25 not measured on low-latency cells.
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.24, aime25_pct: null },
  },

  // ===== B300 (8-GPU HGX, tp 8) — PENDING =====
  { match: { hw: "b300", variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "fp8",   strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "fp8",   strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "int4",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "b300", variant: "default", quant: "int4",  strategy: "low-latency",     nodes: "single" } },

  // ===== GB300 (4-GPU single node, tp 4) — PENDING =====
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8",   strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8",   strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "int4",  strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "int4",  strategy: "low-latency",     nodes: "single" } },
];
