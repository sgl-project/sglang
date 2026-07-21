// Laguna-S-2.1 benchmarks — one entry per cell `match` (same 5 keys as laguna-s21.jsx cells).
//
// H200 cells: REAL measured values (sglang 0.5.15.post1, lmsysorg/sglang:latest,
//   8×H200 TP=8, 2026-07-20). GSM8K = sgl-eval run gsm8k, full 1319 questions, greedy.
//   AIME25 = sgl-eval run aime25, 30 problems × 16 repeats, temp 1.0, top_p 0.95,
//   max_tokens 64000, 128 threads, thinking ON via patched sgl-eval (enable_thinking).
//   AIME25 measured on high-throughput cells only.
//   BF16 AIME25 null: BF16 reasons ~2× longer than FP8/INT4 (median 34.8k vs 16.9k tokens),
//   truncating at 33.8% at max_tokens=64000. Valid score requires max_tokens ≥ 131072.
//
// GB300 cells (BF16/FP8/INT4): REAL measured values (sglang dev/custom build,
//   4×GB300 TP=4, 2026-07-20). GSM8K = sgl-eval run gsm8k, full 1319 questions, greedy.
//   AIME25 not measured on GB300: the sgl-eval `--thinking` flag sends the generic
//   `thinking` key which Laguna's template ignores (it requires `enable_thinking`);
//   without the sgl-eval patch applied on H200, thinking was not activated — the
//   fp8_ht result (13.54%) is therefore invalid and omitted.
//   GB300 NVFP4: quality broken (degenerate loop) — no accuracy numbers.
//
// B300 cells: REAL measured values (sglang 0.5.15.post1, lmsysorg/sglang:v0.5.15.post1-cu130,
//   8×B300 TP=8, 2026-07-21). GSM8K = sgl-eval run gsm8k, full 1319 questions, greedy.
//   AIME25 not measured on B300.

export const benchmarks = [
  // ===== H200 (8-GPU HGX, tp 8) — ✅ REAL, full GSM8K =====
  {
    // ✅ REAL — 8×H200, BF16 dense, tp8, fa3, mem-frac 0.80. GSM8K 93.18%.
    // AIME25 null: truncation at 64k (BF16 reasons ~2× longer; needs max_tokens ≥ 131072).
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
    // ✅ REAL — 8×H200, INT4 dense, tp8, fa3. INT4 shared expert stays bf16.
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

  // ===== B300 (8-GPU HGX, tp 8) — ✅ REAL, full GSM8K =====
  {
    // ✅ REAL — 8×B300, BF16 dense, tp8, trtllm_mha (auto-select). GSM8K 93.71%.
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.71, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×B300, BF16 + DFlash (matched bf16 draft), tp8, trtllm_mha, mem-frac 0.7.
    // GSM8K 93.63%.
    match: { hw: "b300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 93.63, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×B300, FP8 dense, tp8 ep8, trtllm_mha, SGLANG_SHARED_EXPERT_TP1=1.
    // GSM8K 94.39%.
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.39, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×B300, FP8 + DFlash (fp8-calibrated draft, patched rope_theta), tp8 ep8,
    // trtllm_mha, mem-frac 0.7. GSM8K 94.69%.
    match: { hw: "b300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×B300, NVFP4 dense, tp8, trtllm_mha (auto-select). GSM8K 94.54%.
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.54, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×B300, NVFP4 + DFlash (nvfp4-calibrated draft, patched rope_theta), tp8,
    // trtllm_mha, mem-frac 0.7. GSM8K 95.30%.
    match: { hw: "b300", variant: "default", quant: "nvfp4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 95.30, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×B300, INT4 dense, tp8 ep8, trtllm_mha. GSM8K 94.62%.
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.62, aime25_pct: null },
  },
  {
    // ✅ REAL — 8×B300, INT4 + DFlash (int4-calibrated draft), tp8 ep8, trtllm_mha,
    // mem-frac 0.7. GSM8K 94.69%.
    match: { hw: "b300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "0.5.15.post1",
    accuracy: { gsm8k_pct: 94.69, aime25_pct: null },
  },

  // ===== GB300 (4-GPU single node, tp 4) =====
  {
    // ✅ REAL — 4×GB300, BF16 dense, tp4, trtllm_mha (auto-select).
    // GSM8K 93.33%. AIME25 not measured (enable_thinking patch not applied on GB300).
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 93.33, aime25_pct: null },
  },
  {
    // ✅ REAL — 4×GB300, BF16 + DFlash (matched bf16 draft), tp4, trtllm_mha.
    // GSM8K 93.86%. AIME25 not measured.
    match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 93.86, aime25_pct: null },
  },
  {
    // ✅ REAL — 4×GB300, FP8 dense, tp4, trtllm_mha, SGLANG_SHARED_EXPERT_TP1=1.
    // GSM8K 94.77%. AIME25 not measured (enable_thinking patch not applied on GB300).
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 94.77, aime25_pct: null },
  },
  {
    // ✅ REAL — 4×GB300, FP8 + DFlash (fp8-calibrated draft, patched rope_theta),
    // tp4, trtllm_mha, SGLANG_SHARED_EXPERT_TP1=1. GSM8K 94.54%.
    match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 94.54, aime25_pct: null },
  },
  // NVFP4: quality broken on GB300 (degenerate loop) — no accuracy numbers, pending stubs.
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "high-throughput", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "low-latency",     nodes: "single" } },
  {
    // ✅ REAL — 4×GB300, INT4 dense, tp4, trtllm_mha. INT4 shared expert stays bf16.
    // GSM8K 94.77%. AIME25 skipped per user decision.
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 94.77, aime25_pct: null },
  },
  {
    // ✅ REAL — 4×GB300, INT4 + DFlash (int4-calibrated draft), tp4, trtllm_mha.
    // GSM8K 95.15%.
    match: { hw: "gb300", variant: "default", quant: "int4", strategy: "low-latency", nodes: "single" },
    verified: true,
    sglang_version: "dev/custom build (2026-07-20)",
    accuracy: { gsm8k_pct: 95.15, aime25_pct: null },
  },
];
