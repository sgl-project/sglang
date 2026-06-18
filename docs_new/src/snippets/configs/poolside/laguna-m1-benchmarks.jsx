// Laguna-M.1 benchmarks — one entry per cell `match` (same 5 keys as laguna-m1.jsx cells).
//
// All numbers below are REAL measured values; cells without measurements are bare `{ match }`
// pending stubs (the card renders "pending"). NO fabricated/dummy numbers remain.
// (cookbook_guide §3 forbids fabricated numbers in a published page.)
//
// FP8 is HOPPER-ONLY (not compatible with Blackwell) → the only FP8 entry is H200; there are no
// Blackwell FP8 entries. Blackwell cells are BF16 / NVFP4.
//
// REAL numbers (sgl-eval; GSM8K non-thinking; AIME via the enable_thinking wrapper, max_tokens=32768):
//   H200 BF16 (tp8): GSM8K 93.02% · AIME25 53.33% overall (~0.80 stop-only) · perf (cc1, cc128).
//   H200 FP8  (tp8): GSM8K 93.25% · AIME25 50.0% overall (~0.79 stop-only). g_proj FP8 fix validated; no perf (BF16-only scope).
//   B200 BF16 (tp8): GSM8K 91.88% · AIME25 66.88% (n_repeats=16) · perf A/B (cc1, cc128).
//   B200 NVFP4 (tp8): GSM8K 89.38%.
// ⚠️ AIME OVERALL is depressed by ~33–37% truncation at the 32k cap (M.1 reasoning is long → no boxed
// answer → scored 0); stop-only (~0.80) is the truer signal, and a 48–64k cap would lift the overall.
// (perf tokens_per_sec_per_gpu = measured output tok/s ÷ 8 GPUs; TTFT = median.)
//
// sglang_version reflects the REQUIRED build = PR #28400 (per-element gating) + PR #28604
// (global-attention SWA fix). The plain #28400 wheel (0.5.14.dev20260618+g343aeeef39) is NOT
// enough — it crashes M.1 under load. H200 numbers were taken on a main build @ 3f668733.

export const benchmarks = [
  // ===== H200 — BF16 / FP8 =====
  {
    // ✅ REAL — 8xH200, BF16, tp8. GSM8K 0.9302; perf bench_serving random 4096/1024.
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "main @ 3f668733 (#28400 + #28604)",
    speed: [
      // cc=1: median TTFT 81.89 ms, median TPOT 8.91 ms, output 109.96 tok/s (÷8 ≈ 13.7/GPU).
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 1 },
        ttft_ms: 81.9, tpot_ms: 8.91, tokens_per_sec_per_gpu: 13.7 },
      // cc=128: median TTFT 200.11 ms (mean 1221), median TPOT 52.09 ms, output 2266 tok/s (÷8 ≈ 283/GPU); total 11311 tok/s.
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 128 },
        ttft_ms: 200.1, tpot_ms: 52.1, tokens_per_sec_per_gpu: 283 },
    ],
    accuracy: { gsm8k_pct: 93.02, aime25_pct: 53.33 }, // AIME overall, n_repeats=1 (32k-truncation-limited, ~33% trunc; stop-only ~0.80)
  },
  {
    // ✅ REAL — 8xH200, FP8, tp8. GSM8K 93.25 + AIME25 0.50 (overall). The g_proj quant fix is
    // validated (FP8 now loads past layer 0). Perf not measured on FP8 (BF16-only scope) → no speed row.
    // FP8 is Hopper-only — not compatible with Blackwell (no Blackwell FP8 entry below).
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "main @ 3f668733 (#28400 + #28604 + g_proj FP8 fix)",
    accuracy: { gsm8k_pct: 93.25, aime25_pct: 50.0 }, // AIME overall (32k-truncation-limited; stop-only ~0.79)
  },

  // ===== B200 (8-GPU HGX) — BF16 / NVFP4 =====
  {
    // ✅ REAL — 8xB200, BF16, tp8. GSM8K 91.88 + AIME25 66.88; perf A/B (laguna-m1-results.md).
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "PR #28400 + #28604",
    speed: [
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 1 },
        ttft_ms: 108, tpot_ms: 9.0, tokens_per_sec_per_gpu: 13.6 },
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 128 },
        ttft_ms: 170, tpot_ms: 43.3, tokens_per_sec_per_gpu: 331 },
    ],
    accuracy: { gsm8k_pct: 91.88, aime25_pct: 66.88 },
  },
  {
    // ✅ REAL (GSM8K only) — 8xB200, NVFP4, tp8. GSM8K 89.38 (laguna-m1-results.md).
    // AIME + perf not measured yet → omitted (card shows them pending).
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "PR #28400 + #28604",
    accuracy: { gsm8k_pct: 89.38 },
  },

  // ===== B300 / GB200 / GB300 — BF16 / NVFP4, UNVERIFIED, no data yet → bare "pending" stubs
  // (no fabricated numbers; FP8 is Hopper-only so no Blackwell FP8 entries). =====
  { match: { hw: "b300",  variant: "default", quant: "bf16",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
];
