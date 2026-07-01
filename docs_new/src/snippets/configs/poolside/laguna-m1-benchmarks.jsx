// Laguna-M.1 benchmarks — one entry per cell `match` (same 5 keys as laguna-m1.jsx cells).
//
// All numbers below are REAL measured values; cells without measurements are bare `{ match }`
// pending stubs (the card renders "pending"). NO fabricated/dummy numbers.
// Accuracy axis is GSM8K-only for now (AIME 25 will be re-added once truncation-free numbers exist).
//
// REAL GSM8K (sgl-eval `run gsm8k`, full 1319, non-thinking):
//   H200 BF16 (tp8): 93.02%  · perf bench_serving random 4096/1024 (cc1, cc128).
//   H200 FP8  (tp8): 93.25%.
//   B200 BF16 (tp8): 91.88%  · perf A/B (cc1, cc128).
//   B200 FP8  (tp8): 93.78%  — with `--fp8-gemm-backend triton` (DeepGEMM UE8M0 workaround; ~19% slower).
//   B200 NVFP4 (tp8): 89.38%.
// (perf tokens_per_sec_per_gpu = measured output tok/s ÷ 8 GPUs; TTFT = median.)
//
// sglang_version = the build the numbers ran on (PR #28400 + #28604, +#28649 for FP8 load,
// +#28662/triton-workaround for Blackwell FP8). H200 numbers taken on a main build @ 3f668733.

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
      // cc=128: median TTFT 200.11 ms (mean 1221), median TPOT 52.09 ms, output 2266 tok/s (÷8 ≈ 283/GPU).
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 128 },
        ttft_ms: 200.1, tpot_ms: 52.1, tokens_per_sec_per_gpu: 283 },
    ],
    accuracy: { gsm8k_pct: 93.02 },
  },
  {
    // ✅ REAL — 8xH200, FP8, tp8. GSM8K 93.25%. (Hopper: no --fp8-gemm-backend flag needed.)
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "main @ 3f668733 (#28400 + #28604 + g_proj FP8 fix #28649)",
    accuracy: { gsm8k_pct: 93.25 },
  },

  // ===== B200 (8-GPU HGX) — BF16 / FP8 / NVFP4 =====
  {
    // ✅ REAL — 8xB200, BF16, tp8. GSM8K 91.88; perf A/B (laguna-m1-results.md).
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "PR #28400 + #28604",
    speed: [
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 1 },
        ttft_ms: 108, tpot_ms: 9.0, tokens_per_sec_per_gpu: 13.6 },
      { workload: { dataset: "random", isl: 4096, osl: 1024, max_concurrency: 128 },
        ttft_ms: 170, tpot_ms: 43.3, tokens_per_sec_per_gpu: 331 },
    ],
    accuracy: { gsm8k_pct: 91.88 },
  },
  {
    // ✅ REAL — 8xB200, FP8, tp8, with --fp8-gemm-backend triton. GSM8K 93.78% (full 1319,
    // laguna-m1-results.md). Matches H200 FP8 (93.25) within noise; sits above B200 NVFP4 (89.38).
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "main + #28649 + --fp8-gemm-backend triton (DeepGEMM UE8M0 workaround; fix = PR #28662)",
    accuracy: { gsm8k_pct: 93.78 },
  },
  {
    // ✅ REAL — 8xB200, NVFP4, tp8. GSM8K 89.38% (laguna-m1-results.md).
    match: { hw: "b200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" },
    verified: true,
    sglang_version: "PR #28400 + #28604",
    accuracy: { gsm8k_pct: 89.38 },
  },

  // ===== B300 / GB200 / GB300 — BF16 / FP8 / NVFP4, UNVERIFIED → bare "pending" stubs (no
  // fabricated numbers). Blackwell FP8 cells carry --fp8-gemm-backend triton in the config. =====
  { match: { hw: "b300",  variant: "default", quant: "bf16",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "fp8",   strategy: "balanced", nodes: "single" } },
  { match: { hw: "b300",  variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "bf16",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "fp8",   strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb200", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16",  strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8",   strategy: "balanced", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "nvfp4", strategy: "balanced", nodes: "single" } },
];
