// Ling-3.0-flash per-cell benchmark numbers, keyed by the same `match` tuple as
// ling-3.0-flash.jsx cells. See _deployment.jsx for the speed/accuracy schema.
//
// Accuracy uses sgl-eval full GSM8K (1319 questions). Speed uses 80 exact-length
// random requests (ISL 8192 / OSL 1024, --random-range-ratio 1, --flush-cache).
// TTFT/TPOT are P50; tokens_per_sec_per_gpu is total (input + output) tok/s/GPU.
//
// Cells with no entry (H20-3e / H800 / H100, both quantizations) had no matching
// allocation and were never gated.
export const benchmarks = [
  // ====================================================================
  // H200 + BF16 (TP4)
  // ====================================================================
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #33561 @ c5071ded",
    accuracy: { gsm8k_pct: 96.59 },
  },
  {
    // Rejected by the full GSM8K gate at request 1319: a no-EOS runaway generated
    // >33k tokens. Recipe stays `verified: false` in ling-3.0-flash.jsx.
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    accuracy: { gsm8k_pct: null },
    notes: "Full GSM8K gate did not complete: one request ran away without emitting EOS (>33k generated tokens).",
  },

  // ====================================================================
  // H200 + FP8 (TP4 + EP4)
  // ====================================================================
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #33561 @ e57e030b",
    accuracy: { gsm8k_pct: 95.83 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ e57e030b",
    accuracy: { gsm8k_pct: 96.51 },
  },
  {
    match: { hw: "h200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ e1a24a18",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 205.96, tpot_ms: 3.68, tokens_per_sec_per_gpu: 1159 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1891.74, tpot_ms: 8.16, tokens_per_sec_per_gpu: 7184 },
    ],
    accuracy: { gsm8k_pct: 95.30 },
    notes: "Full GSM8K stop rate 100%; default decode CUDA Graph captured 32 shapes through batch 218.",
  },
  {
    match: { hw: "h200", variant: "default", quant: "mxfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ e1a24a18",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 305.23, tpot_ms: 6.05, tokens_per_sec_per_gpu: 709 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 2725.52, tpot_ms: 11.17, tokens_per_sec_per_gpu: 5215 },
    ],
    accuracy: { gsm8k_pct: 96.29 },
    notes: "Full GSM8K stop rate 100%; default decode CUDA Graph captured 33 shapes through batch 227.",
  },

  // ====================================================================
  // H200 + HiCache (Mooncake tiered cache)
  // ====================================================================
  {
    match: { hw: "h200", variant: "default", quant: "bf16", strategy: "hicache", nodes: "single" },
    sglang_version: "PR #33561 @ 51bcd89c",
    accuracy: { gsm8k_pct: 96.44 },
  },

  // ====================================================================
  // B200 + BF16 (TP4)
  // ====================================================================
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #33561 @ c5071ded",
    accuracy: { gsm8k_pct: 96.44 },
  },
  {
    match: { hw: "b200", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ c5071ded",
    accuracy: { gsm8k_pct: 96.51 },
  },

  // ====================================================================
  // B200 + FP8 (TP4 + EP4)
  // ====================================================================
  {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" },
    sglang_version: "PR #33561 @ e57e030b",
    accuracy: { gsm8k_pct: 96.59 },
  },
  {
    match: { hw: "b200", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ e57e030b",
    accuracy: { gsm8k_pct: 97.04 },
  },
  {
    match: { hw: "b200", variant: "default", quant: "int4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ e1a24a18",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 641.41, tpot_ms: 8.72, tokens_per_sec_per_gpu: 482 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 6202.62, tpot_ms: 48.01, tokens_per_sec_per_gpu: 1344 },
    ],
    accuracy: { gsm8k_pct: 96.74 },
    notes: "Full GSM8K stop rate 99.70%; default decode CUDA Graph captured 40 shapes through batch 305.",
  },
  {
    match: { hw: "b200", variant: "default", quant: "mxfp4", strategy: "high-throughput", nodes: "single" },
    sglang_version: "PR #33561 @ e1a24a18",
    speed: [
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 1 },
        ttft_ms: 127.97, tpot_ms: 5.60, tokens_per_sec_per_gpu: 785 },
      { workload: { dataset: "random", isl: 8192, osl: 1024, max_concurrency: 16 },
        ttft_ms: 1117.16, tpot_ms: 7.98, tokens_per_sec_per_gpu: 7933 },
    ],
    accuracy: { gsm8k_pct: 96.29 },
    notes: "Full GSM8K stop rate 99.62%; default decode CUDA Graph captured 40 shapes through batch 314.",
  },

  // ====================================================================
  // GB300 + BF16 (TP4)
  // ====================================================================
  // TODO: both cells are `verified: true` (gated on the final head) but the GSM8K
  // percentages were not recorded in the PR body or in the verifying commits —
  // fill from the run logs.
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "bf16", strategy: "high-throughput", nodes: "single" } },

  // ====================================================================
  // GB300 + FP8 (TP4 + EP4)
  // ====================================================================
  // TODO: both cells are `verified: true` on the final head; the 96.66% / 96.44%
  // pair in the PR body predates the TP+EP change — fill with the re-measured values.
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "low-latency", nodes: "single" } },
  { match: { hw: "gb300", variant: "default", quant: "fp8", strategy: "high-throughput", nodes: "single" } },
];
